import os 
import math
import glob

import PIL.Image as Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch
# import kaolin as kal
# import nvdiffrast
# from load_obj import load_obj
from torch.utils.data import DataLoader, Dataset, IterableDataset
# import multiprocessing as mp
import cv2
# Set the multiprocessing start method to 'spawn'
# mp.set_start_method('spawn')
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import imutils

class TrainDiffDataset(Dataset):
    def __init__(self, args,root, device, uv_template, size=512):
        self.root = root
        self.img_size = size

        self.device=device
        self.args=args

        # RGBA
        self.subject_list = self.get_subjects(self.root)
        self.num_subjects = len(self.subject_list)

        self.transform= transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

        self.transform_rgba= transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

        self.transform_rgba_mask= transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),  ])


        self.transform_clip = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                 [0.26862954, 0.26130258, 0.27577711]),
        ])

        self.transform_clip_mask = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

    def augment_random_background(self, image, mask):
    # Random background
        bg_color = torch.rand(3) * 2 - 1
        bg = torch.ones_like(image) * bg_color.view(3,1,1)
        return image * mask + bg * (1 - mask)

    def get_subjects(self, data_dir):
        paths = glob.glob(f'{data_dir}/*/f*/*.color.jpg')
        return paths

    def __getitem__(self, index):

        image_path = self.subject_list[index]
        mask_path = image_path.replace('.color.jpg', '.obj_rend_full.png')
        partial_mask_path = image_path.replace('.color.jpg', '.obj_mask_m.jpg')

        image_np = np.array(Image.open(image_path).convert('RGB'))
        tgt_image_np = np.array(Image.open(mask_path).convert('RGB'))
        mask_np = np.array(Image.open(mask_path).split()[-1])
        partial_mask_np = np.array(Image.open(partial_mask_path).convert('L'))
        mask_diff_np = mask_np - partial_mask_np
        mask_diff_np[mask_diff_np < 0] = 0

        crop_resize_fn, M_crop = imutils.crop_and_resize_bbox(self.img_size, mask=[mask_np])
        image_crop = crop_resize_fn(image_np)
        tgt_image_crop = crop_resize_fn(tgt_image_np)
        mask_crop = crop_resize_fn(mask_np)
        partial_mask_crop = crop_resize_fn(partial_mask_np)
        mask_diff_crop = crop_resize_fn(mask_diff_np)

        image_rgb = self.transform_rgba(image_crop)
        mask_rgb = self.transform_rgba_mask(partial_mask_crop)
        src_image = image_rgb * mask_rgb
        print('src_image:',src_image.shape)

        rgb_clip = self.transform_clip(image_crop)
        mask_clip = self.transform_clip_mask(partial_mask_crop)
        src_clip_image = rgb_clip * mask_clip
        print('src_clip_image:', src_clip_image.shape)

        image_rgb_tgt = self.transform_rgba(tgt_image_crop)
        mask_rgb_tgt = self.transform_rgba_mask(mask_crop)
        target = image_rgb_tgt * mask_rgb_tgt
        print('target:', target.shape)

        mask_rgb_diff = self.transform_rgba_mask(mask_diff_crop)
        print('mask_rgb_diff:', mask_rgb_diff.shape)

        # view condition is always the same for back images
        view_cond = torch.stack(
            [   torch.tensor(0.0),
                torch.sin(torch.tensor(math.pi)),
                torch.cos(torch.tensor(math.pi)),
                torch.tensor(0.0)] ).view(-1,1,1).repeat(1, self.img_size, self.img_size)

        return {'src_ori_image': src_image,
                'src_image': src_clip_image,
                'target_img': target,
                'inpaint_mask': mask_rgb_diff,
                # 'tgt_uv':  tgt_uv.permute(2,0,1) * 2. - 1,
                'view_cond': view_cond,
                'filename': self.subject_list[index].split('/')[-3]+'/' +self.subject_list[index].split('/')[-2]+'/'+self.subject_list[index].split('/')[-1]
        }

    def __len__(self):
        return self.num_subjects