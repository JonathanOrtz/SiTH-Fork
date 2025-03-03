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

from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from PIL.ImageFilter import GaussianBlur
import torch
import kornia
from kornia.geometry.transform import get_affine_matrix2d, warp_affine
import cv2
def flip_image(img):
    img = Image.fromarray(img)
    flipped = transforms.RandomHorizontalFlip(p=1.0)(img)
    img = np.array(flipped)
    return img

def masks2bbox(masks, thres=127):
    """
    convert a list of masks to an bbox of format xyxy
    :param masks:
    :param thres:
    :return:
    """
    mask_comb = np.zeros_like(masks[0])
    for m in masks:
        mask_comb += m
    mask_comb = np.clip(mask_comb, 0, 255)
    # print(mask_comb)
    ret, threshed_img = cv2.threshold(mask_comb, thres, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bmin, bmax = np.array([50000, 50000]), np.array([-100, -100])
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        bmin = np.minimum(bmin, np.array([x, y]))
        bmax = np.maximum(bmax, np.array([x + w, y + h]))
    return bmin, bmax

def blur_image(img, aug_blur=0):
    assert isinstance(img, np.ndarray)
    if aug_blur > 0.000001:
        x = np.random.uniform(0, aug_blur)*255. # input image is in range [0, 255]
        blur = GaussianBlur(x)
        img = Image.fromarray(img)
        return np.array(img.filter(blur))
    return img

def crop(img, center, crop_size):
    """
    crop image around the given center, pad zeros for boraders
    :param img:
    :param center:
    :param crop_size: size of the resulting crop
    :return:
    """
    assert isinstance(img, np.ndarray)
    h, w = img.shape[:2]
    topleft = np.round(center - crop_size / 2).astype(int)
    bottom_right = np.round(center + crop_size / 2).astype(int)

    x1 = max(0, topleft[0])
    y1 = max(0, topleft[1])
    x2 = min(w - 1, bottom_right[0])
    y2 = min(h - 1, bottom_right[1])
    cropped = img[y1:y2, x1:x2]

    p1 = max(0, -topleft[0])  # padding in x, top
    p2 = max(0, -topleft[1])  # padding in y, top
    p3 = max(0, bottom_right[0] - w+1)  # padding in x, bottom
    p4 = max(0, bottom_right[1] - h+1)  # padding in y, bottom

    dim = len(img.shape)
    if dim == 3:
        padded = np.pad(cropped, [[p2, p4], [p1, p3], [0, 0]])
    elif dim == 2:
        padded = np.pad(cropped, [[p2, p4], [p1, p3]])
    else:
        raise NotImplemented
    return padded

def resize(img, img_size, mode=cv2.INTER_LINEAR):
    """
    resize image to the input
    :param img:
    :param img_size: (width, height) of the target image size
    :param mode:
    :return:
    """
    h, w = img.shape[:2]
    load_ratio = 1.0 * w / h
    netin_ratio = 1.0 * img_size[0] / img_size[1]
    assert load_ratio == netin_ratio, "image aspect ration not matching, given image: {}, net input: {}".format(img.shape, img_size)
    resized = cv2.resize(img, img_size, interpolation=mode)
    return resized

def get_affine_matrix_box(boxes, w2, h2):
    # boxes [left, top, right, bottom]
    width = boxes[:, 2] - boxes[:, 0]  # (N,)
    height = boxes[:, 3] - boxes[:, 1]  # (N,)
    center = torch.tensor([(boxes[:, 0] + boxes[:, 2]) / 2.0,
                            (boxes[:, 1] + boxes[:, 3]) / 2.0]).T.to(torch.float32)  # (N,2)
    scale = torch.min(torch.tensor([w2 / width, h2 / height]),
                        dim=0)[0].unsqueeze(1).repeat(1, 2) * 0.9  # (N,2)
    transl = torch.cat([w2 / 2.0 - center[:, 0:1], h2 / 2.0 - center[:, 1:2]], dim=1).to(torch.float32)  # (N,2)
    M = get_affine_matrix2d(transl, center, scale.to(torch.float32), angle=torch.tensor([
                                                                            0.,
                                                                        ] * transl.shape[0]))

    return M
def crop_and_resize_bbox(input_res, mask=None, bbox=None):
    if mask is not None:
        bmin, bmax = masks2bbox(mask)
        bbox = np.expand_dims(np.hstack([bmin, bmax]), 0)
    M_crop = get_affine_matrix_box(bbox, input_res, input_res)
    def process_image(image):
        is_rbg = len(image.shape) == 3
        img_ori = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() if is_rbg \
            else torch.tensor(image).unsqueeze(0).unsqueeze(0).float()
        img_crop = warp_affine(
            img_ori,
            M_crop[0:0 + 1, :2], (input_res,) * 2,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        img_crop = img_crop.squeeze(0).permute(1, 2, 0) if is_rbg else img_crop.squeeze(0).squeeze(0)
        img_crop = img_crop.numpy().astype(np.uint8)
        return img_crop
    return process_image, M_crop

def crop_bbox_from_PIL(image, bbox):
    """
    Crops PIL image using bounding box.

    Args:
        image (PIL.Image): Image to be cropped.
        bbox (tuple): Integer bounding box (xyxy).
    """
    bbox = np.array(bbox)
    if image.mode == "RGB":
        default = (255, 255, 255)
    elif image.mode == "RGBA":
        default = (255, 255, 255, 255)
    else:
        default = 0
    bg = Image.new(image.mode, (bbox[2] - bbox[0], bbox[3] - bbox[1]), default)
    bg.paste(image, (-bbox[0], -bbox[1]))
    return bg

class TrainDiffDataset(Dataset):
    def __init__(self, args,root, device, size=512):
        self.root = root
        self.img_size = size

        self.device=device
        self.args=args

        # RGBA
        self.subject_list = self.get_subjects(self.root)
        self.num_subjects = len(self.subject_list)

        self.transform= transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

        self.transform_rgba= transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

        self.transform_rgba_mask= transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR)
        ])


        self.transform_clip = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                                 [0.26862954, 0.26130258, 0.27577711]),
        ])

        self.transform_clip_mask = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
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

        crop_resize_fn, M_crop = crop_and_resize_bbox(self.img_size, mask=[mask_np])
        image_crop = crop_resize_fn(image_np)
        tgt_image_crop = crop_resize_fn(tgt_image_np)
        mask_crop = crop_resize_fn(mask_np)
        partial_mask_crop = crop_resize_fn(partial_mask_np)
        mask_diff_crop = crop_resize_fn(mask_diff_np)

        image_rgb = self.transform_rgba(image_crop)
        mask_rgb = self.transform_rgba_mask(partial_mask_crop)
        src_image = image_rgb * mask_rgb
        #print('src_image:',src_image.shape)

        rgb_clip = self.transform_clip(image_crop)
        mask_clip = self.transform_clip_mask(partial_mask_crop)
        src_clip_image = rgb_clip * mask_clip
        #print('src_clip_image:', src_clip_image.shape)

        image_rgb_tgt = self.transform_rgba(tgt_image_crop)
        mask_rgb_tgt = self.transform_rgba_mask(mask_crop)
        target = image_rgb_tgt * mask_rgb_tgt
        #print('target:', target.shape)

        mask_rgb_diff = self.transform_rgba_mask(mask_diff_crop)
        #print('mask_rgb_diff:', mask_rgb_diff.shape)

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
                'filename': self.subject_list[index].split('/')[-3]+'_' +self.subject_list[index].split('/')[-2]+'_'+self.subject_list[index].split('_')[-1]
        }

    def __len__(self):
        return self.num_subjects