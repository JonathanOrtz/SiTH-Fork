"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""

import argparse
import os
import numpy as np
import torch
import torch.utils.checkpoint
from packaging import version
from PIL import Image
from tqdm.auto import tqdm
from accelerate.utils import  set_seed
from transformers import CLIPVisionModelWithProjection
import torchvision.transforms.functional as TF
from kornia.geometry.transform import get_affine_matrix2d, warp_affine
import shutil

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from diffusion.lib.test_diffusion_dataset_chore import TestDiffDataset
from diffusion.lib.pipeline import BackHallucinationPipeline
from diffusion.lib.ccprojection import CCProjection
from diffusion.lib.utils import tensor_to_np, image_grid

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0")
UV_TEMPLATE = 'data/smplx_uv.obj'
def unwrap(image, matrix):   
    print(image.shape, matrix.shape) 
    if image.shape[0]==1:
        if image.shape[1]!=3:
            image=image.permute(3,1,2)
            device = image.device    
            img_square = warp_affine(image,torch.inverse(matrix)[0,:,:2].to(device),(1536, 2048),mode='bilinear', padding_mode='zeros',align_corners=True)    
    
        else:
            device = image.device    
            img_square = warp_affine(image,torch.inverse(matrix)[0,:,:2].to(device),(1536, 2048),mode='bilinear', padding_mode='zeros',align_corners=True)    
    

    else:
        if image.shape[0]!=3:
            image=image.permute(2,0,1)
            image=image.unsqueeze(0)
            device = image.device    
            img_square = warp_affine(image,torch.inverse(matrix)[0,:,:2].to(device),(1536, 2048),mode='bilinear', padding_mode='zeros',align_corners=True)    
        else:
            image=image.unsqueeze(0)
            device = image.device    
            img_square = warp_affine(image,torch.inverse(matrix)[0,:,:2].to(device),(1536, 2048),mode='bilinear', padding_mode='zeros',align_corners=True)    
    # img_ori = warp_affine(img_square,torch.inverse(uncrop_param["M_square"])[:, :2].to(device), uncrop_param["ori_shape"], mode='bilinear',padding_mode='zeros', align_corners=True)    
    return img_square

def unwrap_v2(matrix,cropped_img):
    inv_matrix = torch.inverse(matrix[:, :2, :2])  # Invert the linear part
    translation = matrix[:, :2, 2].unsqueeze(-1)   # Get the translation vector
    inv_translation = -torch.matmul(inv_matrix, translation).squeeze(-1)  # Invert translation

    # Construct the full inverted affine matrix
    inverted_matrix = torch.eye(3).repeat(matrix.shape[0], 1, 1)
    inverted_matrix[:, :2, :2] = inv_matrix
    inverted_matrix[:, :2, 2] = inv_translation
    angle = 0
    trans_inv = inverted_matrix[:, :2, 2]
    scale = [1.0, 1.0]  # The scaling factor to go back to original size (no scaling for inverse)
    shear = [0.0, 0.0]  # Assuming no shear

    # Apply the transformation
    original_img_approx = TF.affine(cropped_img, angle=angle, translate=(trans_inv[0], trans_inv[1]), scale=scale, shear=shear)

    return original_img_approx
def main(args):
    
    os.makedirs(args.output_path, exist_ok=True)
    logging_dir = os.path.join(args.output_path, 'all_images')
    os.makedirs(logging_dir, exist_ok=True)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    weight_dtype = torch.float32

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        generator = torch.Generator(device=device).manual_seed(args.seed)
    else:
        generator = None

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path_controlnet, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.pretrained_model_name_or_path_controlnet, subfolder="controlnet")
    refer_clip_proj = CCProjection.from_pretrained(args.pretrained_model_name_or_path_controlnet, subfolder="projection", clip_image_encoder=clip_image_encoder)
    
    # Freeze the model
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    clip_image_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)
    refer_clip_proj.requires_grad_(False)

    # Load the dataset
    val_dataset = TestDiffDataset(args, args.input_path, device, size=args.resolution)

    val_dataloader =  torch.utils.data.DataLoader(dataset=val_dataset, 
                        batch_size=1, 
                        shuffle=False, 
                        num_workers=0,
                        pin_memory=True)

    pipeline = BackHallucinationPipeline(
        vae=vae,
        clip_image_encoder=clip_image_encoder,
        unet=unet,
        controlnet=controlnet,
        scheduler=noise_scheduler,
        refer_clip_proj=refer_clip_proj,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            pipeline.enable_xformers_memory_efficient_attention()

        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
    
    for i, data in enumerate(tqdm(val_dataloader)):
        if i  in range(args.start_idx, args.end_idx):
            fname = data['filename'][0]
            if data['yes_diff']:
                images = []
                images.append(tensor_to_np(data['src_ori_image']))
                # images.append(tensor_to_np(data['tgt_uv']))
                

                with torch.autocast("cuda"):
                        im = pipeline.forward(args,  data,  generator, num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale, controlnet_conditioning_scale=args.conditioning_scale,
                        num_images_per_prompt = args.num_validation_images
                        )
                for j in range(args.num_validation_images):
                    rewarped_img=unwrap(torch.tensor(im[j]), data['M_crop'])
                    image=rewarped_img.cpu().permute(0, 2, 3, 1).float().numpy()
                    pil_img = Image.fromarray((image[0]*255).astype(np.uint8))
                    # pil_img.save(os.path.join(logging_dir,  f"%s_%03d.png" % (fname, j)))
                    os.makedirs(os.path.join(args.output_path, os.path.split(fname)[0]),exist_ok=True)
                    # if j == 0:
                    pil_img.save(os.path.join(args.output_path, os.path.split(fname)[0],f"%03d_%s" % (j,os.path.split(fname)[-1])))

                    # images.append(im[j:j+1])

                # grid = image_grid(images, 1/, args.num_validation_images +2 )
                # grid.save(os.path.join(logging_dir, f"{fname}_all.png"))
            else:
                rewarped_img=unwrap(data['src_ori_image'], data['M_crop'])
                image=rewarped_img.cpu().permute(0, 2, 3, 1).float().numpy()
                pil_img = Image.fromarray((image[0] * 255).astype(np.uint8))
                os.makedirs(os.path.join(args.output_path, os.path.split(fname)[0]),exist_ok=True)
                
                pil_img.save(os.path.join(args.output_path, os.path.split(fname)[0],f"%03d_%s" % (0,os.path.split(fname)[-1])))
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        default="./data/examples",
        help=(
            "The path to the dataset. The directory should contain a images folder and a smplx meshes folder."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="./data/examples/back_images",
        help=(
            "The output path for the generated images. The generated images will be saved in this path."
        ),
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='hohs/SiTH_diffusion',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path_controlnet",
        type=str,
        default='hohs/SiTH_diffusion',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--conditioning_channels",
        type=int,
        default=1,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="classifier free guidence scale"
    )
    parser.add_argument(
        "--conditioning_scale",
        type=float,
        default=1.0,
        help="Controlnet conditioning scale"
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to be generated",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Number of images to be generated",
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=1,
        help="Number of images to be generated",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=150,
        help="Number of inference steps",
    )
    parser.add_argument(
        "--both_cond",
        action="store_true",
        help="Number of images to be generated",
    )
    parser.add_argument(
        "--only_mask",
        action="store_true",
        help="Number of images to be generated",
    )
    parser.add_argument(
        "--only_normal",
        action="store_true",
        help="Number of images to be generated",
    )
    parser.add_argument(
        "--create_normals",
        action="store_true",
        help="Number of images to be generated",
    )

    args = parser.parse_args()


    main(args)
