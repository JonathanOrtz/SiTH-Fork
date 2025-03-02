#!/bin/bash
wandb disabled;
exec /mnt/fast/nobackup/users/ad02037/anaconda3/envs/sith/bin/accelerate launch /mnt/fast/nobackup/users/ad02037/SiTH_fork/diffusion/train_obj_diffusion.py  --output_dir /mnt/fast/nobackup/users/ad02037/out_inpainting --conditioning_channels 1 --white_background --validation --config /mnt/fast/nobackup/users/ad02037/SiTH_fork/diffusion/config.yaml --input_path /mnt/fast/datasets/ucdatasets/synHOR/sequences --input_path_test /mnt/fast/datasets/ucdatasets/synHOR/sequences --batch_size 6 --train_batch_size 6 --exp_name diff_mask --num_validation_images 5
