# -*- coding: utf-8 -*-
# @Time    : 2021/11/18 22:40
# @Author  : zhao pengfei
# @Email   : zsonghuan@gmail.com
# @File    : run_mae_vis.py
# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
from loguru import logger
import argparse
import datetime
from scipy.ndimage import gaussian_filter
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import cv2
from PIL import Image, ImageFilter

from pathlib import Path

from timm.models import create_model

import utils
import modeling_pretrain
from datasets import DataAugmentationForMAE

from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
def restore_image(remaining_image, mask):
    """
    根据剩余图像和掩码还原完整图像。

    参数：
    - remaining_image: ndarray，剩余拼接图像 (h, w, 3)，BGR 格式。
    - mask: ndarray，掩码矩阵 (m, n)。

    返回：
    - restored_image: ndarray，还原的图像 (224, 224, 3)，RGB 格式。
    """
    mask_shape = mask.shape
    remaining_patches = remaining_image.shape[0] // 16
    patch_size = 224 // mask_shape[0]

    restored_image = np.zeros((224, 224, 3), dtype=np.uint8)
    mask = mask.reshape(mask_shape)

    idx = 0
    for i in range(mask_shape[0]):
        for j in range(mask_shape[1]):
            if mask[i, j] == 1 and idx < remaining_patches**2:
                start_row, start_col = i * patch_size, j * patch_size
                patch_row, patch_col = divmod(idx, remaining_patches)
                patch = remaining_image[patch_row * patch_size:(patch_row + 1) * patch_size,
                                         patch_col * patch_size:(patch_col + 1) * patch_size, :]
                restored_image[start_row:start_row + patch_size, start_col:start_col + patch_size, :] = patch
                idx += 1
    
    return restored_image  # BGR 转为 RGB
def get_args():
    parser = argparse.ArgumentParser('MAE visualization reconstruction script', add_help=False)
    parser.add_argument('img_path', type=str, help='input image path')
    parser.add_argument('save_path', type=str, help='save image path')
    parser.add_argument('model_path', type=str, help='checkpoint path of model')
    parser.add_argument('remaining_image_path', type=str, help='path to remaining image (npy)')
    parser.add_argument('mask_path', type=str, help='path to mask (npy)')
    parser.add_argument('img_mean_path', type=str, help='path to img mean (npy)')
    parser.add_argument('img_std_path', type=str, help='path to img std (npy)')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.25, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    # Model parameters
    parser.add_argument('--model', default='pretrain_mae_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    return model

def save_images(outputs, img, bool_masked_pos, img_std, img_mean, patch_size, save_path, idx, device):
    """
    保存原始图像、重建图像和随机掩码图像。
    """
    # 保存原始图像
    mean = torch.as_tensor([0.485, 0.456, 0.406]).to(device)[None, :, None, None]
    std = torch.as_tensor([0.229, 0.224, 0.225]).to(device)[None, :, None, None]
    ori_img = img * std + mean  # 将归一化的图像转回 [0, 1]
    ori_img_pil = ToPILImage()(ori_img[0, :])
    ori_img_pil.save(f"{save_path}/ori_img_{idx}.jpg")
    #import pdb;pdb.set_trace()
    # 处理重建图像
    img_squeeze = rearrange(ori_img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', 
                            p1=patch_size[0], p2=patch_size[1])
    img_norm = (img_squeeze - img_mean) / img_std
    img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
    logger.info(f"img_patch shape: {img_patch.shape}")
    img_patch[bool_masked_pos] = outputs

    # 构造掩码
    mask_ = torch.ones_like(img_patch)
    mask_[bool_masked_pos] = 0
    mask_ = rearrange(mask_, 'b n (p c) -> b n p c', c=3)
    mask_ = rearrange(mask_, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', 
                      p1=patch_size[0], p2=patch_size[1], h=14, w=14)

    # 保存重建图像
    rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
    rec_img = rec_img * img_std + img_mean  
    rec_img = rearrange(rec_img, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', 
                        p1=patch_size[0], p2=patch_size[1], h=14, w=14)
############
    # 转换为 NumPy 格式
    rec_img_np = rec_img.cpu().numpy()  # 假设 rec_img 是一个 torch.Tensor

    # 对每个通道应用高斯滤波
    rec_img_smoothed = np.zeros_like(rec_img_np)
    for b in range(rec_img_np.shape[0]):  # 遍历 batch
        for c in range(rec_img_np.shape[1]):  # 遍历通道
            rec_img_smoothed[b, c] = gaussian_filter(rec_img_np[b, c], sigma=2.0)  # 调整 sigma 减弱块状痕迹

    # 转换回 PyTorch 格式
    rec_img_smoothed = torch.from_numpy(rec_img_smoothed).to(device)
###########
    # rec_img_pil = ToPILImage()(rec_img_smoothed[0, :].clip(0, 0.996))
    # rec_img_pil.save(f"{save_path}/rec_img_{idx}.jpg")

    # # 保存掩码图像
    # img_mask = rec_img * mask_
    # img_mask_pil = ToPILImage()(img_mask[0, :])
    # img_mask_pil.save(f"{save_path}/mask_img_{idx}.jpg")

    print(f"Images saved to {save_path}")


def main(args):
    print(args)
    #logger.info(args)

    device = torch.device(args.device)
    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    #logger.info(f"load model from {args.model_path}")

    transforms = DataAugmentationForMAE(args)


    remaining_image = np.load(args.remaining_image_path)
    mask = np.load(args.mask_path)
    restored_image = restore_image(remaining_image, mask)
    #import pdb;pdb.set_trace()
    img = Image.fromarray(restored_image)
    img, _ = transforms(img)
    bool_masked_pos = torch.from_numpy(1 - mask).view(1, -1).to(device, non_blocking=True).to(torch.bool)
    #bool_masked_pos = 1 - bool_masked_pos
    #import pdb;pdb.set_trace()
        # 统计 bool_masked_pos 第二个维度中有多少个 true
    num_true_per_batch = torch.sum(bool_masked_pos, dim=1)

    for i, num_true in enumerate(num_true_per_batch):
        logger.info(f"Batch {i}: {num_true} true values")


    with torch.no_grad():
        img = img[None, :]
        # bool_masked_pos = bool_masked_pos[None, :]
        img = img.to(device, non_blocking=True)
        img_ = rearrange(img, 'b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=16, pw=16)
        # bool_masked_pos = bool_masked_pos.to(device, non_blocking=True).flatten(1).to(torch.bool)
        img4 = img.repeat(4, 1, 1, 1) #复制四份
        logger.debug(f" final img shape: {img4.shape}")
        img_4=img_.repeat(4, 1, 1) #复制四份
        logger.debug(f" final img_ shape: {img_4.shape}")
        bool_masked_pos4 = bool_masked_pos.repeat(4,1)#复制四份
        logger.debug(f" final bool_masked_pos shape: {bool_masked_pos4.shape}")

        outputs = model(img_4, bool_masked_pos4)
        # import pdb;pdb.set_trace()
        img_mean = np.load(args.img_mean_path)
        img_mean = torch.as_tensor(img_mean, dtype=torch.float32).repeat(4, 1, 1,1).to(device)
        img_std = np.load(args.img_std_path)
        img_std = torch.as_tensor(img_std, dtype=torch.float32).repeat(4, 1, 1,1).to(device)

        logger.debug(f" final outputs shape: {outputs.shape}")
        logger.debug(f" final img_mean shape: {img_mean.shape}")
        logger.debug(f" final img_std shape: {img_std.shape}")

        for i in range(len(img4)):
            save_images(outputs[i:i+1], img4[i:i+1], bool_masked_pos4[i:i+1], img_std[i:i+1], img_mean[i:i+1], patch_size=(16, 16), 
                save_path=args.save_path, idx=i, device=device)

if __name__ == '__main__':
    opts = get_args()
    main(opts)
