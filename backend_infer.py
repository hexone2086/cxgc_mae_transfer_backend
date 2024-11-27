import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from pathlib import Path
from timm.models import create_model
import utils
import modeling_pretrain
from datasets import DataAugmentationForMAE
from torchvision.transforms import ToPILImage
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from loguru import logger

class MAEVisualization:
    def __init__(self, model_path, input_size=224, device='cuda:0', imagenet_default_mean_and_std=True, mask_ratio=0.9,
                 model_name='pretrain_mae_base_patch16_224', drop_path=0.0):
        self.input_size = input_size
        self.device = device
        self.imagenet_default_mean_and_std = imagenet_default_mean_and_std
        self.mask_ratio = mask_ratio
        self.model_name = model_name
        self.drop_path = drop_path

        self.device = torch.device(self.device)
        cudnn.benchmark = True

        self.model = self.get_model()
        self.patch_size = self.model.encoder.patch_embed.patch_size
        logger.info("Patch size = %s" % str(self.patch_size))
        self.window_size = (self.input_size // self.patch_size[0], self.input_size // self.patch_size[1])

        self.model.to(self.device)
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

        logger.info(f"load model from {model_path}")

        self.transforms = DataAugmentationForMAE(self)

    def restore_image(self, remaining_image, mask):
        """
        根据剩余图像和掩码还原完整图像。

        参数：
        - remaining_image: ndarray，剩余拼接图像 (1 * patch_size, n * patch_size, 3)，BGR 格式。
        - mask: ndarray，掩码矩阵 (m, n)。

        返回：
        - restored_image: ndarray，还原的图像 (224, 224, 3)，RGB 格式。
        """
        mask_shape = mask.shape
        patch_size = 224 // mask_shape[0]

        restored_image = np.zeros((224, 224, 3), dtype=np.uint8)
        mask = mask.reshape(mask_shape)

        idx = 0
        for i in range(mask_shape[0]):
            for j in range(mask_shape[1]):
                if mask[i, j] == 1 and idx < remaining_image.shape[1] // patch_size:
                    start_row, start_col = i * patch_size, j * patch_size
                    patch = remaining_image[:, idx * patch_size:(idx + 1) * patch_size, :]
                    restored_image[start_row:start_row + patch_size, start_col:start_col + patch_size, :] = patch
                    idx += 1
        
        return restored_image  # BGR 转为 RGB

    def get_model(self):
        logger.info(f"Creating model: {self.model_name}")
        model = create_model(
            self.model_name,
            pretrained=False,
            drop_path_rate=self.drop_path,
            drop_block_rate=None,
        )
        return model

    def save_images(self, outputs, img, bool_masked_pos, img_std, img_mean, img_list, idx):
        """
        保存原始图像、重建图像和随机掩码图像。
        """
        # 保存原始图像
        mean = torch.as_tensor([0.485, 0.456, 0.406]).to(self.device)[None, :, None, None]
        std = torch.as_tensor([0.229, 0.224, 0.225]).to(self.device)[None, :, None, None]
        ori_img = img * std + mean  # 将归一化的图像转回 [0, 1]

        # 处理重建图像
        img_squeeze = rearrange(ori_img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', 
                                p1=self.patch_size[0], p2=self.patch_size[1])
        img_norm = (img_squeeze - img_mean) / img_std
        img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
        # logger.debug(f"img_patch shape: {img_patch.shape}")
        img_patch[bool_masked_pos] = outputs

        # 构造掩码
        mask_ = torch.ones_like(img_patch)
        mask_[bool_masked_pos] = 0
        mask_ = rearrange(mask_, 'b n (p c) -> b n p c', c=3)
        mask_ = rearrange(mask_, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', 
                        p1=self.patch_size[0], p2=self.patch_size[1], h=14, w=14)

        
        rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
        rec_img = rec_img * img_std + img_mean  
        rec_img = rearrange(rec_img, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', 
                            p1=self.patch_size[0], p2=self.patch_size[1], h=14, w=14)
        # logger.debug(f"rec_img shape: {rec_img.shape}")

        # 保存重建图像
        rec_img_pil = ToPILImage()(rec_img[0, :].clip(0, 0.996))
        rec_img_pil.save(f"./output/rec_img_{idx}.jpg")    
        # 保存掩码图像
        img_mask = rec_img * mask_
        img_mask_pil = ToPILImage()(img_mask[0, :])
        img_mask_pil.save(f"./output/mask_img_{idx}.jpg")



        img_list.append(rec_img)


    def infer(self, data_dict_list):

        img_list = []
        mask_list = []
        std_list = []
        mean_list = []

        # FIXME 方差均值可能有误
        for i in range(len(data_dict_list)):
            data_dict = data_dict_list[i]
            remaining_image = data_dict['remaining_image']
            mask = data_dict['mask']
            img_mean = data_dict['img_mean']
            img_std = data_dict['img_std']

            # 压入数据到列表中
            restored_image = self.restore_image(remaining_image, mask)
            img_t = Image.fromarray(restored_image)
            img_t, _ = self.transforms(img_t)
            # logger.debug(img_t.shape)
            img_t = img_t[None, :]
            img_list.append(img_t)

            mask_t = torch.from_numpy(1 - mask).view(1, -1).to(self.device, non_blocking=True).to(torch.bool)
            mask_list.append(mask_t)

            std_list.append(img_std)
            mean_list.append(img_mean)

        img = torch.cat(img_list, dim=0)
        bool_masked_pos = torch.cat(mask_list, dim=0)
        img_std = np.concatenate(std_list, axis=0)
        img_mean = np.concatenate(mean_list, axis=0)

        logger.debug(f"img shape: {img.shape}")
        logger.debug(f"bool_masked_pos shape: {bool_masked_pos.shape}")
        logger.debug(f"img_mean shape: {img_mean.shape}")
        logger.debug(f"img_std shape: {img_std.shape}")

        # remaining_image = data_dict['remaining_image']
        # mask = data_dict['mask']
        # img_mean = data_dict['img_mean']
        # img_std = data_dict['img_std']
        
        # # 复制四份
        # img_list = []
        # for _ in range(4):
        #     restored_image = self.restore_image(remaining_image, mask)
        #     img_t = Image.fromarray(restored_image)
        #     img_t, _ = self.transforms(img_t)
        #     # logger.debug(img_t.shape)
        #     img_t = img_t[None, :]
        #     img_list.append(img_t)
        # img = torch.cat(img_list, dim=0)

        # # 复制四份
        # mask_list = []
        # for _ in range(4):
        #     mask_t = torch.from_numpy(1 - mask).view(1, -1).to(self.device, non_blocking=True).to(torch.bool)
        #     mask_list.append(mask_t)
        # bool_masked_pos = torch.cat(mask_list, dim=0)


        # std_t = []
        # mean_t = []
        # for _ in range(4):
        #     std_t.append(img_std)
        #     mean_t.append(img_mean)
        # img_std = np.concatenate(std_t, axis=0)
        # img_mean = np.concatenate(mean_t, axis=0)


        with torch.no_grad():
            # img shape: (n, 3, 224, 224)
            # img_ shape: (n, 196, 768)         
            # bool_masked_pos shape: (n, 196)
            # outputs shape: (n, 47, 768)
            # img_mean shape: (n, 196, 1, 3)
            # img_std shape: (n, 196, 1, 3)

            img = img.to(self.device, non_blocking=True)
            img_ = rearrange(img, 'b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=16, pw=16)

            # logger.debug(f"img shape: {img.shape}")
            # logger.debug(f"img_ shape: {img_.shape}")
            # logger.debug(f"bool_masked_pos shape: {bool_masked_pos.shape}")

            outputs = self.model(img_, bool_masked_pos)
            # logger.debug(f"outputs shape: {outputs.shape}")

            # Save original image
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(self.device)[None, :, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(self.device)[None, :, None, None]
            ori_img = img * std + mean  # in [0, 1]

            logger.debug(ori_img.shape)

            # Save reconstruction image
            img_mean = np.copy(img_mean)
            img_std = np.copy(img_std)
            img_mean = torch.as_tensor(img_mean, dtype=torch.float32).to(self.device)
            img_std = torch.as_tensor(img_std, dtype=torch.float32).to(self.device)
            logger.debug(f"img_mean shape: {img_mean.shape}")
            logger.debug(f"img_std shape: {img_std.shape}")

            img_squeeze = rearrange(ori_img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=self.patch_size[0], p2=self.patch_size[0])
            img_norm = (img_squeeze - img_mean) / img_std
            img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
            logger.debug(f"img_patch shape: {img_patch.shape}")

            rec_img_list = []

            logger.debug(f"idx of image: {len(img)}")
            for i in range(len(img)):
                self.save_images(outputs[i:i+1], img[i:i+1], bool_masked_pos[i:i+1], img_std[i:i+1], img_mean[i:i+1], rec_img_list,idx=i)

            return rec_img_list

if __name__ == '__main__':
    mae_vis = MAEVisualization(
        model_path='./model/checkpoint-1599.pth',
        mask_ratio=0.75
    )

    data_dict = {
        'remaining_image': np.load('./input/combined_image_w.npy'),
        'mask': np.load('./input/mask_positions.npy'),
        'img_mean': np.load('./input/patch_means.npy'),
        'img_std': np.load('./input/patch_stds.npy')
    }

    output = mae_vis.infer(data_dict)
