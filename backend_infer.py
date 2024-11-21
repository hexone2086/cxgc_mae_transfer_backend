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

    def infer(self, data_dict):
        remaining_image = data_dict['remaining_image']
        mask = data_dict['mask']
        img_mean = data_dict['img_mean']
        img_std = data_dict['img_std']

        restored_image = self.restore_image(remaining_image, mask)
        img = Image.fromarray(restored_image)
        img, _ = self.transforms(img)
        bool_masked_pos = torch.from_numpy(1 - mask).view(1, -1).to(self.device, non_blocking=True).to(torch.bool)

        with torch.no_grad():
            img = img[None, :]
            img = img.to(self.device, non_blocking=True)
            img_ = rearrange(img, 'b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=16, pw=16)
            outputs = self.model(img_, bool_masked_pos)

            # Save original image
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(self.device)[None, :, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(self.device)[None, :, None, None]
            ori_img = img * std + mean  # in [0, 1]

            # Save reconstruction image
            img_mean = np.copy(img_mean)  # 复制 NumPy 数组以使其可写
            img_std = np.copy(img_std)  # 复制 NumPy 数组以使其可写
            img_mean = torch.as_tensor(img_mean, dtype=torch.float32).to(self.device)
            img_std = torch.as_tensor(img_std, dtype=torch.float32).to(self.device)
            img_squeeze = rearrange(ori_img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=self.patch_size[0], p2=self.patch_size[0])
            img_norm = (img_squeeze - img_mean) / img_std
            img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
            img_patch[bool_masked_pos] = outputs

            mask_ = torch.ones_like(img_patch)
            mask_[bool_masked_pos] = 0
            mask_ = rearrange(mask_, 'b n (p c) -> b n p c', c=3)
            mask_ = rearrange(mask_, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=self.patch_size[0], p2=self.patch_size[1], h=14, w=14)

            rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
            rec_img = rec_img * img_std + img_mean
            rec_img = rearrange(rec_img, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=self.patch_size[0], p2=self.patch_size[1], h=14, w=14)

            return rec_img.cpu().numpy()

if __name__ == '__main__':
    mae_vis = MAEVisualization(
        model_path='path/to/model'
    )

    data_dict = {
        'remaining_image': np.load('path/to/remaining_image.npy'),
        'mask': np.load('path/to/mask.npy'),
        'img_mean': np.load('path/to/img_mean.npy'),
        'img_std': np.load('path/to/img_std.npy')
    }

    output_image = mae_vis.infer(data_dict)
    logger.info(f"Inference completed. Output image shape: {output_image.shape}")

    # 将输出图像转换为 OpenCV 的 cv2.Mat 格式并显示
    output_image_cv = (output_image * 255).astype(np.uint8)
    output_image_cv = cv2.cvtColor(output_image_cv, cv2.COLOR_RGB2BGR)

    cv2.imshow('Reconstructed Image', output_image_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()