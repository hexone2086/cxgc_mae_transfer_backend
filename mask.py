import cv2
import numpy as np
import random
from typing import List, Tuple, Union
import pickle

DynamicMask = False

class patch_generator:
    def __init__(self):
        self.mask_positions_default = None
        self.patch_size = 16
        pass

    def forward(self, x):
        # Convert cv2.Mat to numpy array
        x = np.array(x)

        h, w, c = x.shape
        # assert h == self.img_size and w == self.img_size, \
        #     f"Input size ({h}x{w}) does not match expected size ({self.img_size}x{self.img_size})."

        # Divide the image into patches
        ph, pw = self.patch_size, self.patch_size
        patches = x.reshape(h // ph, ph, w // pw, pw, c)
        patches = patches.transpose(0, 2, 1, 3, 4).reshape(-1, ph * pw * c)

        patches_reshaped = patches.reshape(1, -1, ph*pw, c)
        # 归一化 0~1
        patches_reshaped = patches_reshaped / 255
        patch_means = np.mean(patches_reshaped, axis=-2, keepdims=True)
        patch_stds = np.sqrt(np.var(patches_reshaped, axis=-2, ddof=1, keepdims=True)) + 1e-6
        # print(patch_means, patch_stds)
        return patches, patch_means.astype(np.float32), patch_stds.astype(np.float32)

    def load_and_resize_image(self, image: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """加载并调整图像大小"""
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a NumPy array")
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        return image

    def split_into_patches(self, image: np.ndarray, patch_size: Tuple[int, int] = (16, 16)) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """将图像分割成小块，并保留每个小块的坐标信息"""
        height, width, _ = image.shape
        patch_height, patch_width = patch_size
        patches = []
        for y in range(0, height, patch_height):
            for x in range(0, width, patch_width):
                patch = image[y:y + patch_height, x:x + patch_width]
                patches.append((patch, (x // patch_width, y // patch_height)))
        return patches

    def generate_mask(self, grid_size: Tuple[int, int] = (14, 14), num_patches: int = 49) -> np.ndarray:
        """生成随机掩码的位置"""
        mask = np.zeros(grid_size, dtype=bool)
        positions = random.sample(range(grid_size[0] * grid_size[1]), num_patches)
        for pos in positions:
            x = pos % grid_size[0]
            y = pos // grid_size[0]
            mask[y, x] = True
        return mask

    def select_random_patches(self, patches: List[Tuple[np.ndarray, Tuple[int, int]]], mask: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """通过掩码的位置选取 patch"""
        selected_patches = []
        for patch, (x, y) in patches:
            if mask[y, x]:
                selected_patches.append((patch, (x, y)))
        return selected_patches

    def reconstruct_image(self, selected_patches: List[Tuple[np.ndarray, Tuple[int, int]]], patch_size: Tuple[int, int] = (16, 16), grid_size: Tuple[int, int] = (7, 7)) -> np.ndarray:
        """将小块重新拼合成新图像，依据坐标信息进行拼接"""
        patch_height, patch_width = patch_size
        grid_height, grid_width = grid_size
        new_image = np.zeros((grid_height * patch_height, grid_width * patch_width, 3), dtype=np.uint8)
        
        # 按照从上至下、从左至右的顺序排序
        selected_patches.sort(key=lambda x: (x[1][1], x[1][0]))
        
        for i, (patch, (x, y)) in enumerate(selected_patches):
            new_x = i % grid_width
            new_y = i // grid_width
            new_image[new_y * patch_height:(new_y + 1) * patch_height, new_x * patch_width:(new_x + 1) * patch_width] = patch
        return new_image

    def apply_mask_to_image(self, image: np.ndarray, mask: np.ndarray, patch_size: Tuple[int, int] = (16, 16)) -> np.ndarray:
        """将mask应用到原图上，未被选中的patch设置为黑色"""
        height, width, _ = image.shape
        patch_height, patch_width = patch_size
        masked_image = image.copy()
        for y in range(0, height, patch_height):
            for x in range(0, width, patch_width):
                if not mask[y // patch_height, x // patch_width]:
                    masked_image[y:y + patch_height, x:x + patch_width] = 0
        return masked_image

    def initialize_default_mask_positions(self, grid_size: Tuple[int, int] = (14, 14), num_patches: int = 49):
        """初始化默认的随机掩码位置"""
        self.mask_positions_default = self.generate_mask(grid_size, num_patches)

    def save_mask_positions(self, filename: str):
        """保存掩码位置到文件"""
        with open(filename, 'wb') as f:
            pickle.dump(self.mask_positions_default, f)

    def load_mask_positions(self, filename: str):
        """从文件加载掩码位置"""
        with open(filename, 'rb') as f:
            self.mask_positions_default = pickle.load(f)

    def process_image(self, image: np.ndarray, num_patches: int = 49) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """处理图像，返回掩码后拼合的图像和布尔数组"""
        if not isinstance(image, np.ndarray):
            raise TypeError("Input image must be a NumPy array")
        
        # 加载并调整图像大小
        image = self.load_and_resize_image(image)
        
        # 计算均值和标准差
        _, patch_means, patch_stds = self.forward(image)

        # 将图像分割成小块，并保留每个小块的坐标信息
        patches = self.split_into_patches(image)
        
        # 生成或加载掩码
        if DynamicMask:
            mask = self.generate_mask(grid_size=(14, 14), num_patches=num_patches)
        else:
            if self.mask_positions_default is None:
                self.initialize_default_mask_positions(grid_size=(14, 14), num_patches=num_patches)
            mask = self.mask_positions_default
        
        # 通过掩码的位置选取 patch
        selected_patches = self.select_random_patches(patches, mask)
        
        # 将小块重新拼合成新图像，依据坐标信息进行拼接
        combined_image = self.reconstruct_image(selected_patches)
        
        # 将mask应用到原图上
        masked_image = self.apply_mask_to_image(image, mask)

        return masked_image, combined_image, patch_means, patch_stds, mask

if __name__ == "__main__":
    image_path = "ILSVRC2012_test_00078306.JPEG"  # 替换为你的输入图像路径
    output_path = "output_image.jpg"  # 替换为你的输出图像路径
    mask_positions_file = "mask_positions.pkl"  # 掩码位置保存文件

    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image")
    
    # 创建 patch_generator 实例
    pg = patch_generator()
    
    # 加载或初始化掩码位置
    try:
        pg.load_mask_positions(mask_positions_file)
    except FileNotFoundError:
        pg.initialize_default_mask_positions()
        pg.save_mask_positions(mask_positions_file)
    
    # 处理图像
    masked_image, combined_image, mask = pg.process_image(image)
    
    # 保存结果图像
    cv2.imwrite(output_path, masked_image)
    
    # 显示结果图像
    cv2.imshow("Masked Image", masked_image)
    cv2.imshow("Combined Image", combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 打印布尔数组
    print("Mask (14x14 bool array):")
    print(mask)