# import numpy as np
# from PIL import Image

# remaining_image = np.load('/home/hanshengliang/mae_deploy/input/combined_image.npy', encoding="latin1")  # 112, 112, 3
# mask = np.load('/home/hanshengliang/mae_deploy/input/mask_positions.npy', encoding="latin1")  # 14, 14

# mask_shape = mask.shape  # (14, 14)
# remaining_patches = remaining_image.shape[0] // 16  # 7
# patch_size = 224 // mask_shape[0]  # 16
# mask_ratio = 1 - (remaining_patches * remaining_patches) / (mask_shape[0] * mask_shape[0])

# restored_image = np.zeros((224, 224, 3), dtype=np.uint8)

# mask = mask.reshape(mask_shape)

# idx = 0
# for i in range(mask_shape[0]):
#     for j in range(mask_shape[1]):
#         if mask[i, j] == 1 and idx < remaining_patches**2:
#             start_row, start_col = i * patch_size, j * patch_size
#             patch_row, patch_col = divmod(idx, remaining_patches)
#             patch = remaining_image[patch_row * patch_size:(patch_row + 1) * patch_size,
#                                      patch_col * patch_size:(patch_col + 1) * patch_size, :]
#             restored_image[start_row:start_row + patch_size, start_col:start_col + patch_size, :] = patch
#             idx += 1

# restored_image = restored_image[:, :, ::-1]  # 通过切片倒转最后一个维度

# Image.fromarray(restored_image).save("restored_image.png")
# print("还原图像已保存为 'restored_image.png'")



import numpy as np
from PIL import Image
import os

# 读取图片并保存为 NumPy
def save_image_as_numpy(image_path, save_path):
    """
    读取一张图片并保存为 NumPy 数组文件。
    
    :param image_path: 图片文件路径
    :param save_path: NumPy 文件保存路径
    """
    try:
        # 打开图片
        image = Image.open(image_path).convert('RGB')
        
        # 转换为 NumPy 数组
        image_array = np.array(image)
        
        # 保存为 .npy 文件
        np.save(save_path, image_array)
        print(f"图片已保存为 NumPy 数组文件: {save_path}")
        
    except Exception as e:
        print(f"处理图片时出错: {e}")

# 示例用法
image_file = "/home/hanshengliang/WechatIMG12549.jpg"  # 替换为你的图片路径
numpy_file = "example.npy"  # 替换为你希望保存的 NumPy 文件名
save_image_as_numpy(image_file, numpy_file)

# 可选：加载并检查保存的 NumPy 文件
loaded_array = np.load(numpy_file)
print("图片转换后的 NumPy 数组形状:", loaded_array.shape)