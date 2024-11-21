import subprocess
import threading
import cv2
import numpy as np
import zlib
import time
from loguru import logger

from backend_infer import MAEVisualization
from backend_require import require


received_data = None

def main():

    req = require(5555)
    req_thread = threading.Thread(target = req.worker)
    logger.info("start req thread")
    req_thread.start()

    mae_infer = MAEVisualization(model_path='model/checkpoint-1599.pth',)

    while True:
        if len(req.buffer) > 0:
            received_data = req.buffer.pop()

            # 创建字典，用于送入推理模块
            data_dict = {
                'remaining_image': received_data['mat2'],
                'mask': received_data['bool_array'],
                'img_mean': received_data['patch_means'],
                'img_std': received_data['patch_stds']
            }
            
            logger.info("start infer")
            output = mae_infer.infer(data_dict)
            logger.info("infer completed")

            output = output.transpose(0, 2, 3, 1) # 推理原始输出为 (bcz, 3, 224, 224)
            n = output.shape[0]
            for i in range(n):
                # 限制 output 数值 0-1 防止溢出
                output[i] = np.clip(output[i], 0, 1)
                output[i] = (output[i] * 255).astype(np.uint8)
                cv2.imwrite(f'./output/output_{i}.png', cv2.cvtColor(output[i], cv2.COLOR_RGB2BGR) )
        else:
            pass

    req_thread.join()
    pass

if __name__ == '__main__':
    main()