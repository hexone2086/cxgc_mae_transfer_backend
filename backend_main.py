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
        data_dict_list = []

        if len(req.buffer) > 3:
            for i in range(4):
                received_data = req.buffer.pop()

                # 创建字典，用于送入推理模块
                data_dict = {
                    'remaining_image': received_data['mat2'],
                    'mask': received_data['bool_array'],
                    'img_mean': received_data['patch_means'],
                    'img_std': received_data['patch_stds']
                }

                data_dict_list.append(data_dict)

            logger.info("start infer")
            output = mae_infer.infer(data_dict_list)
            logger.info("infer completed")

            output_list = [
                np.clip(img[0].cpu().numpy().transpose(1, 2, 0), 0, 1) * 255
                for img in output
            ]
            output_list = [img.astype(np.uint8) for img in output_list]

            for i in range(len(output_list)):
                cv2.imwrite(f'output/{i}.png', cv2.cvtColor(output_list[i], cv2.COLOR_BGR2RGB))

        else:
            pass

    req_thread.join()
    pass

if __name__ == '__main__':
    main()