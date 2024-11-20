import subprocess
import threading
import cv2
import numpy as np
import zlib
from loguru import logger

from backend_require import require

OUTPUT_DIR='./output'
IMAGE_PATH= None
MODEL_PATH='./model/checkpoint-1419.pth'
REMAINING_IMG_PATH='./input/combined_image.npy'
MASK_PATH='./input/mask_positions.npy'
IMG_MEAN_PATH='./input/patch_means.npy'
IMG_STD_PATH='./input/patch_stds.npy'
MASK_RATIO=0.75

received_data = None

def main():

    req = require(5555)
    req_thread = threading.Thread(target = req.worker)
    logger.info("start req thread")
    req_thread.start()

    while True:
        if len(req.buffer) > 0:
            received_data = req.buffer.pop()

            mat1 = received_data['mat1']
            mat2 = received_data['mat2']
            bool_array = received_data['bool_array']
            patch_means = received_data['patch_means']
            patch_stds = received_data['patch_stds']

            np.save('./input/combined_image.npy', mat2)
            np.save('./input/mask_positions.npy', bool_array)
            np.save('./input/patch_means.npy', patch_means)
            np.save('./input/patch_stds.npy', patch_stds)

            # 创建运行命令，和eval.sh一样
            try:
                # 运行命令
                result = subprocess.run(['bash', 'eval.sh'], check=True)
                logger.error(f"exec result: {result}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"exec failed and return: {e}")

            # cv2.imshow('image', received_data['mat2'])
            # cv2.waitKey(1)
        else:
            pass

    req_thread.join()
    pass

if __name__ == '__main__':
    main()