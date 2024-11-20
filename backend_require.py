import zmq
import zlib
import numpy as np
import cv2
from loguru import logger

from backend_utils import ThreadSafeBuffer


class require:
    def __init__(self, port):
        # 初始化rb
        self.buffer = ThreadSafeBuffer(max_size=10)

        # 初始化 ZMQ 上下文和套接字
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:" + str(port))  # 绑定到端口
        logger.info(f"open tcp server at localhost:{port}")
        pass
    def recv_pyobj(self):
        # 接收数据
        data = self.socket.recv_pyobj()

        # 解压缩数据
        masked_image = data['mat1']
        combined_image = data['mat2']
        mask_matrix = data['bool_array']
        patch_means = data['patch_means']
        patch_stds = data['patch_stds']

        # 解压缩图像
        mat1 = cv2.imdecode(np.frombuffer(masked_image, dtype=np.uint8), cv2.IMREAD_COLOR)
        mat2 = cv2.imdecode(np.frombuffer(combined_image, dtype=np.uint8), cv2.IMREAD_COLOR)

        # 解压缩 bool 数组

        mask_matrix = np.frombuffer(zlib.decompress(mask_matrix), dtype=bool)
        # logger.debug("mask_matrix shape: " + str(mask_matrix.shape))
        mask_matrix = mask_matrix.reshape((14, 14))
        patch_means = np.frombuffer(zlib.decompress(patch_means), dtype=np.float32)
        # logger.debug("patch_means shape: " + str(patch_means.shape))
        patch_means = patch_means.reshape((1, 196, 1, 3))
        patch_stds = np.frombuffer(zlib.decompress(patch_stds), dtype=np.float32)
        patch_stds = patch_stds.reshape((1, 196, 1, 3))
        self.socket.send_pyobj("OK")


        # 返回解压缩后的数据
        return {'mat1': mat1, 'mat2': mat2, 'bool_array': mask_matrix, 'patch_means': patch_means, 'patch_stds': patch_stds}
    
    def worker(self):
        while True:
            data = self.recv_pyobj()
            self.buffer.append(data) # 加入缓冲区
    
    def __del__(self):
        # 关闭套接字
        self.socket.close()
        self.context.term()
        pass

if __name__ == '__main__':

    req = require(5555)
    while True:
        data = req.recv_pyobj()
        print(data)
