import zmq
import zlib
import numpy as np
import cv2


class require:
    def __init__(self, port):
        # 初始化 ZMQ 上下文和套接字
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:5555")  # 绑定到端口
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

        # data = {
        #     'mat1': mat1_compressed,            # 掩码后图
        #     'mat2': mat2_compressed,            # 剩余拼合图
        #     'patch_means': patch_means,         # 均值
        #     'patch_stds': patch_stds,           # 标准差
        #     'bool_array': bool_array_compressed # 掩码
        # }
    

        # 解压缩图像
        mat1 = cv2.imdecode(np.frombuffer(masked_image, dtype=np.uint8), cv2.IMREAD_COLOR)
        mat2 = cv2.imdecode(np.frombuffer(combined_image, dtype=np.uint8), cv2.IMREAD_COLOR)

        # 解压缩 bool 数组
        bool_array = np.frombuffer(zlib.decompress(bool_array_compressed), dtype=bool)

        # 返回解压缩后的数据
        return {'mat1': mat1, 'mat2': mat2, 'bool_array': bool_array}
    
    def __del__(self):
        # 关闭套接字
        self.socket.close()
        self.context.term()
        pass