import mask

import time
import cv2
import zmq
import zlib
# import numpy as np

from loguru import logger

# 初始化 ZMQ 上下文和套接字
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://192.168.20.243:5555")  # 连接到服务器

# 打开摄像头或本地视频文件
cap = cv2.VideoCapture(0)  # 0 表示默认摄像头，也可以传入视频文件路径
patch_generator = mask.patch_generator()

while True:
    # 读取帧
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 将帧送入 mask 函数
    mat1, mat2, patch_means, patch_stds, bool_array = patch_generator.process_image(frame)

    cv2.imshow('mat1', mat1)
    cv2.imshow('mat2', mat2)

    # 使用 PIL 将 mat 数组压缩为 JPEG
    mat1_compressed = cv2.imencode('.jpg', mat1)[1].tobytes()
    mat2_compressed = cv2.imencode('.jpg', mat2)[1].tobytes()

    # 使用 zlib 压缩 bool 数组
    bool_array_compressed = zlib.compress(bool_array.tobytes())
    patch_stds_compressed = zlib.compress(patch_stds.tobytes())
    patch_means_compressed = zlib.compress(patch_means.tobytes())

    # 封装成一个对象
    data = {
        'mat1': mat1_compressed,                # 掩码后图
        'mat2': mat2_compressed,                # 剩余拼合图
        'bool_array': bool_array_compressed,    # 掩码
        'patch_means': patch_means_compressed,  # 均值
        'patch_stds': patch_stds_compressed     # 标准差
    }

    # print(mat1.shape, mat2.shape, bool_array.shape, patch_means.shape, patch_stds.shape)

    time.sleep(0.5)
    
    # 发送数据
    socket.send_pyobj(data)

    # 等待接收响应
    response = socket.recv_pyobj()
    logger.info(f"Received response: {response}")
    cv2.waitKey(1)

# 释放资源
cap.release()
socket.close()
context.term()