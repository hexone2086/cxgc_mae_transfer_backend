import collections
import threading
import time

class ThreadSafeBuffer:
    def __init__(self, max_size=None):
        self.buffer = collections.deque(maxlen=max_size)
        self.lock = threading.Lock()

    def append(self, item):
        with self.lock:
            self.buffer.append(item)

    def appendleft(self, item):
        with self.lock:
            self.buffer.appendleft(item)

    def pop(self):
        with self.lock:
            if self.buffer:
                return self.buffer.pop()
            else:
                return None

    def popleft(self):
        with self.lock:
            if self.buffer:
                return self.buffer.popleft()
            else:
                return None

    def __len__(self):
        with self.lock:
            return len(self.buffer)

    def __iter__(self):
        with self.lock:
            return iter(self.buffer)


if __name__ == '__main__':

    # 示例使用
    buffer = ThreadSafeBuffer(max_size=10)

    # 生产者线程
    def producer():
        for i in range(20):
            buffer.append(i)
            print(f"Produced: {i}")

    # 消费者线程
    def consumer():
        while True:
            item = buffer.popleft()
            print(f"Consumed: {item}")

    # 创建并启动线程
    producer_thread = threading.Thread(target=producer)
    consumer_thread = threading.Thread(target=consumer)

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()