import random


class Buffer:
    def __init__(self, cap):
        self.cap = cap
        # 表示缓冲区的最大容量
        self.mem = []
        # 创建一个空列表，用于储存经验
        self.pos = -1
        # 用于标记最后一个写入元素的位置

    def __len__(self):
        return len(self.mem)
    # 返回储存区的元素个数

    def add(self, element):
        if len(self.mem) < self.cap:
            self.mem.append(None)
        new_pos = (self.pos + 1) % self.cap
        self.mem[new_pos] = element
        self.pos = new_pos
        # 本方法用于向缓冲区添加新的经验，如果没有满，则在末尾添加一个None。然后计算新的位置，并将新的经验元素储存在该位置

    def sample(self, batch_size):
        return random.sample(self.mem, batch_size)
    # 用于随机取样，随机选择batch_size个样本

    def __getitem__(self, k):
        return self.mem[(self.pos + 1 + k) % self.cap]
    # 用于索引，取元素