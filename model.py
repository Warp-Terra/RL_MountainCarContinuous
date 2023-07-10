import torch
import torch.nn as nn
# 导入神经网络模块

class Model(nn.Module):
# 构建一个神经网络
    def __init__(self, input_dim, output_dim):
        # 初始化，包含输入和输出的维度
        super().__init__()
        lin1 = nn.Linear(input_dim, 128)
        torch.nn.init.xavier_normal_(lin1.weight)
        lin2 = nn.Linear(128, 64)
        torch.nn.init.xavier_normal_(lin2.weight)
        lin3 = nn.Linear(64, output_dim)
        torch.nn.init.xavier_normal_(lin3.weight)
        # 定义三层神经网络
        self.layers = nn.Sequential(lin1, nn.ReLU(), lin2, nn.ReLU(), lin3)
        # 组合神经网络,用ReLU作为激活函数

    def forward(self, input):
        return self.layers(input)
    # 将神经网络从输入转化为输出


class Critic(Model):
    # 继承至model类

    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim + action_dim, 1)
    # 初始化:把状态和动作的维度之和作为输入,然后输出1

    def forward(self, x, a):
        return super().forward(torch.cat((x, a), 1))
    # 将输入输出合并在一起,向前输出


class Actor(Model):
    # 定义一个新的actor类,继承自model

    def __init__(self, state_dim):
        super().__init__(state_dim, 1)
        # 初始化:把状态作为输入,输出1

    def forward(self, x):
        return torch.tanh(super().forward(x))
    # 向前输出x,使用tanh函数,把范围固定在-1到1