"""Actor / Critic / 双 Q。

相对第一版的改动：
    * ``xavier_normal_`` -> He/Kaiming，配 ReLU 才是正确的缩放；
    * bias 显式置零，而不是留着 PyTorch 的默认均匀分布；
    * Actor 的动作缩放来自 ``action_space``，不再硬编码 [-1, 1]；
    * Actor 最后一层用 3e-3 的小初始化，初始动作 ≈ 0。
      这一点对 MountainCarContinuous 特别关键：如果初始策略一上来就饱和到 ±1，
      小车会被一直往一个方向推，永远摆不起来。
"""

from __future__ import annotations

import torch
import torch.nn as nn


def mlp(input_dim: int, output_dim: int, hidden=(256, 256), last_init_scale=None) -> nn.Sequential:
    layers = []
    prev = int(input_dim)
    for width in hidden:
        width = int(width)
        lin = nn.Linear(prev, width)
        nn.init.kaiming_uniform_(lin.weight, nonlinearity="relu")
        nn.init.zeros_(lin.bias)
        layers.extend([lin, nn.ReLU()])
        prev = width

    last = nn.Linear(prev, int(output_dim))
    if last_init_scale is not None:
        nn.init.uniform_(last.weight, -last_init_scale, last_init_scale)
        nn.init.uniform_(last.bias, -last_init_scale, last_init_scale)
    else:
        nn.init.xavier_uniform_(last.weight)
        nn.init.zeros_(last.bias)
    layers.append(last)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """确定性策略，输出经过 tanh 后被映射到真实动作区间。"""

    def __init__(self, obs_dim, act_dim, action_scale, action_bias, hidden=(256, 256)):
        super().__init__()
        self.net = mlp(obs_dim, act_dim, hidden, last_init_scale=3e-3)
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(action_bias, dtype=torch.float32))

    def forward(self, obs):
        return torch.tanh(self.net(obs)) * self.action_scale + self.action_bias


class Critic(nn.Module):
    """Q(s, a)：把状态和动作拼接后过 MLP。"""

    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.net = mlp(int(obs_dim) + int(act_dim), 1, hidden)

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


class TwinCritic(nn.Module):
    """TD3 的两个独立 Q，取 min 抑制高估。"""

    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.q1 = Critic(obs_dim, act_dim, hidden)
        self.q2 = Critic(obs_dim, act_dim, hidden)

    def forward(self, obs, act):
        return self.q1(obs, act), self.q2(obs, act)
