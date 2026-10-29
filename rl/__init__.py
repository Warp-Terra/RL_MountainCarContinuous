"""MountainCarContinuous 上的 DDPG / TD3 实现（重构版）。

第二版把第一版的单文件脚本拆成可复用的包：
    rl.envs        环境封装 + 观测归一化 + gym/gymnasium 兼容
    rl.buffers     numpy 版经验回放
    rl.networks    Actor / Critic
    rl.algos       DDPG / TD3
    train.py       训练入口
    eval.py        加载模型做确定性评估

第一版的教学实现（training.py / model.py / buffer.py / noise.py）保留在
仓库根目录作对照，见 README 的「版本说明」。
"""

from __future__ import annotations

from .config import Config

__all__ = ["Config"]
__version__ = "2.0.0"
