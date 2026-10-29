"""探索噪声。

第一版用的是 OU 过程，并且 sigma 在约 10 万步内衰减到 1e-4。
两个问题：
    * OU 噪声强相关，反而限制了策略去发现「来回摆动」这种周期性行为；
    * 衰减得太早，还没学会摆动探索就关了。

MountainCarContinuous 的最优解是一个极限环（不断从谷底泵能量），
所以这里默认用**不衰减的独立高斯噪声**。OU 保留下来做消融对比。
"""

from __future__ import annotations

import numpy as np


class GaussianNoise:
    """独立高斯噪声：简单、无相关性、实现无歧义。"""

    def __init__(self, act_dim: int, sigma: float = 0.15, seed=None):
        self.act_dim = int(act_dim)
        self.sigma = float(sigma)
        self.rng = np.random.default_rng(seed)

    def sample(self) -> np.ndarray:
        return (self.rng.standard_normal(self.act_dim) * self.sigma).astype(np.float32)

    def state_dict(self) -> dict:
        return {"state": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict) -> None:
        try:
            self.rng.bit_generator.state = state["state"]
        except Exception:  # pragma: no cover - 字典结构不符时直接忽略
            pass


class OUNoise:
    """Ornstein-Uhlenbeck 过程，仅用于消融实验。"""

    def __init__(self, act_dim: int, mu=0.0, theta=0.15, sigma=0.2, decay=1.0, seed=None):
        self.act_dim = int(act_dim)
        self.mu = float(mu)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.decay = float(decay)  # 每步乘性衰减，1.0 表示不衰减
        self.rng = np.random.default_rng(seed)
        self.state = np.ones(self.act_dim, dtype=np.float64) * self.mu

    def reset(self) -> None:
        self.state = np.ones(self.act_dim, dtype=np.float64) * self.mu

    def sample(self) -> np.ndarray:
        self.state = self.state + self.theta * (self.mu - self.state) + self.sigma * self.rng.standard_normal(
            self.act_dim
        )
        self.sigma *= self.decay
        return self.state.astype(np.float32)
