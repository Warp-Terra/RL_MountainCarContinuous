"""在线观测归一化（Welford / 并行方差合并）。

MountainCarContinuous 的观测是 ``[position, velocity]``：
position 范围约 ±1，velocity 范围只有 ±0.07，两者相差一个数量级以上。
不做缩放，网络输入层几乎只能感知到位置，而「该往哪边加速」恰恰要靠速度判断。
"""

from __future__ import annotations

import numpy as np


class RunningMeanStd:
    """流式估计均值/方差，支持单条观测和整批观测。"""

    def __init__(self, shape=(), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.shape == self.mean.shape:
            x = x[None, ...]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = float(x.shape[0])

        delta = batch_mean - self.mean
        total = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total

        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": float(self.count),
        }

    def load_state_dict(self, state: dict) -> None:
        self.mean = np.array(state["mean"], dtype=np.float64)
        self.var = np.array(state["var"], dtype=np.float64)
        self.count = float(state["count"])

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"RunningMeanStd(mean={self.mean}, var={self.var}, count={self.count:.1f})"
