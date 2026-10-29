"""预分配 ndarray 的经验回放。

第一版用 ``list`` 存元组、``random.sample`` 取样，再对每个字段
``np.array(...)`` 重新拼装，每一步训练都要付一次 Python 层的搬运开销。
这里把五个字段各自存成定长 ndarray，采样退化成一次花式索引，
``torch.as_tensor`` 可以零拷贝接住 float32 的结果。
"""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, seed: int = 0):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        self.obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.act = np.zeros((self.capacity, self.act_dim), dtype=np.float32)
        self.rew = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        # done = 1.0 只表示「真正的终止」，时间上限截断必须留 0，
        # 否则会把终止后的价值错误地截断（或反过来把截断当成终止）。
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def add(self, obs, act, rew, next_obs, done) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.done[i] = done
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=int(batch_size))
        return (
            self.obs[idx],
            self.act[idx],
            self.rew[idx],
            self.next_obs[idx],
            self.done[idx],
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.size

    def state_dict(self) -> dict:
        return {
            "obs": self.obs,
            "act": self.act,
            "rew": self.rew,
            "next_obs": self.next_obs,
            "done": self.done,
            "ptr": self.ptr,
            "size": self.size,
        }

    def load_state_dict(self, state: dict) -> None:
        """容量不一致时按较小者拷贝，保证续训不会因改配置而崩。"""
        n = min(self.capacity, int(len(state["obs"])))
        self.obs[:n] = state["obs"][:n]
        self.act[:n] = state["act"][:n]
        self.rew[:n] = state["rew"][:n]
        self.next_obs[:n] = state["next_obs"][:n]
        self.done[:n] = state["done"][:n]
        self.ptr = int(state["ptr"]) % self.capacity
        self.size = min(int(state["size"]), self.capacity)
