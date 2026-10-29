"""DDPG / TD3 的公共部分：动作选择、经验存储、目标网络、checkpoint。"""

from __future__ import annotations

import copy

import numpy as np
import torch

from ..buffers import ReplayBuffer
from ..networks import Actor
from ..noise import GaussianNoise
from ..utils import soft_update


class OffPolicyAgent:
    algo = "base"

    def __init__(self, cfg, env, device=None):
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.obs_dim = int(env.obs_dim)
        self.act_dim = int(env.act_dim)

        # 动作边界：网络输出 tanh(-1..1) 之后映射到真实区间
        self._low_np = np.asarray(env.action_low, dtype=np.float32).reshape(self.act_dim)
        self._high_np = np.asarray(env.action_high, dtype=np.float32).reshape(self.act_dim)
        self.action_low = torch.as_tensor(self._low_np, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(self._high_np, dtype=torch.float32, device=self.device)
        scale = (self._high_np - self._low_np) / 2.0
        bias = (self._high_np + self._low_np) / 2.0

        self.actor = Actor(self.obs_dim, self.act_dim, scale, bias, cfg.hidden_sizes).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        for param in self.actor_target.parameters():
            param.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)

        self.buffer = ReplayBuffer(cfg.buffer_size, self.obs_dim, self.act_dim, seed=cfg.seed)
        self.noise = GaussianNoise(self.act_dim, cfg.explore_sigma, seed=cfg.seed + 1)

        self.total_steps = 0
        self.updates = 0

    # ------------------------------------------------------------------ 交互
    def select_action(self, obs, explore: bool = True) -> np.ndarray:
        obs_t = torch.as_tensor(
            np.asarray(obs, dtype=np.float32), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            act = self.actor(obs_t).squeeze(0).cpu().numpy()
        if explore:
            act = act + self.noise.sample()
        return np.clip(act, self._low_np, self._high_np).astype(np.float32)

    def store(self, obs, act, rew, next_obs, terminated) -> None:
        # 只把「真终止」记成 1；时间上限截断记 0，保证仍然 bootstrap
        self.buffer.add(obs, act, rew, next_obs, float(terminated))

    def ready(self) -> bool:
        return self.buffer.size >= max(int(self.cfg.batch_size), int(self.cfg.learning_starts))

    def sample_batch(self):
        obs, act, rew, nobs, done = self.buffer.sample(self.cfg.batch_size)
        as_tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)  # noqa: E731
        return as_tensor(obs), as_tensor(act), as_tensor(rew), as_tensor(nobs), as_tensor(done)

    # ------------------------------------------------------------------ 更新
    def update_targets(self) -> None:
        soft_update(self.actor_target, self.actor, self.cfg.tau)
        self._soft_update_critics()

    def _soft_update_critics(self) -> None:
        raise NotImplementedError

    def _clip(self, module) -> None:
        if self.cfg.grad_clip and self.cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(module.parameters(), self.cfg.grad_clip)

    def update(self) -> dict:  # pragma: no cover - 由子类实现
        raise NotImplementedError

    # ------------------------------------------------------------------ 存档
    def state_dict(self) -> dict:
        payload = {
            "algo": self.algo,
            "cfg": self.cfg.to_dict(),
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "buffer": self.buffer.state_dict(),
            "noise": self.noise.state_dict(),
            "total_steps": self.total_steps,
            "updates": self.updates,
        }
        payload.update(self._extra_state())
        return payload

    def load_state_dict(self, payload: dict) -> None:
        self.actor.load_state_dict(payload["actor"])
        self.actor_target.load_state_dict(payload["actor_target"])
        self.actor_opt.load_state_dict(payload["actor_opt"])
        if "buffer" in payload:
            self.buffer.load_state_dict(payload["buffer"])
        if "noise" in payload:
            self.noise.load_state_dict(payload["noise"])
        self.total_steps = int(payload.get("total_steps", 0))
        self.updates = int(payload.get("updates", 0))
        self._load_extra(payload)

    def _extra_state(self) -> dict:
        return {}

    def _load_extra(self, payload: dict) -> None:
        return None
