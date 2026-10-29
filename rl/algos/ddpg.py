"""DDPG：作为基线保留，用来和 TD3 做对比。"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from ..networks import Critic
from ..utils import soft_update
from .base import OffPolicyAgent


class DDPG(OffPolicyAgent):
    algo = "ddpg"

    def __init__(self, cfg, env, device=None):
        super().__init__(cfg, env, device)

        self.critic = Critic(self.obs_dim, self.act_dim, cfg.hidden_sizes).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for param in self.critic_target.parameters():
            param.requires_grad_(False)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

    def update(self) -> dict:
        obs, act, rew, nobs, done = self.sample_batch()

        # ---- critic ----
        with torch.no_grad():
            q_next = self.critic_target(nobs, self.actor_target(nobs))
            # (1 - done)：终止转移不 bootstrap
            target = rew + self.cfg.gamma * (1.0 - done) * q_next

        q_value = self.critic(obs, act)
        critic_loss = F.mse_loss(q_value, target)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._clip(self.critic)
        self.critic_opt.step()

        # ---- actor ----
        # 先清掉 critic 的梯度：下面这次 backward 也会给 critic 留下 grad，
        # 第一版是靠「下一轮 zero_grad」才侥幸没出事的，这里显式隔离。
        self.critic.zero_grad(set_to_none=True)
        actor_loss = -self.critic(obs, self.actor(obs)).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._clip(self.actor)
        self.actor_opt.step()

        # ---- targets ----
        self.update_targets()
        self.updates += 1

        return {
            "critic_loss": float(critic_loss.detach()),
            "actor_loss": float(actor_loss.detach()),
        }

    # ------------------------------------------------------------------
    def _soft_update_critics(self) -> None:
        soft_update(self.critic_target, self.critic, self.cfg.tau)

    def _extra_state(self) -> dict:
        return {
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }

    def _load_extra(self, payload: dict) -> None:
        self.critic.load_state_dict(payload["critic"])
        self.critic_target.load_state_dict(payload["critic_target"])
        self.critic_opt.load_state_dict(payload["critic_opt"])
