"""TD3：在 DDPG 上补三个机制，稳定性提升明显，计算量几乎不变。

    1. 双 Q 取 min          -> 抑制 Q 高估
    2. 目标策略平滑          -> 防止 Q 在尖峰上过拟合
    3. 延迟策略更新          -> 等 critic 先收敛一些再动 actor
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F

from ..networks import TwinCritic
from ..utils import soft_update
from .base import OffPolicyAgent


class TD3(OffPolicyAgent):
    algo = "td3"

    def __init__(self, cfg, env, device=None):
        super().__init__(cfg, env, device)

        self.critic = TwinCritic(self.obs_dim, self.act_dim, cfg.hidden_sizes).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for param in self.critic_target.parameters():
            param.requires_grad_(False)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

    def update(self) -> dict:
        self.updates += 1

        obs, act, rew, nobs, done = self.sample_batch()

        # ---- critic ----
        with torch.no_grad():
            next_act = self.actor_target(nobs)
            noise = (torch.randn_like(next_act) * self.cfg.policy_noise).clamp(
                -self.cfg.noise_clip, self.cfg.noise_clip
            )
            next_act = (next_act + noise).clamp(self.action_low, self.action_high)

            q1_next, q2_next = self.critic_target(nobs, next_act)
            # (1 - done)：只对真终止做掩码，截断继续 bootstrap
            target = rew + self.cfg.gamma * (1.0 - done) * torch.min(q1_next, q2_next)

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._clip(self.critic)
        self.critic_opt.step()

        # ---- actor（延迟更新）----
        actor_loss_value = math.nan
        if self.updates % max(int(self.cfg.policy_delay), 1) == 0:
            self.critic.zero_grad(set_to_none=True)
            actor_loss = -self.critic.q1(obs, self.actor(obs)).mean()

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self._clip(self.actor)
            self.actor_opt.step()

            actor_loss_value = float(actor_loss.detach())
            self.update_targets()  # target 也跟着延迟更新

        return {"critic_loss": float(critic_loss.detach()), "actor_loss": actor_loss_value}

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
