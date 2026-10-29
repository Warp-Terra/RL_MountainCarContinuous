"""算法注册表。"""

from __future__ import annotations

from .base import OffPolicyAgent
from .ddpg import DDPG
from .td3 import TD3

ALGOS = {"ddpg": DDPG, "td3": TD3}

__all__ = ["ALGOS", "OffPolicyAgent", "DDPG", "TD3", "build_agent"]


def build_agent(cfg, env, device=None):
    key = str(cfg.algo).lower()
    if key not in ALGOS:
        raise ValueError(f"未知算法 {cfg.algo!r}，可选: {sorted(ALGOS)}")
    return ALGOS[key](cfg, env, device=device)
