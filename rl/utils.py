"""杂项工具：随机种子、目标网络更新、checkpoint。"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    """Polyak 平均。用 no_grad 包住，而不是操作 ``.data``。"""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.mul_(1.0 - tau).add_(sp, alpha=tau)


@torch.no_grad()
def hard_update(target: torch.nn.Module, source: torch.nn.Module) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.copy_(sp)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def rng_state() -> dict:
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}


def set_rng_state(state) -> None:
    if not state:
        return
    try:
        torch.set_rng_state(state["torch"])
        np.random.set_state(state["numpy"])
    except Exception:  # pragma: no cover - 跨版本/跨设备时尽力恢复
        pass


def save_checkpoint(payload: dict, path: str) -> str:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(path: str, map_location="cpu") -> dict:
    """兼容 torch 2.6 起 ``weights_only`` 默认为 True 的行为。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch < 1.13 不认识这个参数
        return torch.load(path, map_location=map_location)


def human_time(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    return f"{int(minutes)}m{sec:04.1f}s"
