"""集中管理所有超参数。

设计要点：
    * 全部字段都有默认值，命令行只覆盖用户显式传入的部分，
      所以 ``python train.py`` 不需要任何参数就能跑出一个合理配置。
    * 序列化用 ``to_dict()`` / ``from_dict()``，checkpoint 里存的就是它，
      保证训练与评估使用完全一致的配置（尤其 ``normalize_obs``）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Tuple


@dataclass
class Config:
    # ---------------- 运行 ----------------
    algo: str = "td3"
    env_id: str = "MountainCarContinuous-v0"
    seed: int = 0
    device: str = ""  # 留空 = 自动选择 cuda / cpu
    threads: int = 1  # 小网络单线程通常更快

    # ---------------- 训练预算 ----------------
    episodes: int = 300
    max_steps: int = 999

    # ---------------- 优化 ----------------
    gamma: float = 0.99
    tau: float = 0.005  # 目标网络软更新系数
    lr: float = 1e-3
    grad_clip: float = 10.0

    # ---------------- 经验回放 ----------------
    buffer_size: int = 100_000
    batch_size: int = 256
    learning_starts: int = 1_000
    updates_per_step: int = 1

    # ---------------- 网络与观测 ----------------
    hidden: int = 256
    normalize_obs: bool = True
    reward_scale: float = 0.1  # +100 的稀疏奖励压到 Q 网络舒服的区间

    # ---------------- 探索 ----------------
    explore_sigma: float = 0.15  # 独立高斯噪声，不衰减

    # ---------------- TD3 专属 ----------------
    policy_delay: int = 2
    policy_noise: float = 0.2
    noise_clip: float = 0.5

    # ---------------- 评估与早停 ----------------
    eval_every: int = 20
    eval_episodes: int = 10
    target_return: float = 90.0
    patience: int = 2
    early_stop: bool = True

    # ---------------- 落盘 ----------------
    save_every: int = 50
    out_dir: str = "runs"
    tag: str = ""
    resume: str = ""

    # ---------------------------------------------------------------
    @property
    def hidden_sizes(self) -> Tuple[int, int]:
        return (int(self.hidden), int(self.hidden))

    @property
    def run_name(self) -> str:
        return self.tag or f"{self.algo}_s{self.seed}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Config":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in payload.items() if k in known})

    def summary(self) -> str:
        lines = ["=" * 62, "配置", "=" * 62]
        for key, value in self.to_dict().items():
            lines.append(f"  {key:<18} {value}")
        lines.append("=" * 62)
        return "\n".join(lines)
