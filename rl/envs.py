"""环境封装：gym / gymnasium 双兼容 + 观测归一化 + 动作边界。

第一版里三处硬伤在这里一次性解决：
    1. ``state, reward, done, _ = env.step(a)`` 只认 gym 0.23 的四元组，
       换到 gymnasium 直接崩。这里统一成 ``(obs, reward, terminated, truncated)``。
    2. 观测不做归一化。这里在 wrapper 层做，buffer 里存的**已经是**归一化后的值，
       彻底避免「训练用归一化、评估忘了归一化」这类不一致。
    3. 动作范围硬编码假设 [-1, 1]。这里从 ``action_space`` 读真实上下界。
"""

from __future__ import annotations

import numpy as np

from .normalizer import RunningMeanStd

try:  # pragma: no cover - 取决于本机装的是哪一个
    import gymnasium as gym

    GYM_BACKEND = "gymnasium"
except ImportError:  # pragma: no cover
    import gym

    GYM_BACKEND = "gym"


class MountainCarEnv:
    """把任意 gym 版本的环境包成统一接口。"""

    def __init__(
        self,
        env_id: str = "MountainCarContinuous-v0",
        seed=None,
        normalize_obs: bool = True,
        clip: float = 5.0,
        render_mode: str = None,
    ):
        self.env_id = env_id
        self._manual_render = False
        self._render_enabled = render_mode is not None

        if render_mode is None:
            self.raw = gym.make(env_id)
        else:
            try:
                self.raw = gym.make(env_id, render_mode=render_mode)
            except TypeError:
                # gym < 0.26 不认识 render_mode，退回手动 render()
                self.raw = gym.make(env_id)
                self._manual_render = True

        self.observation_space = self.raw.observation_space
        self.action_space = self.raw.action_space

        self.obs_dim = int(np.prod(self.observation_space.shape))
        self.act_dim = int(np.prod(self.action_space.shape))

        self.action_low = np.asarray(self.action_space.low, dtype=np.float32).reshape(self.act_dim)
        self.action_high = np.asarray(self.action_space.high, dtype=np.float32).reshape(self.act_dim)

        spec = getattr(self.raw, "spec", None)
        self.max_episode_steps = getattr(spec, "max_episode_steps", None)

        self.clip = float(clip)
        self.normalize_obs = bool(normalize_obs)
        # track=False 时只读统计量、不更新，供评估使用
        self.track = True
        self.rms = RunningMeanStd(self.observation_space.shape) if self.normalize_obs else None

        self._pending_seed = int(seed) if seed is not None else None

    # ------------------------------------------------------------------
    def seed(self, seed) -> "MountainCarEnv":
        """下一次 reset 时生效，可以反复调用（评估时每个 episode 序列前重设）。"""
        self._pending_seed = None if seed is None else int(seed)
        return self

    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        seed, self._pending_seed = self._pending_seed, None

        if seed is None:
            out = self.raw.reset()
        else:
            # 先试老接口 seed() + reset()，再退回新接口 reset(seed=)。
            # 顺序很关键：反过来会踩到「reset 接受 **kwargs 但静默忽略 seed」的坑，
            # 那样种子根本没生效，还不报错。
            legacy = False
            try:
                self.raw.seed(seed)
                legacy = True
            except (AttributeError, TypeError, NotImplementedError):
                legacy = False

            out = self.raw.reset() if legacy else self.raw.reset(seed=seed)

        obs = out[0] if isinstance(out, tuple) else out
        return self._process(obs)

    def step(self, action):
        act = np.asarray(action, dtype=np.float32).reshape(self.act_dim)
        act = np.clip(act, self.action_low, self.action_high)

        out = self.raw.step(act)
        if len(out) == 5:  # gymnasium / gym >= 0.26
            obs, reward, terminated, truncated, info = out
            terminated = bool(terminated)
            truncated = bool(truncated)
        else:  # gym < 0.26：四元组，截断信息藏在 info 里
            obs, reward, done, info = out
            truncated = bool(info.get("TimeLimit.truncated", False))
            terminated = bool(done) and not truncated

        if self._render_enabled and self._manual_render:
            self.raw.render()

        return self._process(obs), float(reward), terminated, truncated

    # ------------------------------------------------------------------
    def _process(self, obs) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(self.obs_dim)
        if self.rms is None:
            return obs
        if self.track:
            self.rms.update(obs)
        scaled = (obs - self.rms.mean) / np.sqrt(self.rms.var + 1e-8)
        return np.clip(scaled, -self.clip, self.clip).astype(np.float32)

    def render(self):
        self.raw.render()

    def close(self) -> None:
        try:
            self.raw.close()
        except Exception:  # pragma: no cover - 老版本 gym 可能没有 close
            pass

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"MountainCarEnv({self.env_id!r}, backend={GYM_BACKEND}, "
            f"obs_dim={self.obs_dim}, act_dim={self.act_dim})"
        )


def make_env(env_id: str = "MountainCarContinuous-v0", seed=None, **kwargs) -> MountainCarEnv:
    return MountainCarEnv(env_id, seed=seed, **kwargs)
