"""快速自检：不追求训练效果，只保证各部件形状与数值正确。

    python -m pytest tests -q
    python tests/test_smoke.py     # 没装 pytest 也能跑

全部是秒级用例，适合在正式训练前先确认环境没装错。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.algos import build_agent  # noqa: E402
from rl.buffers import ReplayBuffer  # noqa: E402
from rl.config import Config  # noqa: E402
from rl.envs import MountainCarEnv  # noqa: E402
from rl.normalizer import RunningMeanStd  # noqa: E402
from rl.utils import load_checkpoint  # noqa: E402


def test_running_mean_std():
    rms = RunningMeanStd(shape=(2,))
    data = np.random.default_rng(0).normal(loc=3.0, scale=2.0, size=(1000, 2))
    for row in data:
        rms.update(row)

    assert np.allclose(rms.mean, data.mean(axis=0), atol=1e-3)
    assert np.allclose(rms.var, data.var(axis=0), atol=1e-2)

    clone = RunningMeanStd(shape=(2,))
    clone.load_state_dict(rms.state_dict())
    assert np.allclose(clone.mean, rms.mean)


def test_buffer_wraps_around():
    buf = ReplayBuffer(capacity=3, obs_dim=2, act_dim=1, seed=0)
    for i in range(5):
        buf.add(
            np.full(2, i, dtype=np.float32),
            np.array([i], dtype=np.float32),
            float(i),
            np.full(2, i + 1, dtype=np.float32),
            0.0,
        )

    assert len(buf) == 3
    # 第 4、5 条覆盖了 slot 0 / slot 1
    assert buf.obs[0, 0] == 3.0
    assert buf.obs[1, 0] == 4.0
    assert buf.size == 3

    obs, act, rew, nobs, done = buf.sample(2)
    assert obs.shape == (2, 2)
    assert act.shape == (2, 1)
    assert rew.shape == (2, 1)
    assert nobs.shape == (2, 2)
    assert done.shape == (2, 1)


def test_env_api():
    env = MountainCarEnv(Config().env_id, seed=0)
    obs = env.reset()
    assert obs.shape == (2,)
    assert obs.dtype == np.float32

    next_obs, reward, terminated, truncated = env.step(np.array([0.0], dtype=np.float32))
    assert next_obs.shape == (2,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)

    # 归一化后不应该出现 NaN / inf
    assert np.all(np.isfinite(next_obs))
    env.close()


def _tiny_config(algo: str) -> Config:
    return Config(
        algo=algo,
        buffer_size=64,
        batch_size=8,
        learning_starts=8,
        hidden=32,
        episodes=1,
        eval_every=1000,
    )


def test_agents_update():
    env = MountainCarEnv(Config().env_id, seed=0)
    try:
        for algo in ("ddpg", "td3"):
            cfg = _tiny_config(algo)
            agent = build_agent(cfg, env, device="cpu")

            obs = env.reset()
            for _ in range(24):
                action = agent.select_action(obs, explore=True)
                next_obs, reward, terminated, truncated = env.step(action)
                agent.store(obs, action, reward * cfg.reward_scale, next_obs, terminated)
                obs = env.reset() if (terminated or truncated) else next_obs

            assert agent.ready(), f"{algo} 回放区没有就绪"

            for _ in range(4):
                info = agent.update()
                assert np.isfinite(info["critic_loss"]), f"{algo} critic loss 出现 NaN"
                assert info["critic_loss"] >= 0.0

            for name, param in agent.actor.named_parameters():
                assert torch.isfinite(param).all(), f"{algo} actor 参数 {name} 出现 NaN"
    finally:
        env.close()


def test_checkpoint_roundtrip():
    import tempfile

    env = MountainCarEnv(Config().env_id, seed=0)
    try:
        cfg = _tiny_config("td3")
        agent = build_agent(cfg, env, device="cpu")
        path = os.path.join(tempfile.mkdtemp(), "agent.pt")
        torch.save(agent.state_dict(), path)

        payload = load_checkpoint(path)
        restored = build_agent(cfg, env, device="cpu")
        restored.load_state_dict(payload)

        for (name_a, param_a), (name_b, param_b) in zip(
            agent.actor.named_parameters(), restored.actor.named_parameters()
        ):
            assert name_a == name_b
            assert torch.allclose(param_a, param_b), f"actor 参数 {name_a} 未正确恢复"
    finally:
        env.close()


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_")]
    for name, fn in sorted(tests):
        print(f"[run] {name}")
        fn()
        print(f"[ok ] {name}")
    print("\n全部自检通过。")
