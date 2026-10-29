#!/usr/bin/env python
"""加载训练好的 policy.pt，跑确定性评估。

    python eval.py --ckpt runs/td3_s0/policy.pt --episodes 20
    python eval.py --ckpt runs/td3_s0/policy.pt --render
"""

from __future__ import annotations

import argparse

import numpy as np

from rl.algos import build_agent
from rl.config import Config
from rl.envs import MountainCarEnv
from rl.utils import load_checkpoint, set_seed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="评估 MountainCarContinuous 策略")
    parser.add_argument("--ckpt", required=True, help="训练产出的 policy.pt 路径")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render", action="store_true", help="开窗口看小车跑（需要 pygame 和图形环境）")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="渲染帧率上限，配合 --render 放慢观察（默认跟随环境元数据，通常是 30）",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)

    payload = load_checkpoint(args.ckpt, map_location=args.device)
    cfg = Config.from_dict(payload.get("cfg") or {})
    cfg.buffer_size = 1  # 评估不需要回放区
    cfg.device = args.device

    env = MountainCarEnv(
        cfg.env_id,
        seed=args.seed,
        normalize_obs=cfg.normalize_obs,
        render_mode="human" if args.render else None,
    )

    if args.render and args.fps:
        # gymnasium 的经典控制环境在 render() 里用 clock.tick(render_fps) 限速，
        # 改这个值就能放慢（或加快）播放速度。
        try:
            env.raw.metadata["render_fps"] = float(args.fps)
        except Exception:
            print("[warn] 无法修改渲染帧率，沿用环境默认值")

    normalizer = payload.get("obs_normalizer")
    if normalizer is not None and env.rms is not None:
        env.rms.load_state_dict(normalizer)
    env.track = False

    agent = build_agent(cfg, env, device=args.device)
    agent.actor.load_state_dict(payload["actor"])
    agent.actor.eval()

    returns, lengths, successes = [], [], []
    for index in range(int(args.episodes)):
        obs = env.reset()
        total, steps, reached = 0.0, 0, False
        while steps < int(cfg.max_steps):
            action = agent.select_action(obs, explore=False)
            obs, reward, terminated, truncated = env.step(action)
            total += reward
            steps += 1
            reached = reached or terminated
            if terminated or truncated:
                break

        returns.append(total)
        lengths.append(steps)
        successes.append(1.0 if reached else 0.0)
        print(f"  episode {index + 1:>3}: return={total:8.2f}  steps={steps:>4}  reached={reached}")

    returns = np.asarray(returns, dtype=np.float64)
    print("-" * 52)
    print(f"  episodes    : {len(returns)}")
    print(f"  mean return : {returns.mean():.2f} ± {returns.std():.2f}")
    print(f"  max return  : {returns.max():.2f}")
    print(f"  success rate: {np.mean(successes) * 100:.1f}%")
    print(f"  mean length : {np.mean(lengths):.1f}")
    print("-" * 52)

    env.close()
    return {
        "mean_return": float(returns.mean()),
        "success_rate": float(np.mean(successes)),
        "mean_length": float(np.mean(lengths)),
    }


if __name__ == "__main__":
    main()
