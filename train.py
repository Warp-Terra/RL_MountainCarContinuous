#!/usr/bin/env python
"""在 MountainCarContinuous 上训练 DDPG / TD3。

常用：
    python train.py                                  # 默认 td3, seed=0
    python train.py --algo ddpg --tag ddpg_baseline  # 跑 DDPG 基线做对比
    python train.py --seed 3 --episodes 400          # 换种子 / 加预算
    python train.py --resume runs/td3_s0/last_state.pt

进度会写进 runs/<run>/train.csv，用 tools/plot_results.py 画学习曲线。
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from rl.algos import build_agent
from rl.config import Config
from rl.envs import MountainCarEnv
from rl.logger import CSVLogger
from rl.utils import human_time, load_checkpoint, rng_state, save_checkpoint, set_rng_state, set_seed

FIELDS = [
    "episode",
    "step",
    "ep_return",
    "ep_len",
    "critic_loss",
    "actor_loss",
    "eval_return",
    "eval_std",
    "eval_len",
    "eval_success",
    "elapsed_s",
]


# ---------------------------------------------------------------------- 参数
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="MountainCarContinuous 上的 DDPG / TD3 训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--algo", choices=["ddpg", "td3"], default=None)
    parser.add_argument("--env-id", dest="env_id", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--grad-clip", dest="grad_clip", type=float, default=None)
    parser.add_argument("--buffer-size", dest="buffer_size", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--learning-starts", dest="learning_starts", type=int, default=None)
    parser.add_argument("--updates-per-step", dest="updates_per_step", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--reward-scale", dest="reward_scale", type=float, default=None)
    parser.add_argument("--explore-sigma", dest="explore_sigma", type=float, default=None)
    parser.add_argument("--policy-delay", dest="policy_delay", type=int, default=None)
    parser.add_argument("--policy-noise", dest="policy_noise", type=float, default=None)
    parser.add_argument("--noise-clip", dest="noise_clip", type=float, default=None)
    parser.add_argument("--eval-every", dest="eval_every", type=int, default=None)
    parser.add_argument("--eval-episodes", dest="eval_episodes", type=int, default=None)
    parser.add_argument("--target-return", dest="target_return", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--no-early-stop", dest="early_stop", action="store_false", default=None)
    parser.add_argument("--save-every", dest="save_every", type=int, default=None)
    parser.add_argument("--out-dir", dest="out_dir", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--normalize-obs", dest="normalize_obs", action="store_true", default=None)
    parser.add_argument("--no-normalize-obs", dest="normalize_obs", action="store_false", default=None)
    return parser.parse_args(argv)


def build_config(args) -> Config:
    """只覆盖用户显式传入的字段，其余保持 dataclass 默认值。"""
    cfg = Config()
    for key, value in vars(args).items():
        if value is not None and hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


# ---------------------------------------------------------------------- 评估
def evaluate(agent, env, episodes, seed, max_steps=999):
    env.track = False
    env.seed(seed)

    returns, lengths, successes = [], [], []
    for _ in range(int(episodes)):
        obs = env.reset()
        total, steps, reached = 0.0, 0, False
        while steps < int(max_steps):
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

    return {
        "eval_return": float(np.mean(returns)),
        "eval_std": float(np.std(returns)),
        "eval_len": float(np.mean(lengths)),
        "eval_success": float(np.mean(successes)),
    }


# ---------------------------------------------------------------------- 存档
def save_train_state(agent, env, episode, path):
    payload = {
        "agent": agent.state_dict(),
        "episode": int(episode),
        "obs_normalizer": env.rms.state_dict() if env.rms is not None else None,
        "rng": rng_state(),
    }
    return save_checkpoint(payload, path)


def save_policy(agent, env, path):
    """自包含的推理包：actor 权重 + 配置 + 观测归一化统计量。"""
    payload = {
        "algo": agent.algo,
        "cfg": agent.cfg.to_dict(),
        "actor": agent.actor.state_dict(),
        "obs_normalizer": env.rms.state_dict() if env.rms is not None else None,
    }
    return save_checkpoint(payload, path)


def _safe_mean(values):
    finite = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


# ---------------------------------------------------------------------- 主循环
def train(cfg: Config) -> dict:
    set_seed(cfg.seed)
    if cfg.threads:
        torch.set_num_threads(int(cfg.threads))

    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(cfg.out_dir, cfg.run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(cfg.summary())
    print(f"device  = {device}")
    print(f"run_dir = {run_dir}\n")

    env = MountainCarEnv(cfg.env_id, seed=cfg.seed, normalize_obs=cfg.normalize_obs)
    eval_env = MountainCarEnv(cfg.env_id, seed=cfg.seed + 10_000, normalize_obs=cfg.normalize_obs)
    eval_env.track = False

    agent = build_agent(cfg, env, device=device)

    start_episode = 0
    if cfg.resume:
        checkpoint = load_checkpoint(cfg.resume, map_location=device)
        agent.load_state_dict(checkpoint["agent"])
        normalizer = checkpoint.get("obs_normalizer")
        if normalizer is not None and env.rms is not None:
            env.rms.load_state_dict(normalizer)
        start_episode = int(checkpoint.get("episode", 0))
        set_rng_state(checkpoint.get("rng"))
        print(f"[resume] 从 episode {start_episode} 继续")

    logger = CSVLogger(os.path.join(run_dir, "train.csv"), FIELDS, append=start_episode > 0)

    best_eval = -float("inf")
    success_streak = 0
    stopped_early = False
    episode = start_episode
    started = time.time()

    try:
        while episode < cfg.episodes:
            obs = env.reset()
            ep_return, ep_len = 0.0, 0
            critic_losses, actor_losses = [], []

            for _ in range(int(cfg.max_steps)):
                action = agent.select_action(obs, explore=True)
                next_obs, reward, terminated, truncated = env.step(action)

                # 奖励缩放只作用于写进 buffer 的值，评估仍用原始奖励
                agent.store(obs, action, reward * cfg.reward_scale, next_obs, terminated)
                agent.total_steps += 1

                obs = next_obs
                ep_return += reward
                ep_len += 1

                if agent.ready():
                    for _ in range(int(cfg.updates_per_step)):
                        losses = agent.update()
                        critic_losses.append(losses["critic_loss"])
                        actor_losses.append(losses["actor_loss"])

                if terminated or truncated:
                    break

            episode += 1
            row = {
                "episode": episode,
                "step": agent.total_steps,
                "ep_return": round(ep_return, 3),
                "ep_len": ep_len,
                "critic_loss": _safe_mean(critic_losses),
                "actor_loss": _safe_mean(actor_losses),
                "elapsed_s": round(time.time() - started, 2),
            }

            if episode == 1 or episode % max(int(cfg.eval_every), 1) == 0:
                if env.rms is not None:
                    eval_env.rms.load_state_dict(env.rms.state_dict())
                stats = evaluate(agent, eval_env, cfg.eval_episodes, cfg.seed + 10_000, cfg.max_steps)
                row.update(stats)

                marker = ""
                if stats["eval_return"] > best_eval:
                    best_eval = stats["eval_return"]
                    marker = "  *best"
                print(
                    f"[ep {episode:>4}] return={ep_return:8.2f} len={ep_len:>4} "
                    f"| eval={stats['eval_return']:7.2f}±{stats['eval_std']:5.2f} "
                    f"succ={stats['eval_success'] * 100:5.1f}% "
                    f"| {human_time(time.time() - started)}{marker}"
                )

                if (
                    cfg.early_stop
                    and stats["eval_return"] >= cfg.target_return
                    and stats["eval_success"] >= 1.0
                ):
                    success_streak += 1
                else:
                    success_streak = 0
            elif episode % 10 == 0:
                print(
                    f"[ep {episode:>4}] return={ep_return:8.2f} len={ep_len:>4} "
                    f"| {human_time(time.time() - started)}"
                )

            logger.log(row)

            if cfg.save_every and episode % int(cfg.save_every) == 0:
                save_train_state(agent, env, episode, os.path.join(run_dir, "last_state.pt"))

            if cfg.early_stop and success_streak >= max(int(cfg.patience), 1):
                print(f"[early-stop] 连续 {success_streak} 次评估达标，提前结束于 episode {episode}")
                stopped_early = True
                break

    except KeyboardInterrupt:
        print("\n[interrupt] 收到中断，保存进度后退出……")
        save_train_state(agent, env, episode, os.path.join(run_dir, "last_state.pt"))
        logger.close()
        raise
    finally:
        logger.close()

    state_path = save_train_state(agent, env, episode, os.path.join(run_dir, "last_state.pt"))
    policy_path = save_policy(agent, env, os.path.join(run_dir, "policy.pt"))

    if env.rms is not None:
        eval_env.rms.load_state_dict(env.rms.state_dict())
    final = evaluate(agent, eval_env, max(int(cfg.eval_episodes), 20), cfg.seed + 10_000, cfg.max_steps)

    summary = {
        "algo": cfg.algo,
        "seed": cfg.seed,
        "episodes": episode,
        "total_steps": agent.total_steps,
        "updates": agent.updates,
        "final_return": final["eval_return"],
        "final_std": final["eval_std"],
        "success_rate": final["eval_success"],
        "solved": bool(final["eval_return"] >= cfg.target_return and final["eval_success"] >= 1.0),
        "early_stopped": stopped_early,
        "elapsed_s": round(time.time() - started, 2),
        "policy": policy_path,
        "train_state": state_path,
    }
    save_checkpoint(summary, os.path.join(run_dir, "summary.pt"))

    print("\n" + "=" * 62)
    for key, value in summary.items():
        print(f"  {key:<14} {value}")
    print("=" * 62)

    env.close()
    eval_env.close()
    return summary


def main(argv=None):
    cfg = build_config(parse_args(argv))
    train(cfg)


if __name__ == "__main__":
    main()
