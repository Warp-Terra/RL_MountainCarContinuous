#!/usr/bin/env python
"""基准测试：同样的批量下，CPU 和 CUDA 的梯度更新吞吐差多少。

MountainCarContinuous 的观测只有 2 维、网络是几百宽的小 MLP，
单次前向反向的计算量是微秒级，GPU 的 kernel 启动开销可能反而更大。
所以别猜，跑一下。

    python tools/bench_device.py
    python tools/bench_device.py --iters 2000 --threads 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from rl.algos import build_agent
from rl.config import Config
from rl.envs import MountainCarEnv


def bench(device, cfg, steps, iters):
    env = MountainCarEnv(cfg.env_id, seed=0)
    agent = build_agent(cfg, env, device=device)

    # 先把 buffer 填起来，保证 update 能跑
    obs = env.reset()
    for _ in range(steps):
        action = agent.select_action(obs, explore=True)
        next_obs, reward, terminated, truncated = env.step(action)
        agent.store(obs, action, reward * cfg.reward_scale, next_obs, terminated)
        obs = env.reset() if (terminated or truncated) else next_obs

    for _ in range(50):  # 预热，排除首次 kernel 编译 / 内存分配
        agent.update()

    is_cuda = str(device).startswith("cuda")
    if is_cuda:
        torch.cuda.synchronize()

    started = time.perf_counter()
    for _ in range(iters):
        agent.update()
    if is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    env.close()
    return elapsed


def main(argv=None):
    parser = argparse.ArgumentParser(description="比较 CPU / CUDA 的更新吞吐")
    parser.add_argument("--steps", type=int, default=3000, help="预热填充 buffer 的步数")
    parser.add_argument("--iters", type=int, default=1000, help="计时的更新次数")
    parser.add_argument("--algo", default="td3", choices=["ddpg", "td3"])
    parser.add_argument("--threads", type=int, default=1, help="CPU 线程数")
    args = parser.parse_args(argv)

    if args.threads:
        torch.set_num_threads(int(args.threads))

    cfg = Config(
        algo=args.algo,
        buffer_size=20000,
        batch_size=256,
        learning_starts=0,
        hidden=256,
        episodes=1,
    )

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    else:
        print("[warn] CUDA 不可用，只测 CPU")

    print(f"torch {torch.__version__} | algo={args.algo} | iters={args.iters} | threads={args.threads}")
    results = {}
    for device in devices:
        elapsed = bench(device, cfg, args.steps, args.iters)
        rate = args.iters / elapsed
        results[device] = rate
        print(
            f"  {device:<5} {elapsed:7.2f} s  ->  {rate:8.0f} updates/s  "
            f"({elapsed / args.iters * 1000.0:.3f} ms/update)"
        )

    if "cuda" in results and "cpu" in results:
        ratio = results["cuda"] / results["cpu"]
        if ratio >= 1.0:
            print(f"\n结论：GPU 快 {ratio:.2f}x")
        else:
            print(f"\n结论：GPU 慢 {1 / ratio:.2f}x（这个任务太小，CPU 反而更合适）")


if __name__ == "__main__":
    main()
