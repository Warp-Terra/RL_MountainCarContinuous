#!/usr/bin/env python
"""把 runs/*/summary.pt 汇总成一张对比表。

    python tools/collect_results.py
    python tools/collect_results.py --min-episodes 10
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from rl.utils import load_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description="汇总所有训练结果")
    parser.add_argument("--pattern", default=os.path.join("runs", "*", "summary.pt"))
    parser.add_argument("--min-episodes", type=int, default=10, help="过滤掉测试用的极短运行")
    args = parser.parse_args(argv)

    rows = []
    for path in sorted(glob.glob(args.pattern)):
        summary = load_checkpoint(path, map_location="cpu")
        if int(summary.get("episodes", 0)) < args.min_episodes:
            continue
        summary["_run"] = os.path.basename(os.path.dirname(path))
        rows.append(summary)

    if not rows:
        print("没有找到结果，先跑一次 train.py")
        return

    header = (
        f"{'run':<10} {'algo':<5} {'eps':>4} {'steps':>7} "
        f"{'return':>8} {'±std':>6} {'succ':>6} {'solved':>7} {'time(s)':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['_run']:<10} {row['algo']:<5} {int(row['episodes']):>4} {int(row['total_steps']):>7} "
            f"{row['final_return']:>8.2f} {row['final_std']:>6.2f} {row['success_rate'] * 100:>5.0f}% "
            f"{'yes' if row['solved'] else 'no':>7} {row['elapsed_s']:>8.1f}"
        )

    print()
    for algo in sorted({row["algo"] for row in rows}):
        group = [row for row in rows if row["algo"] == algo]
        episodes = np.array([row["episodes"] for row in group], dtype=float)
        returns = np.array([row["final_return"] for row in group], dtype=float)
        solved = sum(1 for row in group if row["solved"])
        print(
            f"{algo:<5} n={len(group)}  solved={solved}/{len(group)}  "
            f"收敛 episode={episodes.mean():.0f}±{episodes.std():.0f} (中位数 {np.median(episodes):.0f})  "
            f"最终 return={returns.mean():.2f}±{returns.std():.2f}"
        )


if __name__ == "__main__":
    main()
