#!/usr/bin/env python
"""把 runs/*/train.csv 画成学习曲线。

    python tools/plot_results.py runs/td3_s0/train.csv runs/ddpg_s0/train.csv
    python tools/plot_results.py runs/*/train.csv -o results/learning_curve.png

只依赖 matplotlib（可选组件）。
"""

from __future__ import annotations

import argparse
import csv
import os


def load_eval_curve(path):
    """只取有 eval_return 的行，返回 (episodes, returns)。"""
    episodes, returns = [], []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get("eval_return") or "").strip()
            if not raw or raw.lower() == "nan":
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if value != value:  # NaN
                continue
            episodes.append(int(float(row["episode"])))
            returns.append(value)
    return episodes, returns


def main(argv=None):
    parser = argparse.ArgumentParser(description="绘制 MountainCarContinuous 学习曲线")
    parser.add_argument("csv", nargs="+", help="一个或多个 train.csv")
    parser.add_argument("-o", "--out", default=os.path.join("results", "learning_curve.png"))
    parser.add_argument("--target", type=float, default=90.0, help="solved 参考线")
    args = parser.parse_args(argv)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("需要 matplotlib：pip install matplotlib")

    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=140)

    plotted = 0
    for path in args.csv:
        episodes, returns = load_eval_curve(path)
        if not episodes:
            print(f"[skip] {path} 中没有 eval_return 记录")
            continue
        label = os.path.basename(os.path.dirname(os.path.abspath(path))) or os.path.basename(path)
        axis.plot(episodes, returns, marker="o", markersize=3, linewidth=1.6, label=label)
        plotted += 1

    if plotted == 0:
        raise SystemExit("没有可绘制的数据")

    axis.axhline(args.target, color="gray", linestyle="--", linewidth=1, label=f"solved ({args.target:.0f})")
    axis.set_xlabel("episode")
    axis.set_ylabel("eval mean return")
    axis.set_title("MountainCarContinuous")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    figure.savefig(args.out)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
