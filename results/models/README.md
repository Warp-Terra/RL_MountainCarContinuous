# 预训练模型

仓库里带一个训练好的策略，方便 clone 之后直接观察效果，不必先花时间训练。

## `td3_s0_policy.pt`

| 项 | 值 |
|---|---|
| 算法 | TD3 |
| 训练种子 | 0 |
| 收敛 episode | 40 |
| 评估表现 | **94.84 ± 1.30** |
| 评估协议 | 10 个 episode，确定性策略，固定种子 |
| 平均登顶步数 | 87 |
| 成功率 | 100% |
| 文件大小 | 271 KB |

### 包含内容

- Actor 网络权重
- 完整训练配置（`Config` 的序列化形式）
- 观测归一化统计量（`RunningMeanStd` 的 mean / var / count）

### 用法

```bash
# 纯评估，打印每个 episode 的回报与步数
python eval.py --ckpt results/models/td3_s0_policy.pt --episodes 10

# 开窗口实时观看（--fps 10 约等于 7 秒一个 episode）
python eval.py --ckpt results/models/td3_s0_policy.pt --episodes 5 --render --fps 10
```

> ⚠️ **观测归一化统计量和权重是绑定的。** 不要只替换 Actor 权重而沿用别的归一化参数，
> 否则输入尺度对不上，策略会直接失效。评估时请始终通过 `eval.py` 调用，
> 它会把 `obs_normalizer` 一并恢复。

### 重新生成

```bash
python train.py --algo td3 --seed 0 --episodes 200 --device cuda --tag td3_s0
```

训练产物会写到 `runs/td3_s0/`，其中 `policy.pt` 与这里提交的是同一份内容。
