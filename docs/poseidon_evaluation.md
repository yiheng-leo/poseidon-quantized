# Poseidon：方法比较与 SAFE-SVD 评估对齐

此处的比较基于两个本地仓库的实现；并未运行真实大模型精度实验。

## 压缩方法的区别

| 项目 | Structural-Aware Probabilistic Quantization | SAFE-SVD-v0 |
| --- | --- | --- |
| 主要压缩变量 | PPQ/SAPQ 优化量化步长和有效位宽；另提供多种量化基线 | 用两个低秩 Linear 替换原 Linear，rank 决定参数量 |
| 结构信息 | SAPQ 用 Sobolev/散度/MSE 敏感度调节概率模型，部分变体使用 Fisher 重构 | 用输出敏感度/Fisher 和输入统计变换权重，再执行截断 SVD；combined 模式采集 teacher/student 输入 |
| Fidelity | 对照 FP 输出的概率似然或重构损失；不同变体的归约不同 | 通过 teacher/student 输入的 Gram 与交叉统计补偿逐层累积误差，alpha 控制混合 |
| 校准数据 | 默认训练集前 64 条完整轨迹；支持固定 seed 的随机轨迹 | Poseidon helper 同样保留完整轨迹；实际轨迹数、seed、final_time 取决于运行参数 |
| 保存与运行 | 当前产物为浮点 QDQ，不能据文件大小推断整数压缩率或整数加速 | 低秩因子减少参数；实际速度仍受 rank、层形状和设备影响 |

对齐 evaluation 后可以比较误差，但位宽和 rank_ratio 并非等价预算。
公平比较仍需固定 FP checkpoint、数据版本、评估参数，以及实际参数/存储预算。
本次不修改压缩算法或校准损失。

## 原评估与 SAFE-SVD 的差异

| 项目 | 原 `endpoint` | 默认 `safe-svd` |
| --- | --- | --- |
| 测试选择 | 每条轨迹一个固定起终点 | 完整 test split，全 `0 <= t1 <= t2 <= final_time` 时间对，包含零时间差 |
| 前向 | 可多步自回归，终点计分 | 每个时间对一次直接预测，ar_steps=1 |
| 掩码 | 不传标签，固定通道从输入恢复 | 沿 ScOT Trainer 的 labels/pixel_mask 约定，用标签填固定通道 |
| 主损失 | normalized L1 总和、RelL1 百分数 | Trainer `_loss`，遵循保存模型 config 中的 p/channel_slice_list_normalized_loss 等设置 |
| 空间物理指标 | 一阶 Sobolev、内部网格散度/涡量 | SAFE-SVD 原公式：累计 0/1/2 阶 Sobolev、H1、方程残差；包含边界差分 |
| 时间指标 | 无 | 可压缩流和波动额外跑 PrevTimeDataset，用前一目标时刻预测计算时间导数 |
| 物理聚合 | 按样本加权 | 每个物理批次等权平均，含不足一批的末批 |
| 输出 | 平均 bit 与压缩模型指标 | 平均 bit、原模型/压缩模型指标及变化百分比 |

默认参数依据 SAFE-SVD 的 **Python 实现**：final_time=20、num_trajectories=64、
batch_size=256、physics_batch_size=10、eval_obj=ongt。其 README 写的 rollout
batch_size=1024 与实现不一致，这里以实现的 256 为准。

ScOT 的 num_trajectories 表示训练容量，test split 的轨迹数来自数据集 N_test。
因此 `--num_trajectories 64` 不表示只测 64 条测试轨迹。final_time=20 时，
每条时变轨迹有 231 个时间对，总评估样本数为 N_test × 231。

物理公式保留 SAFE-SVD 的固定参数：dx=dy=1/128、不可压缩 Re=2500、
可压缩 dt=1/20、波动 dt=1/15；压力偏置与 transpose 也按参考实现处理。
`base_loss` 不能替换为旧 `RelL1`。物理 `rel_*` 是比值，change_percent 是百分比，保留六位小数；
原模型指标为 0 时 change_percent 为 null。model_size/linear_size/rel_base_loss 不写入结果。
不可压缩流不计算时间项，因此其 sobolev_time=0 与参考实现一致。

## 运行

```bash
python -m evaluation.evaluate \
  --model_path quantization_artifacts/NS-PwC/sapq \
  --checkpoint /path/to/original-finetuned-checkpoint \
  --dataset fluids.incompressible.PiecewiseConstants \
  --data_path /path/to/down_streams --DEV cuda \
  --final_time 20 --num_trajectories 64 \
  --batch_size 256 --physics_batch_size 10 \
  --output_path quantization_evaluation/NS-PwC/sapq.json
```

`--model_path` 接受 artifact 目录或完整模型文件；沿用现有兼容加载器和校验和检查。
不依赖 SAFE-SVD checkout，不重新量化。省略 checkpoint 时只输出 compressed，
不会生成 origin 或 change_percent。也可以只提供 checkpoint 评估原模型。
`--eval_obj oncomp` 需要 checkpoint，先用原模型预测替换标签，再以同一标签评估两模型。

为方便复用 SAFE-SVD 参数，支持 `--model`、`--data-root`、`--device`、`--final-time`、
`--num-trajectories`、`--batch-size`、`--physics-batch-size`、`--eval-obj` 和 `--output` 别名。
本仓库使用自带 scOT，无需 --poseidon-root。Wave-Gaussians 必须设置 final_time=15；
静态问题推荐使用 Poseidon 的 `.time` 数据集名，与参考入口相同。

需要原终点协议时显式指定：

```bash
python -m evaluation.evaluate --protocol endpoint \
  --model_path /path/to/artifact --dataset fluids.incompressible.PiecewiseConstants \
  --data_path /path/to/down_streams --DEV cuda \
  --initial_time 0 --final_time 15 --ar_steps 3 --batch_size 16
```

旧接口 build_evaluation_loader/evaluate_model 仍为 endpoint helper，以兼容既有 Python 调用；
CLI 默认流程使用 evaluation/poseidon/evaluate.py。两种协议都只输出精简结果。

## 精简结果格式

输出文件和 stdout 的 JSON 都仅保留最终平均 bit 与各项指标，格式示例如下（数值为演示）：

```json
{
  "avg_weight_bits": 7.61,
  "metrics": {
    "base_loss": {
      "origin": 0.1,
      "compressed": 0.12,
      "change_percent": 20.0
    }
  }
}
```

change_percent = 100 × (compressed - origin) / origin；正数表示误差增加。
原指标为零时百分比无定义，写为 null。未提供 checkpoint 时每项仅保留 compressed。
指标真实数值不额外舍入；百分比保留六位小数。

avg_weight_bits 是量化权重的最终参数量加权平均位宽，不把未量化层混入该统计。
PPQ/SAPQ 读取量化结束时的 avg_weight_bits，固定位宽基线读取 weight_bits；
不使用目标位宽或初始位宽。产物没有相应统计时为 null；仅评估 FP checkpoint 时为 32。
不输出 artifact 元数据、模型路径、配置、时间对、样本数、数据指纹、协议说明或大小统计。
这些变更仅精简报告，不改变评估计算。已有输出文件需用同一命令重跑后更新。

## 来源与验证

`evaluation/poseidon/PDE_calc.py` 原样移植自 SAFE-SVD-v0/evaluation/poseidon/PDE_calc.py。
physics.py 保留同一公式选择和聚合过程，仅将导入改成本地包形式；reporting.py 负责精简输出。
ScOT Trainer 使用已加载模型，无需为了评估一个 artifact 再加载完整 FP checkpoint。
不影响 base_loss 的通道统计 callback、W&B 上报与外部 Poseidon 路径未引入。

测试包含四类 PDE 与本地 SAFE-SVD 参考代码逐指标直接比较、真实 ScOT 时间索引、
非整批聚合、原模型与保存模型的完整 ongt/oncomp 推理，以及全部量化方法的加载后物理评估。
可用 SAFE_SVD_REFERENCE_ROOT 指定参考仓库；缺少该仓库时仅跳过直接参考对照。
合成数据验证接口和数值口径，不代表真实 Poseidon-L 的最终精度。
