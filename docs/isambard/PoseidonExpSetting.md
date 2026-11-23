# Poseidon 实验设置 · Isambard

2026-11-23。适用于 Isambard 上的 NS-PwC 实验。作业 ID、状态及结果路径见 [PoseidonExperiments.md](PoseidonExperiments.md)。

## 服务器与公共配置

从本机 `ssh canada`，在 canada 上 `cd MyHome` 后执行 `ssh u6abp.aip2.isambard`，再进入 `cjhome/poseidon-quantized`。在项目根目录用 `sbatch launch_scripts/poseidon/<脚本>.sh` 提交。作业使用已有 conda `quant` 环境，申请 1 节点、1 GPU，输出保存在 `logs/`。

| 项目 | 设置 |
| --- | --- |
| 数据集 | `fluids.incompressible.PiecewiseConstants`（NS-PwC） |
| 原始模型 | `models/poseidon/fluids.incompressible.PiecewiseConstants`（Poseidon-L） |
| 数据目录 | `poseidon/camlab-ethz/down_streams` |
| 校准 | 固定前 16 条训练轨迹，`random_calibration=n`，`calibration_seed=42`（固定模式占位） |
| 校准时间 | `final_time=10`、`time_back=0`，共 1,056 个样本 |
| 激活 | 32 bits |
| 训练源码基准 | `a269466b2d46036aca02cfae416a6819c9b63ad0` 上的 PPQ/SBPQ 实现 |

## PPQ layerwise gamma sweep

沿用已完成 Isambard 候选 `results/NS-PwC/ppq-traj16-ft10` 的 seed 1、校准 batch 32、init8、MC10、每层 5 epochs、学习率 `9.1e-4`、`eta=1e-6`、`percentile=1e-4`、每 1,000 步存 checkpoint。该候选 `gamma=1e-6` 的实际平均位宽为 5.889101705 bits，纳入响应曲线。33 个校准 batch，每个候选满预算为 87,945 次更新。

粗扫 gamma 共 24 个，范围为 `1e-9` 到 `1e-2`。每个候选改变 gamma，运行参数中的 `target_bits=8` 沿用原配置标签，最终位宽从 `metadata.json` 的 `quantization.avg_weight_bits` 读取。四个目标为 2、3、4、8 bits，接受区间是 `B-0.1 ≤ avg_weight_bits ≤ B`。补点数组任务各自读取已完成点，按对数区间插值或向外扩展寻找目标，每个目标最多增加 4 个点。返回 `null` 表示尚无合格点。

结果根目录为 `results/NS-PwC/ppq-layerwise-traj16-ft10-b32-sweep/`，每个 gamma 使用独立目录。 `sweep-summary.json` 汇总已完成的位宽及各目标选中的模型。旧 UCL batch 2 的 sweep 配置另行保留。

## SBPQ batch 64

与 Isambard `sbpq-traj16-ft10-steps200` 配对，使用 seed 42，init8、target8、MC10、学习率 `3.75e-5`、Sobolev order 2。batch 设为 64，固定校准顺序产生 16 个满 batch 和 1 个 32 样本尾 batch。103 步覆盖六轮完整校准加一个 64 样本 batch，合计 6,400 个优化样本，与 batch 32 × 200 步相同。似然权重从 `0.0625` 调为 `0.03125`，按满 batch 大小成反比。变动后的批次划分仍会影响敏感度和优化轨迹。输出为 `results/NS-PwC/sbpq-traj16-ft10-b64-steps103/`。

## 批量评估

评估固定为 SAFE-SVD、test、`ongt`、`final_time=20`、推理 batch 256、物理指标 batch 10、workers 0。使用同一个原始 Poseidon-L checkpoint，对照 PPQ、BRECQ、SBPQ 的已有模型、新 SBPQ batch 64 及命中位宽区间的 sweep 模型。

`python -m evaluation.evaluate --models <模型目录1> <模型目录2> ...` 启用批量模式，模型目录也可替换为独立的模型文件。原有 `--model_path` 单模型模式继续输出原格式。批量结果默认写到 `results/NS-PwC/result.json`，在 `protocols` 下按数据、评估参数、原模型和运行环境区分配置。每个模型结束后立即保存，同一配置下已有的原始模型指标与压缩模型结果可复用。旧的单模型 `test.json` / `result.json` 缺完整评估参数，将按统一配置重新评估一次。

报告正式对比时，使用最终实际平均位宽、质量指标、全量化耗时和显存分配峰值，并记录各方法的量化层范围。上述 PPQ sweep 固定 seed 1，SBPQ batch 对比固定 seed 42。
