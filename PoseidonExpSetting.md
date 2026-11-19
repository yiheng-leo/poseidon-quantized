# Poseidon 实验设置：SBPQ 主导协议 v2

设置更新日期：2026-11-19（Asia/Shanghai）。实验进度、历史模型及服务器操作见 [PoseidonExperiments.md](PoseidonExperiments.md)。

本文记录本轮实验采用的配置、与旧结果的区别，以及仍需落实的实现建议。PPQ 固定使用 **layerwise**；用户最新指定其优化预算为 **每层 5 epochs**。

## 1. 三组训练与校准 seed 配对

只执行以下三组绑定组合，不进行训练 seed 与校准 seed 的笛卡尔积：

| 组别 | 训练 `seed` | `random_calibration` | 校准设置 | 实际 CLI | 文件名后缀 |
| --- | --- | --- | --- | --- | --- |
| T1 / Cdefault | 1 | n | 固定选取训练集前 16 条轨迹 | `--seed 1 --random_calibration n` | `default` |
| T7 / C7 | 7 | y | `calibration_seed=7` | `--seed 7 --random_calibration y --calibration_seed 7` | `seed7` |
| T42 / C42 | 42 | y | `calibration_seed=42` | `--seed 42 --random_calibration y --calibration_seed 42` | `seed42` |

`default` 是固定校准模式，不是整数 seed。该组的 `calibration_seed` 不参与轨迹抽样；命令模板将其保留为整数 42 占位，不能因此把它记成随机校准 seed42。

本轮不再使用校准 seed1/seed2/seed3，也不再固定所有运行的训练 seed 为 42。三组同时改变训练随机性与校准数据，因此组间差异反映两者的共同影响，不能单独归因于某一类随机性。固定 seed 也不保证不同硬件／软件环境下逐位相同；同时记录 checkpoint、源码提交、GPU、PyTorch／CUDA 版本及数值精度设置。

## 2. 公共实验协议

| 项目 | 本轮规定 |
| --- | --- |
| 数据集 | NS-PwC、CE-RM、Wave-Gauss |
| 方法 | Uniform、PPQ layerwise、BRECQ、SBPQ |
| 配置位宽 B | 2、3、4、8 |
| 原始模型 | 各数据集对应的同一 Poseidon checkpoint |
| 校准轨迹 | `traj_num=16`，使用所选轨迹的全部规定时间对 |
| 校准时间 | `final_time=10`、`time_back=0` |
| 校准加载 batch | `batch_size=2` |
| workers | `num_workers=0` |
| 训练／校准 seed | 严格使用第 1 节三组配对 |
| 激活精度 | `activation_bits=32`，weight-only |
| 原模型参数 | 优化时冻结；保留各方法的量化参数或舍入参数优化 |
| 评估 | 相同测试划分、评估终点、协议、指标和 batch 设置，并保存实际命令 |
| 位宽记录 | 文件名使用配置位宽；结果另记实际 `avg_weight_bits` |

2026-11-19 共享等待队列的正式评估已明确采用：三个数据集均为 `safe-svd`、test、`final_time=10`、`eval_obj=ongt`、`batch_size=8`、`physics_batch_size=10`、`num_workers=0`。量化完成后串接独立评估，显式保存 v2 结果路径、实际命令和评估协议；gamma 扫描候选不进行完整测试集评估。运行状态及入队范围见进度文档第 5.2 节。

通用 JSON 默认文件中的 `traj_num=64`、`final_time=20` 不代表本轮设置，启动模板会显式覆盖为 16／10，并显式传入 seed 配对。Uniform 不使用离线校准和优化随机性，同位宽三组可能产生相同结果，仍按指定组别登记，不自动伪造独立运行记录。

## 3. 以 SBPQ 为参照的优化配置

| 参数 | SBPQ 基准 | PPQ layerwise | BRECQ |
| --- | --- | --- | --- |
| 优化单位 | 全模型联合优化步长 | 每个 Linear 独立优化步长 | 逐块优化舍入参数 |
| 配置位宽 | `target_bits=B` | `target_bits=B` | `weight_bits=B` |
| 初始化 | `init_bits=8` | **`init_bits=8`**，旧值为 9 | 直接使用 B 位网格 |
| MC 采样数 | `mc_samples=10` | **`mc_samples=10`**，旧值为 5 | 不使用 MC |
| 噪声尺度 | `eta=1e-6` | `eta=1e-6` | 不适用 |
| 优化器 | AdamW | Adam | Adam |
| 学习率 | `learning_rate=3.75e-5` | 保留 `learning_rate=9.1e-4` | 保留 `reconstruction_lr=1e-3` |
| weight decay | 0 | 0 | 0 |
| 学习率调度 | 固定 | 四阶段倍率 `1 / 0.1 / 0.01 / 0.001` | 固定 |
| 优化预算 | 全模型 `total_steps=800` | **每层 `epochs=5`**，旧值为 21 | 每块 `reconstruction_iterations=20000` |
| 重构 batch | 使用校准 batch 2 | 使用校准 batch 2 | `rec_batchsize=32` |
| 优化目标 | MC 边缘似然＋块级 Beta prior | 当前实现的 MC 平均 MSE＋MDL prior | 块重构 MSE＋AdaRound 正则 |
| 保存间隔 | `save_every=100` | `save_every=1000` | `save_every=1000` |

PPQ 的 `configs/ppq/layerwise.json` 已写入 `init_bits=8`、`mc_samples=10`、`epochs=5`；启动模板还会显式传入这些值。其四阶段学习率调度会随总 epochs 从 21 缩短到 5；这是新协议的预算变更，不等同于取旧运行的前 5 个 epoch。

学习率、优化器、预算单位和 BRECQ 的重构 batch 保留各方法的语义。SBPQ 的全模型 800 次更新、PPQ 的每层 5 轮、BRECQ 的每块 20000 次更新不是相同计算预算；相同 MC 数量也不是相同 FLOPs 或相同运行时间。

BRECQ 的 `reconstruction_lr` 才是重构学习率，通用 `learning_rate` 不能替代它。BRECQ 不使用 MC、eta 或 Beta 参数；不得仅因通用参数解析器接受这些字段，就将其记为算法实际生效的配置。

以当前 NS-PwC 的 533 层、528 个校准 batch 估算，新 PPQ 为 `533 × 5 × 528 = 1,407,120` 次更新，对应带噪 MC 前向与似然反向各 14,071,200 次，不包含缓存采集、额外 FP 前向和 prior 反向。层数或批次数变化后必须按实际值重算；该计数不代表与 SBPQ 或 BRECQ 的计算量等价。

## 4. 方法专用参数

### SBPQ

| 参数 | 值／语义 |
| --- | --- |
| `beta_kappa` | 150 |
| `delta_bits` | 2 |
| `prior_scale` | 1 |
| `likelihood_scale` | 1 |
| `minimum_bits` / `maximum_bits` | 1 / 16 |
| `beta_epsilon` | `1e-6` |
| `beta_relative_epsilon` | `1e-12` |
| `task_loss_weight` / `sobolev_loss_weight` | 1 / 1 |
| Sobolev 范数 | L1 |
| 各阶 Sobolev 权重 | 1 |
| `sobolev_order` | NS-PwC：2；CE-RM：1；Wave-Gauss：1 |
| `percentile` | `1e-4`，经验分位数权重范围 |
| 梯度裁剪 | 不启用 |

Sobolev 损失用于准备阶段估计块敏感度，并据此构造 Beta prior；当前每步优化不是额外直接叠加一项 Sobolev loss。

### PPQ layerwise

- MDL prior：现有 `gamma=0.005` 作为明确记录的起点；不同实际平均位宽所采用的 gamma 需逐次记录。
- 权重范围：`percentile=1e-4`，当前为 Gaussian／std 范围估计；与 SBPQ 的经验分位数并非同一种估计。
- 当前 `bmax_bits=20`，步长约束对应有效位宽约 `[0,20]`。SBPQ 的 `minimum_bits`／`maximum_bits` 参数不会自动改变 PPQ 的约束；统一为 `[1,16]` 需要另行修改实现。
- 当前 layerwise 使用 MC 平均 MSE，其似然还除以层数；没有使用 SBPQ 的 Beta prior、Sobolev 敏感度或全模型 MC 边缘似然。
- `target_bits` 是本轮配置标签，MDL prior 不以它为中心。只改变 `target_bits`、而其余数值参数完全相同，不能当成已经获得不同实际位宽。调 gamma 的试验要独立保存，并按实际平均位宽与精度选定结果。
- `avg_cap_bits` 是 global PPQ 使用的参数；layerwise 不使用该 cap，不用它替代 gamma 调参。

### BRECQ

- MSE scales 初始化，随后使用 AdaRound 舍入优化。
- `reconstruction_iterations=20000`，`rec_batchsize=32`，`reconstruction_lr=1e-3`。
- 舍入正则权重 `0.01`；warmup 比例 `0.2`；温度 `20 → 2`。
- 使用非对称重构输入与 FP 目标，保留块级重构策略。

## 5. 尚需代码落实的严格对齐项

这些是建议修改与后续工作，**本次文档更新没有把它们自动实现为新的算法行为**。

| 项目 | 当前实际状态 | SBPQ 主导的建议 | 后续工作 |
| --- | --- | --- | --- |
| 量化层范围 | 已完成 NS-PwC：SBPQ 524 个结构块内 Linear；PPQ 533 个 Linear；BRECQ 541 个 Conv2d／Linear | 若做相同选层的受控比较，以 SBPQ 的结构块内 Linear 为统一清单 | 实现可验证的共享层清单／mask；重新导出模型，不能只改 JSON |
| 矩阵计算精度 | SBPQ 显式设置 `torch.set_float32_matmul_precision('high')`；另外两种未显式设置 | 共同入口明确采用同一精度策略，并记录实际 backend／GPU | 修改共同入口并验证缓存、恢复身份；当前没有可直接使用的 `--matmul_precision` 参数 |
| 学习位宽边界 | SBPQ `[1,16]`，PPQ 当前约 `[0,20]`，BRECQ 固定 B | 若要相同搜索范围，另行约束 PPQ 步长 | 需要实现与回归验证；固定 B 的 BRECQ 不参与连续位宽搜索 |
| 评估范围 | 历史 JSON 未保存完整评估参数 | 三种方法采用相同测试协议、同数据集的相同评估终点 | 先核定并记录；不能从校准文件名 ft10 推断评估终点 |

完成上述实现前，可以验证当前已支持的参数配置与启动模板，但不能把运行结果描述为“量化层范围、搜索边界和计算精度均已严格统一”。若这些实现发生变化，保留源码提交、配置与缓存指纹，另开协议版本。

## 6. 提速建议及本轮采用边界

| 建议 | 本轮处理 |
| --- | --- |
| PPQ 降低 checkpoint 频率 | 启动模板使用 `--save_every 1000`；保持每层完成时的保存 |
| 当前层缓存放 RAM／GPU，或使用本地 SSD／分块预取 | 待实现；保留缓存身份校验、样本顺序和全部校准数据 |
| 去掉 forward hook 中重复的 FP Linear 前向 | 待实现；逐层模式直接执行带噪 Linear，并回归梯度及目标值 |
| 减少 MC loss 标量读取和全模型 bits 统计的 GPU 同步 | 待实现；优化统计，不改变目标与采样预算 |
| MC 向量化／批量计算 | 待实现；保持 K=10、独立噪声和正确损失归约，实测显存与速度 |
| 按结构组采集层缓存 | 待实现；减少重复运行 FP 前缀，不丢校准样本 |
| 逐层提前停止 | 作为单独预算实验；当前正式配置仍训练满 5 epochs |
| 减少 MC、增大训练 batch、减少轨迹或时间对 | 不用于本轮已确定配置；分别保持 MC=10、batch=2、traj16、ft10 |
| 多 GPU 并行不同实验 | 可按空闲 GPU 分配不同方法／数据集／seed，登记主机、GPU 和 PID |

用户已明确将 PPQ 从 21 epochs 改为 5；不再使用之前举例的“7 epochs、MC=3”组合。提前停止不能只看日志中的全模型平均 bits，需要观察当前层的损失与步长变化。

## 7. 输出版本与旧结果隔离

新协议使用 `sbpq-led-v2` 子目录，保留此前确认的短文件名格式：

```text
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_artifacts/<数据集>/sbpq-led-v2/<方法>-traj16-ft10-bit<B>-<default|seed7|seed42>/
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_evaluation/<数据集>/sbpq-led-v2/<方法>-traj16-ft10-bit<B>-<default|seed7|seed42>.json
```

原因：旧 default 的训练 seed 是 42，新 default 的训练 seed 是 1；旧 PPQ 还使用 MC=5、init_bits=9、epochs=21。旧模型及其评估继续留在原路径，不改名冒充新实验，不覆盖，不计入 v2 主表的完成格。

当前程序省略路径时仍使用原来的非版本目录。**v2 模板必须显式指定 `--save_path` 或 `--output_root`，评估必须显式指定版本化 `--output_path`**；默认评估路径推断不会自动保留额外的 `sbpq-led-v2` 子目录。

调 gamma 等超参数时，在 v2 下另设 tuning 子目录并显式指定保存／评估路径；只有确定的正式运行才登记到主表。保存完整有效配置，不能通过覆盖同名目录混合不同 gamma 或不同协议。

## 8. 执行与记录

1. 按第 1 节设置绑定 seed 组，按进度表选择数据集、配置位宽及方法。
2. 读取实际生效配置，确认 PPQ 为 `init_bits=8`、`mc_samples=10`、`epochs=5`。历史 checkpoint 的原参数不随默认文件变化；不能把旧运行的完成状态当成新预算结果。
3. 依据第 5 节区分当前可执行参数与待实现的严格对齐项；不要写入尚不存在的 CLI 参数。
4. 按 PoseidonExperiments.md 的 tmux 与启动模板执行，记录主机、GPU、PID、日志、源码提交和参数。
5. 分别记录量化完成和评估完成；保存实际平均位宽，PPQ 同时记录 gamma、初始化、MC 与 epochs。
6. 本轮三组是联合随机设置；报告 mean/std 时注明这一含义，不宣称只测试了训练 seed 的方差。

实际启动、等待队列及完成状态以 [PoseidonExperiments.md](PoseidonExperiments.md) 和服务器记录为准；历史模型与结果保留原始配置及路径。
