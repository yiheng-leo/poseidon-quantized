# Poseidon quantization：MySVD 校准协议

本分支把 Poseidon 的量化流程统一成三个阶段：加载同一份校准集、执行量化、保存模型。
评估由 `python -m evaluation.evaluate` 单独执行。默认生产入口不读取旧的 `/lus/...`
范围、步长、Fisher 或校准 batch 缓存，也不在训练期间构造验证/测试集。

校准策略移植自 `physic_model_comp/poseidon/MySVD/helper/calibration_dataset.py`
及 `MySVD/mysvd.py`。参考代码仅用于阅读和对照测试，运行本仓库不依赖 MySVD。
对应的 ScOT 数据集实现已经逐文件比较：时间索引和规范化逻辑一致。
SAPQ 的 `--origin` 是显式例外，用于对照 main 原始实验设置与旧缓存。

## 校准集的精确定义

- `--traj_num 64`：默认选择训练集前 64 条**完整轨迹**。
- `--random_calibration y --calibration_seed 42`：先加载完整训练划分，然后严格使用
  `random.Random(seed).sample(range(total_trajectories), traj_num)`；保留返回的轨迹顺序。
- 时变问题使用 `fix_input_to_time_step=None`、`time_step_size=1`、
  `max_num_time_steps=final_time-time_back`，保留所有 `t1 <= t2` 的时间对，包括零时间差。
- 椭圆问题和 MySVD 一样，不传上述时间参数。
- `shuffle=False`、`drop_last=False`，没有 `calib_steps`、随机样本截断或隐式子集。
  `final_time=20,time_back=0` 时，每条轨迹有 231 对，64 条共 14,784 个样本。
- `--batch_size` 只控制内存批大小；默认 2。它不减少校准集，且会记录在元数据中。
  算法优化的 batch 大小会影响数值，比较实验应同时固定这个参数。
- Wave-Gauss 的最大时间是 15，需显式 `--final_time 15`，不会静默裁剪成别的协议。

`metadata.json` 记录所选相对/绝对轨迹索引、全部时间对、规范化、实际样本数、
源码摘要、数据文件身份与内容探针、校准指纹以及全部有效参数。
文件内容探针用于防止误用缓存，并非对整个大型数据文件的完整密码学哈希。
每次量化使用独立临时目录，不能误读旧实验的无来源缓存。

## 运行

要求 Python >= 3.10 和适合当前设备的 PyTorch。

```bash
python -m pip install -r setting_environment/quantization_requirements.txt
```

模型路径默认沿 MySVD 的
`checkpoints/Poseidon-L/no_phyloss/<dataset>/origin/none`，数据路径默认
`poseidon/camlab-ethz/down_streams`。本仓库原来的模型/数据软链接指向集群，
请用参数提供实际可访问的路径。数据目录既可以是包含多个 `.nc` 文件的目录，也可以是
包含当前任务 `.nc` 文件的目录；底层文件名与 ScOT 一致。

量化并保存（以下路径需替换成实际路径）：

```bash
python -m SAPQ.sapq --variant global \
  --model_name Poseidon-L \
  --model_path /path/to/poseidon-checkpoint \
  --dataset fluids.incompressible.PiecewiseConstants \
  --data_path /path/to/down_streams \
  --traj_num 64 --random_calibration n --calibration_seed 42 \
  --final_time 20 --time_back 0 --batch_size 2 \
  --init_bits 9 --target_bits 8 --sensitivity sobolev \
  --sensitivity_normalization raw --DEV cuda \
  --save_path quantization_artifacts/NS-PwC/sapq
```

独立评估保存的模型（默认对齐 SAFE-SVD-v0；不重新量化）：

```bash
python -m evaluation.evaluate \
  --model_path quantization_artifacts/NS-PwC/sapq \
  --checkpoint /path/to/poseidon-checkpoint \
  --dataset fluids.incompressible.PiecewiseConstants \
  --data_path /path/to/down_streams \
  --final_time 20 --batch_size 256 --physics_batch_size 10 --DEV cuda \
  --output_path quantization_evaluation/NS-PwC/sapq.json
```

默认使用 `--protocol safe-svd`：完整 test split 的全部 `t1 <= t2` 时间对，
每对独立前向；沿用 ScOT 的 labels/pixel_mask 约定及模型 config 定义的 base_loss。
`--num_trajectories` 默认 64，按 SAFE-SVD 传给数据集构造器，**不截断 test split**。
物理指标按大小为 10 的批次等权平均，末尾不足一批也占一份权重；比较时必须固定此值。
`--checkpoint` 可选，提供时先评估原模型，并输出压缩模型相对原模型的百分比变化。
省略时只评估保存的模型；`--eval_obj oncomp` 则必须提供原模型，以其输出为 fidelity 标签。
Wave-Gaussians 使用 `--final_time 15`。指标及方法差异见 [评估对齐说明](../docs/poseidon_evaluation.md)。

原固定终点评估仍可通过 `--protocol endpoint --initial_time 0 --final_time 15 --ar_steps 3`
显式运行；它按样本加权、从输入恢复掩码通道，且不向模型传递标签。
在该模式下，`--num_trajectories` 才表示 held-out 轨迹前缀。

## main 原实验：--origin

在 ritchie 上运行 main/db1d702 的 `SAPQ/run_sapq_network_global.py` 默认实验设置：

```bash
python -u -m SAPQ.sapq --origin \
  --model_path models/poseidon/fluids.incompressible.PiecewiseConstants \
  --dataset fluids.incompressible.PiecewiseConstants \
  --data_path poseidon/camlab-ethz/down_streams \
  --DEV cuda --seed 0 \
  --save_path quantization_artifacts/NS-PwC/sapq-origin
```

`--origin` 选择整个原实验配置，会覆盖 `--config` 和 CLI 中的实验参数，包括
`--variant`、`--traj_num`、`--random_calibration`、`--final_time`、batch size、位宽、epoch、
学习率、先验及敏感度设置；被替换的值会打印在日志里。模型/数据/输出路径、dataset、设备、
seed、workers、overwrite/keep_work_dir 等运行参数继续生效。不加 `--origin` 则维持原行为。

固定设置为：global；batch size=2；init=9、target=8、上限=20 bit；MC=5；最多 21 epoch；
Adam lr=9.1e-4，沿 main 四段衰减；eta=1e-6；prior_scale=1e-4、sigma0=0.5、alpha=1；
Sobolev 敏感度按块 min-max 归一化；至少完成 10 epoch 后，平均 bit < 4 时早停。
使用 main 的原 MC 似然估计、labels/pixel_mask 前向约定、带加性 epsilon 的平均 bit 统计，
以及 `torch.set_float32_matmul_precision("high")`。没有改用论文第 3.4 节的 Beta prior，
也不是 main 的 `_raw` / `_all` 启动配置。

MC 实现使用两遍前向：先不保存反向计算图，记录各样本的似然权重和原始 RNG 状态；
再逐个恢复 RNG 状态、重算并反传，每次只保留一个样本的计算图。保持原版连续噪声抽样、
MC=5 和 log-mean-exp 梯度，避免原 stacked MC 同时保留五份图造成的显存峰值。
批次结束后恢复原随机序列的后续状态，后续优化更新仍使用同样的随机数。
代价是多一遍 MC 前向；浮点累加顺序不同仍可能带来微小数值误差。

旧文件默认从本仓库根目录寻找。`--origin_root /path/to/original-main-checkout` 可指定原仓库，
或用下列参数直接指定可信的原文件：

| 参数 | 默认寻找的位置 |
| --- | --- |
| `--origin_frozen_batches` | `ppq_artifacts/frozen_calibration_batches.pt` |
| `--origin_sensitivity` | `SAPQ/prior_sensitivity_sobo/<model>/prior_sensitivity.pt` 的 `layer_sensitivity_norm` |
| `--origin_ranges` | `precalculated_ranges/<model>/p1e-04/ranges.pt` |
| `--origin_initial_steps` | `initial_step_sizes/<model>/p1e-04/9bit.pt` |
| `--origin_layers` | `inspect_layers/L_quantize_layers.pt` 或 `.json` |

`<model>` 优先用 checkpoint 目录名。NS-PwC 的 Poseidon-L 另兼容原 `NS-PwC-L` 名称。
指定文件不存在会报错。已有 frozen batches 全部按原顺序使用，不按 512 批截断；
main 的原启动器也是如此，实际数量由缓存决定。

默认位置没有缓存时，会明确打印重建提示：校准加载 `num_trajectories=2048` 的训练数据，
不传时间覆盖参数，保留 ScOT 原默认时间设置（普通时变任务为步长 2、0–14），
从 shuffle=True 的 loader 取前 512 批、通常共 1,024 个样本。随后重算缺失的范围和敏感度；
步长没有旧缓存则按 range/2^9 初始化。它不是取 16/64 条完整轨迹。

metadata.json 记录有效配置、实际校准数量、旧文件路径及 SHA-256、实际完成 epoch 和早停状态。
缺少原校准/敏感度缓存时，结果属于相同规则下的新实验，不能保证复现原来的随机样本和数值。
`--seed` 控制当前运行的随机数；历史运行若没有记录 seed，仍无法由默认设置恢复其 RNG 状态。

模型仍保存为可独立评估的 `compressed_model.pt`，例如：

```bash
python -m evaluation.evaluate \
  --model_path quantization_artifacts/NS-PwC/sapq-origin/compressed_model.pt \
  --checkpoint models/poseidon/fluids.incompressible.PiecewiseConstants \
  --dataset fluids.incompressible.PiecewiseConstants \
  --data_path poseidon/camlab-ethz/down_streams --DEV cuda \
  --final_time 20 --batch_size 256 --physics_batch_size 10 \
  --output_path quantization_evaluation/NS-PwC/sapq-origin.json
```

`--origin` 只切换量化实验；此处评估仍用已对齐的 SAFE-SVD 协议，便于与现有结果比较。

## 方法和范围

| `--method` | 量化逻辑 |
|---|---|
| `fp` | 保存原浮点模型，供相同独立评估协议使用 |
| `uniform` | 原 PPQ 对照的逐输出通道 max-abs 权重网格，`2*maxabs/(2^bits-1)` |
| `dynamic` | DynamicQ 对称权重量化及可选运行时输入激活量化 |
| `static` | 用完整校准集估计输入激活范围的静态 QDQ |
| `ppq` | 全网 MC 平均 MSE + 参数量加权平均位宽上限，保持当前 PPQ trainer 的实际目标 |
| `ppq-layerwise` | 逐层 MC 平均 MSE + MDL 先验 |
| `sapq` | 全网联合步长优化，MC log-mean-exp 似然 + 可选位宽先验 |
| `sapq-block` | 逐 ScOT/残差块的 Fisher 加权重构 |
| `sapq-layerwise` | 逐层重构和可选位宽先验 |
| `sapq-network` | 逐块调度，但在最终网络输出计算似然；保留该消融的 sum 归约和先验缩放 |
| `brecq` | Poseidon 块包装 + AdaRound 重构，保存后不依赖包装器/原模型 |
| `repq` | 所选 Linear 的 scale/shift 重参数化；ScOT 条件化归一化用显式输入仿射保持等价 |
| `ptq4vit` | 所选 Linear 的 Hessian/Fisher 区间搜索，GELU 后采用 twin uniform 激活量化 |

PPQ/SAPQ 当前只支持权重量化，`--activation_bits` 必须为 32；步长学习与模型权重更新分开。
统一的 Linear 选择规则沿原 `inspect_layers`：排除名字中含 `norm`/`time` 的条件化层。
BRECQ 按自己的块定义处理卷积和 Linear，并在结果元数据中记录实际范围。

`repq`、`ptq4vit` 是明确限定为所选 Linear 的 Poseidon 适配，**没有量化 attention
matmul/softmax**，不能当作原 ImageNet 实现的全算子复现。`static` 是可保存的静态 QDQ
基线；历史 `StaticQ/eager_static.py` 的 QNNPACK 探索并不等同于这个入口。

这些模型保存的是浮点量化/反量化模拟（QDQ），不是打包的整数模型，不能由文件大小或
运行时间推断实际低位整数硬件加速。PPQ/SAPQ 的有效位宽是连续代理
`log2(range/step)`，元数据记录实际参数量加权结果，图例/目录名不再充当位宽证据。

SAPQ 默认保留当前 raw 敏感度协议和 `sigma=sigma0*(1+alpha*sensitivity)`。
`--sensitivity_normalization normalized` 是显式切换。不同粒度保留各自原先的
loss reduction 和先验缩放，不把它们伪装成仅改变训练粒度的完全等价实验。
Sobolev 敏感度保留历史实现的 dataset `std` 广播约定，并写入 metadata；本次不静默改动
历史方法的物理损失定义。评估侧则按物理通道恢复单位。

## 量化进度

所有正式入口自动显示 tqdm 进度，无需额外参数。阶段包括校准数据／FP 输出缓存、
激活范围、权重范围、SAPQ 敏感度、PPQ/SAPQ 的层或块、epoch、batch 和 MC 采样，
以及 BRECQ 的范围搜索／AdaRound 重构、PTQ4ViT 的梯度缓存／区间搜索和各方法权重导出。
PPQ/SAPQ 的 batch 进度显示 likelihood、prior、总损失和学习率，每个 epoch 保留平均损失及位宽摘要。
保存阶段按文件计数，模型 SHA-256 校验按字节计数；没有已知长度的迭代器显示完成数量和速率。

进度条写入 stderr，摘要和原有日志写入 stdout。支持终端和重定向的 Slurm 日志，刷新默认限制为每秒一次；
短暂的 MC／候选搜索内层循环延迟显示。需要关闭进度条时设置环境变量，例如：

```bash
TQDM_DISABLE=1 python -u -m SAPQ.sapq --config configs/sapq/default.json --model_path ... --data_path ...
```

## 保存格式与评估口径

每个输出目录包含：

- `compressed_model.pt`：完整、可独立加载的 CPU 模型，与 MySVD 保存形式一致。
- `quantization_state.pt`：算法对应的步长、范围、敏感度、历史或其他量化状态。
- `metadata.json`：方法、校准选择、全部参数、代码版本、模型校验和与量化统计。

已有模型默认不覆盖，另选 `--save_path` 或明确加 `--overwrite`。
完整模型文件使用 Python/PyTorch 序列化，只加载可信来源的文件。
成功后清理本次临时缓存；`--keep_work_dir` 可保留，失败时保留用于诊断。

默认评估使用 SAFE-SVD 的 `PDE_calc.py`：物理量反归一化后计算 0/1/2 阶 Sobolev、
连续性、动量、涡量、H1，以及可压缩流/波动的时间项；按方程类型分派。
网格间距固定 `dx=dy=1/128`，物理批次等权平均。
`base_loss` 为 ScOT Trainer 返回的模型损失，不能当作 `RelL1` 百分数。
结果 JSON 仅包含 `avg_weight_bits` 和 `metrics`。每项指标保留 `origin`、`compressed`
及 `change_percent`（相对原模型的百分比变化）；`rel_*` 指标本身仍为无量纲比值。
不输出模型路径、参数、校准元数据、时间对、指纹、说明或模型大小等辅助信息。
平均 bit 优先读取 SBPQ/PPQ/SAPQ 的最终参数量加权 `avg_weight_bits`，固定位宽方法读取
`weight_bits`；缺少实际统计时为 `null`，不使用 target_bits 冒充最终结果。
原模型指标为 0 时 `change_percent` 为 `null`；未提供 checkpoint 时只保留 compressed 数值。
仅 `endpoint` 模式保留旧 `L1`/`RelL1`（RelL1 为百分数）和一阶物理指标。
两种协议均使用精简输出；旧终点结果需要重新评估后才能与 SAFE-SVD 对比。

## SBPQ 与入口迁移

SBPQ 使用 `python -m SBPQ.sbpq --config configs/sbpq/default.json`，默认模型目录为
`models/poseidon/<dataset>`。它采用上述完整轨迹校准协议，保留本次 NSPwC 复现实验的优化参数。
默认 800 个累计更新沿 batch 顺序循环，**不保证对大型校准集完成一遍 likelihood 遍历**；
敏感度仍使用全部样本。日志与 metadata 明确记录覆盖量，详情见
[SBPQ 迁移说明](../docs/ritchie_migration.md)。

`--config configs/<method>/default.json` 加载配置，显式 CLI 参数覆盖文件。
公共入口 quantization.quantize、quantization.evaluate 保留转发；旧模型类路径由 compat 映射。
原 SAPQ/PPQ/BRECQ/DynamicQ/StaticQ 文件移入 `archive/main/`，正式代码不导入这些历史实现。
本仓库原有 RepQ-ViT、VICON、Aurora、finetuning、绘图和下载材料保留原位置；本次不迁入旧仓库的 shell 调度矩阵。
目录见 [layout](../docs/layout.md)，正式命令见 [仓库说明](../README.md)。

## 本次目录迁移的边界

Baseline 实现取自旧仓库 ritchie，保持其数学定义和训练调度。SBPQ 另作校准与产物适配。
这并不将此前 Poseidon 适配升级为上游算法的完全复现：StaticQ 仍为 QDQ；
RepQ/PTQ4ViT 的算子范围、PTQ 回归梯度来源及此前概率方法调度选择均未在目录整理中改动。
已保留第三方来源和历史 trainer，供单独审查这些算法差异。

## 验证

```bash
python -m pytest -q
```

测试包括 MySVD helper 的直接对照、完整时间对和随机采样、数学/梯度检查、全部方法在真实
小型 ScOT 上的保存/回载、全新进程独立评估、mask/AR 防止标签泄漏。
合成小模型的测试证明接口与计算链路可运行，不证明真实 Poseidon-L 数据集的精度或速度。
