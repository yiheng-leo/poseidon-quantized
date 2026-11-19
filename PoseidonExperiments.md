# Poseidon 实验操作与进度

历史状态核查：2026-11-19 11:14（Asia/Shanghai；跨主机扫描时间及覆盖范围见“正在运行的任务”）。

实验设置更新：2026-11-19。完整配置及待实施的对齐、提速建议见 [PoseidonExpSetting.md](PoseidonExpSetting.md)；新协议标记为 `sbpq-led-v2`。本次修改设置与文档，没有启动新实验，历史进程快照不作为新的实时检查。

本文供后续 agents 查看实验进度、连接服务器、启动量化及补做评估。主实验表的 **PPQ 已由用户确认统一采用 layerwise**。所有状态以服务器文件和实际进程为依据，启动或结束实验后应更新本文。

## 1. 检查空余 GPU、连接服务器、创建 tmux

在本地终端检查 UCL GPU 服务器：

```bash
check_ucl_gpus
```

当前脚本会筛选显存占用低于 1000 MiB、且通过 CUDA 小算子检查的 GPU。它的 CUDA 探测使用 `aimet_env`；正式实验仍需进入 `poseidon-quant` 环境。显示 `FREE` 是检查当时的状态，连接后用 `nvidia-smi` 再确认 GPU 编号和占用。

例如选择 ruddy；其他主机用检查结果中的别名替换：

```bash
ssh ruddy
# 其他已使用过的主机：ssh canada / ssh dory
```

**先创建 tmux，再进入 quant 和激活环境。** 以下让新 tmux 使用 Bash，后面的变量与数组命令可直接使用：

```bash
tmux new-session -s poseidon-nspwc-sbpq-bit8-train1-default /bin/bash
cd quant
source /cs/research/vecg/EasyPDE/miniconda3/etc/profile.d/conda.sh
conda activate poseidon-quant
pwd
nvidia-smi
```

`cd quant` 解析后的服务器仓库根目录应为：

```text
/cs/research/vecg/EasyPDE/quant/poseidon-quantized
```

如当前目录不是登录后的 home，直接进入上面的绝对路径。根据空余 GPU 的实际编号设置可见设备，再检查正式环境：

```bash
export CUDA_VISIBLE_DEVICES=0
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

tmux 常用操作：

```bash
tmux ls
tmux attach-session -t poseidon-nspwc-sbpq-bit8-train1-default
```

离开会话而保持任务运行：按 `Ctrl+b`，松开后按 `d`。重连时先 SSH 到原来的主机，再 attach 原会话。tmux 会话属于具体主机，不能在另一台主机上 attach 同名会话来查看原任务。

## 2. 主实验设置与命名

| 项目 | 本轮约定 |
| --- | --- |
| 数据集 | NS-PwC、CE-RM、Wave-Gauss |
| 配置位宽 | 2、3、4、8 bits |
| 方法 | Uniform、PPQ **layerwise**、BRECQ、SBPQ |
| 第一组 | 训练 `seed=1`；`random_calibration=n`，固定校准，文件后缀 `default` |
| 第二组 | 训练 `seed=7`；`random_calibration=y`、`calibration_seed=7`，后缀 `seed7` |
| 第三组 | 训练 `seed=42`；`random_calibration=y`、`calibration_seed=42`，后缀 `seed42` |
| 量化校准轨迹 | `traj_num=16` |
| 量化校准时间 | `final_time=10`、`time_back=0` |
| 校准加载 batch / workers | `batch_size=2`、`num_workers=0` |
| 激活精度 | 32；weight-only |
| SBPQ | `init_bits=8`、`mc_samples=10`、`total_steps=800`、`learning_rate=3.75e-5` |
| PPQ layerwise | **`init_bits=8`、`mc_samples=10`、每层 `epochs=5`**；保留 `learning_rate=9.1e-4`，模板使用 `save_every=1000` |
| BRECQ | 不使用 MC；每块 `reconstruction_iterations=20000`、`rec_batchsize=32`、`reconstruction_lr=1e-3` |
| 模型文件 | 实验目录内的 `compressed_model.pt` |

三组是绑定配对，不做训练与校准 seed 的 3×3 组合。**不再使用校准 seed1/2/3。** `default` 不是数值 seed：固定选取前 16 条轨迹，该组训练 seed 为 1，模板中的 `calibration_seed=42` 只是不会参与抽样的整数占位。三组结果同时包含训练随机性与校准数据选择的影响。

文件名后缀仍由校准模式决定：`n → default`，`y → seed<calibration_seed>`；训练 seed 保存于有效配置和实验表中。为保留旧训练 seed42 的 default 结果，本轮在两个输出根目录下加 `sbpq-led-v2` 子目录，文件名本身保留已确认的短格式：

```text
模型目录：
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_artifacts/<数据集>/sbpq-led-v2/<方法>-traj16-ft10-bit<B>-<default或seed7或seed42>/

评估 JSON：
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_evaluation/<数据集>/sbpq-led-v2/<方法>-traj16-ft10-bit<B>-<default或seed7或seed42>.json
```

**本轮模板显式传入 `--save_path` 和 `--output_path`。** 当前代码省略路径时仍生成原来的非版本目录；评估的默认路径推断也不会保留额外版本子目录。不要省略这些参数而覆盖或混入旧结果。

方法标识：Uniform → `uniform`，PPQ layerwise → `ppq-layerwise`，BRECQ → `brecq`，SBPQ → `sbpq`。
SBPQ/PPQ 的 B 使用 `target_bits`；Uniform/BRECQ 使用 `weight_bits`。**目录位宽是配置位宽；实际平均位宽另读 `avg_weight_bits`。**

历史 PPQ layerwise 的 default 模型使用 MC=5、init_bits=9、epochs=21、训练 seed42，配置为 8 bits，实际平均约 **2.237410 bits**。layerwise 的 MDL prior 不以 `target_bits` 为中心，`avg_cap_bits` 也不参与该变体；只改 `target_bits` 不能保证得到不同实际位宽。PPQ 应先独立调 gamma，并记录 gamma、初始化、MC、epochs 和实际平均位宽。调参运行另存 tuning 子目录，不覆盖正式目录。

各方法的完整参数与语义见 [PoseidonExpSetting.md](PoseidonExpSetting.md)。量化层范围、matmul 精度和位宽边界仍有实现差异，统一这些项目列为待实施建议；不能仅凭本表宣称计算范围与计算预算已严格相同。

## 3. 数据集与公共路径

| 短名 | `--dataset` | 原始模型目录 | 已完成 SBPQ 使用的 Sobolev order |
| --- | --- | --- | --- |
| NS-PwC | `fluids.incompressible.PiecewiseConstants` | `/cs/research/vecg/EasyPDE/model-store/poseidon/fluids.incompressible.PiecewiseConstants` | 2 |
| CE-RM | `fluids.compressible.RichtmyerMeshkov` | `/cs/research/vecg/EasyPDE/model-store/poseidon/fluids.compressible.RichtmyerMeshkov` | 1 |
| Wave-Gauss | `wave.Gaussians` | `/cs/research/vecg/EasyPDE/model-store/poseidon/wave.Gaussians` | 1 |

公共数据根目录：

```text
/cs/research/vecg/EasyPDE/quant/Structural-Aware-Probablistic-Quantization-for-Foundational-Physical-Models/poseidon/camlab-ethz/down_streams
```

`ft10` 表示量化校准的时间设置。旧评估 JSON 没有保存完整评估命令，不能仅凭文件名认定其评估时间范围。新评估需同时记录 `protocol`、`final_time`、`eval_obj` 和 batch 设置；Wave-Gauss 的数据最大时间为 15。

## 4. 状态定义

| 标记 | 含义 | 核验依据 |
| --- | --- | --- |
| ✅ 已完成 | 量化模型和评估 JSON 均已存在 | 完成状态／元数据、`compressed_model.pt`、结果 JSON |
| 🟨 待评估 | 模型已完成，尚未发现对应评估 JSON | 模型存在，评估文件不存在；不是“训练中” |
| 🔵 量化中／评估中 | 确认有对应的实际进程 | 记录主机、GPU、tmux、PID、日志和检查时间 |
| ⬜ 未做 | 共享目录未发现该配置的启动记录、模型或结果 | 启动前还需复核已有任务登记 |
| ⏸ 中断／失败 | 已启动但进程已结束，存在未完成状态 | 核对 `run_state.json`、checkpoint 和日志后决定 resume |
| ❔ 待核实 | 主机不可访问或文件与进程证据不一致 | 不直接当作空闲或完成 |

历史核查发现 **7 个已保存模型、6 个已有评估 JSON、1 个待评估模型**。它们均为训练 `seed=42`、`random_calibration=n`，不符合本轮“训练 1＋固定校准／训练 7＋随机校准 7／训练 42＋随机校准 42”中的任何一组；保留历史状态，不计入 v2 主表。

## 5. 主实验表：训练与校准三组配对（v2）

共 **4 位宽 × 4 方法 × 3 组配对 × 3 数据集 = 144 个实验格**。PPQ 全部指 **layerwise、MC=10、init_bits=8、每层 5 epochs**。
新协议尚未在本文登记启动，先标为未做；历史 default 结果见下一节。启动前复查共享目录及实际进程，再更新状态。

| 配置位宽 | 方法 | 训练 seed | 校准组 | NS-PwC | CE-RM | Wave-Gauss |
| --- | --- | --- | --- | --- | --- | --- |
| 2 bits | Uniform | 1 | default（n） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 2 bits | Uniform | 7 | seed7（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 2 bits | Uniform | 42 | seed42（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 2 bits | PPQ | 1 | default（n） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 2 bits | PPQ | 7 | seed7（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 2 bits | PPQ | 42 | seed42（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 2 bits | BRECQ | 1 | default（n） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 2 bits | BRECQ | 7 | seed7（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 2 bits | BRECQ | 42 | seed42（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 2 bits | SBPQ | 1 | default（n） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 2 bits | SBPQ | 7 | seed7（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 2 bits | SBPQ | 42 | seed42（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | Uniform | 1 | default（n） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | Uniform | 7 | seed7（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | Uniform | 42 | seed42（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | PPQ | 1 | default（n） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | PPQ | 7 | seed7（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | PPQ | 42 | seed42（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | BRECQ | 1 | default（n） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | BRECQ | 7 | seed7（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | BRECQ | 42 | seed42（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | SBPQ | 1 | default（n） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | SBPQ | 7 | seed7（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | SBPQ | 42 | seed42（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | Uniform | 1 | default（n） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | Uniform | 7 | seed7（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | Uniform | 42 | seed42（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | PPQ | 1 | default（n） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | PPQ | 7 | seed7（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | PPQ | 42 | seed42（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | BRECQ | 1 | default（n） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | BRECQ | 7 | seed7（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | BRECQ | 42 | seed42（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | SBPQ | 1 | default（n） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | SBPQ | 7 | seed7（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | SBPQ | 42 | seed42（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 8 bits | Uniform | 1 | default（n） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 8 bits | Uniform | 7 | seed7（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 8 bits | Uniform | 42 | seed42（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 8 bits | PPQ | 1 | default（n） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 8 bits | PPQ | 7 | seed7（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 8 bits | PPQ | 42 | seed42（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 8 bits | BRECQ | 1 | default（n） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 8 bits | BRECQ | 7 | seed7（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 8 bits | BRECQ | 42 | seed42（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 8 bits | SBPQ | 1 | default（n） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 8 bits | SBPQ | 7 | seed7（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 8 bits | SBPQ | 42 | seed42（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |

## 6. 历史 default 实验进度（训练 seed42）

以下为旧协议的固定校准结果，全部保留训练 `seed=42` 的真实设置及原路径。它们不能替代 v2 的训练 seed1/default，也不能替代随机校准 seed42。

| 配置位宽 | 方法 | 校准 | NS-PwC | CE-RM | Wave-Gauss |
| --- | --- | --- | --- | --- | --- |
| 8 bits | Uniform | default | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 8 bits | PPQ layerwise | default | 🟨 [D02：模型完成，待评估](#d02) | ⬜ 未做 | ⬜ 未做 |
| 8 bits | BRECQ | default | ✅ [D01](#d01) | ⬜ 未做 | ⬜ 未做 |
| 8 bits | SBPQ | default | ✅ [D03](#d03) | ✅ [D04](#d04) | ✅ [D05](#d05) |

补充历史结果：

| 编号 | 数据集 | 方法与设置 | 状态 | 是否计入主表 |
| --- | --- | --- | --- | --- |
| [D06](#d06) | NS-PwC | PPQ **global**，traj16、ft10、配置 8 bits、default | ✅ 模型与评估已完成 | 否；不能替代 PPQ layerwise |
| [D07](#d07) | NS-PwC | SBPQ，**traj64、ft20**、配置 8 bits、default | ✅ 历史模型与评估已完成 | 否；轨迹数与时间设置不同 |

### 模型与结果索引

以下均为服务器上的真实绝对路径。`不存在／待生成` 表示当前未发现文件，不代表评估已经运行。

<a id="d01"></a>

#### D01 — NS-PwC / brecq / default

- 状态：✅ 模型与评估均已完成。
- 配置：traj16、ft10、配置 8 bits；`random_calibration=n`、训练 `seed=42`。
- 实际平均位宽：**8.000000000**；训练主机：`canada-l.cs.ucl.ac.uk`。

模型文件：

```text
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_artifacts/NS-PwC/brecq-traj16-ft10-bit8-default/compressed_model.pt
```

评估 JSON（已存在）：

```text
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_evaluation/NS-PwC/brecq-traj16-ft10-bit8-default.json
```

<a id="d02"></a>

#### D02 — NS-PwC / ppq-layerwise / default

- 状态：🟨 模型已完成，待独立评估。
- 配置：traj16、ft10、配置 8 bits；`random_calibration=n`、训练 `seed=42`。
- 实际平均位宽：**2.237409993**；训练主机：`canada-l.cs.ucl.ac.uk`。
- 变体：**layerwise**；旧配置 `init_bits=9`、`mc_samples=5`、每层 `epochs=21`、`gamma=0.005`。不属于本轮 MC10／init8／5 epochs 配置。
- 位宽目前来自模型 `metadata.json`；对应结果 JSON 尚不存在。`run_state.json` 已是 completed，不再标为量化中。

模型文件：

```text
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_artifacts/NS-PwC/ppq-layerwise-traj16-ft10-bit8-default/compressed_model.pt
```

评估 JSON（目标位置，尚不存在）：

```text
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_evaluation/NS-PwC/ppq-layerwise-traj16-ft10-bit8-default.json
```

<a id="d03"></a>

#### D03 — NS-PwC / sbpq / default

- 状态：✅ 模型与评估均已完成。
- 配置：traj16、ft10、配置 8 bits；`random_calibration=n`、训练 `seed=42`。
- 实际平均位宽：**7.844344616**；训练主机：`canada-l.cs.ucl.ac.uk`。
- Sobolev order：`2`。

模型文件：

```text
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_artifacts/NS-PwC/sbpq-traj16-ft10-bit8-default/compressed_model.pt
```

评估 JSON（已存在）：

```text
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_evaluation/NS-PwC/sbpq-traj16-ft10-bit8-default.json
```

<a id="d04"></a>

#### D04 — CE-RM / sbpq / default

- 状态：✅ 模型与评估均已完成。
- 配置：traj16、ft10、配置 8 bits；`random_calibration=n`、训练 `seed=42`。
- 实际平均位宽：**7.653749943**；训练主机：`dory-l.cs.ucl.ac.uk`。
- Sobolev order：`1`。

模型文件：

```text
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_artifacts/CE-RM/sbpq-traj16-ft10-bit8-default/compressed_model.pt
```

评估 JSON（已存在）：

```text
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_evaluation/CE-RM/sbpq-traj16-ft10-bit8-default.json
```

<a id="d05"></a>

#### D05 — Wave-Gauss / sbpq / default

- 状态：✅ 模型与评估均已完成。
- 配置：traj16、ft10、配置 8 bits；`random_calibration=n`、训练 `seed=42`。
- 实际平均位宽：**7.931745529**；训练主机：`dory-l.cs.ucl.ac.uk`。
- Sobolev order：`1`。
- dory 的 tmux 历史中可见最近评估命令使用 `safe-svd`、test、`final_time=15`、`batch_size=32`、`physics_batch_size=10`；量化校准仍为 ft10。原 JSON 本身不包含这些命令参数。

模型文件：

```text
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_artifacts/Wave-Gauss/sbpq-traj16-ft10-bit8-default/compressed_model.pt
```

评估 JSON（已存在）：

```text
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_evaluation/Wave-Gauss/sbpq-traj16-ft10-bit8-default.json
```

<a id="d06"></a>

#### D06 — NS-PwC / ppq / default

- 状态：✅ 模型与评估均已完成。
- 配置：traj16、ft10、配置 8 bits；`random_calibration=n`、训练 `seed=42`。
- 实际平均位宽：**3.548605535**；训练主机：`canada-l.cs.ucl.ac.uk`。
- 变体：**global（补充结果）**；`init_bits=9`、`gamma=0.005`。

模型文件：

```text
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_artifacts/NS-PwC/ppq-traj16-ft10-bit8-default/compressed_model.pt
```

评估 JSON（已存在）：

```text
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_evaluation/NS-PwC/ppq-traj16-ft10-bit8-default.json
```

<a id="d07"></a>

#### D07 — NS-PwC / sbpq / default

- 状态：✅ 模型与评估均已完成。
- 配置：traj64、ft20、配置 8 bits；`random_calibration=n`、训练 `seed=42`。
- 实际平均位宽：**7.849734306**；训练主机：`历史记录未提供`。
- Sobolev order：`2`。
- 这是 traj64、ft20 的历史结果；没有新版 `run_state.json`，完成依据为已有模型、元数据和评估 JSON。

模型文件：

```text
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_artifacts/NS-PwC/sbpq-traj64-ft20-bit8-default/compressed_model.pt
```

评估 JSON（已存在）：

```text
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_evaluation/NS-PwC/sbpq-traj64-ft20-bit8-default.json
```


## 7. 正在运行的任务与检查范围

跨主机进程核查时间：**2026-11-19 11:11:45 Asia/Shanghai**。共尝试 `check_ucl_gpus` 配置的 54 台主机，成功核查 7 台：`albacore`, `canada`, `dory`, `goosander`, `ruddy`, `scaup`, `shark`。这些主机上未发现本人在本项目运行的量化或评估进程。

其余 47 台因 SSH 连接、超时或 host key 校验问题未完成核查，运行状态为 **❔ 待核实**；不能据此断言所有服务器均无任务。旧核查时共享的三个数据集实验目录未发现新的随机校准运行；本次 v2 设置更新没有重复全主机扫描，启动前需复查。

对最近执行这些实验的服务器还检查了 tmux：

| 主机 | 已有 tmux 会话 | 当前观察 | 本项目任务状态 |
| --- | --- | --- | --- |
| ruddy | 无 | 未发现本人 tmux server | 未发现量化／评估进程 |
| canada | `quant` | 窗口停在 csh；PPQ layerwise 已导出后返回 shell | 未发现量化／评估进程 |
| dory | `poseidon-quant-RM`、`poseidon-quant-gauss` | 窗口均停在 csh；已有 CE-RM、Wave-Gauss 结果文件 | 未发现量化／评估进程 |

GPU 上有其他进程并不代表本项目在跑；不要将其他用户的 GPU 占用填入本实验的“运行中”状态。

后续 agent 启动后在此登记；阶段应明确写“量化中”或“评估中”，不能只因为存在 tmux 就标为正在跑。

| 实验键（版本／数据集／位宽／方法／训练 seed／校准组） | 阶段 | 主机 | GPU | tmux | PID | 日志绝对路径 | 最近核查时间 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 当前已核实主机上未发现本项目在跑任务 | — | — | — | — | — | — | 见上方快照时间 |

历史结果可补做 **D02 的独立评估**，无需重新训练该模型；补评估不会使其成为 v2 的完成记录。

在对应服务器上复查进程与 tmux：

```bash
ps -u "$(id -u)" -o pid,etime,args
tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} pid=#{pane_pid} command=#{pane_current_command} cwd=#{pane_current_path}'
```

对运行中的具体实验，再核对其目录中的 `run_state.json`、`costs.json` 和已登记日志。成本文件包含历史启动会话，早期 interrupted 记录不代表最后一次已完成运行仍然中断。

## 8. 启动量化模板

本节在服务器仓库根目录、已激活的 `poseidon-quant` 环境中执行，使用第 1 节的 Bash tmux。
先填写数据集、位宽与训练 seed；下面以 NS-PwC、8 bits、训练 seed1/default 为例，不自动遍历整张实验表。`POSEIDON_TRAIN_SEED` 只选 1、7、42，校准设置由 case 绑定。

```bash
POSEIDON_LABEL=NS-PwC
POSEIDON_BITS=8
POSEIDON_TRAIN_SEED=1
POSEIDON_PROTOCOL=sbpq-led-v2
POSEIDON_PROJECT_ROOT=/cs/research/vecg/EasyPDE/quant/poseidon-quantized

case "$POSEIDON_TRAIN_SEED" in
  1)
    POSEIDON_RANDOM_CALIBRATION=n
    POSEIDON_CALIBRATION_SEED=42  # n 模式不使用此值抽样
    POSEIDON_CALIBRATION_TAG=default
    ;;
  7|42)
    POSEIDON_RANDOM_CALIBRATION=y
    POSEIDON_CALIBRATION_SEED="$POSEIDON_TRAIN_SEED"
    POSEIDON_CALIBRATION_TAG="seed$POSEIDON_TRAIN_SEED"
    ;;
  *) printf 'Unknown training seed: %s\n' "$POSEIDON_TRAIN_SEED" >&2; exit 1 ;;
esac

case "$POSEIDON_LABEL" in
  NS-PwC)
    POSEIDON_DATASET=fluids.incompressible.PiecewiseConstants
    POSEIDON_SOBOLEV_ORDER=2
    ;;
  CE-RM)
    POSEIDON_DATASET=fluids.compressible.RichtmyerMeshkov
    POSEIDON_SOBOLEV_ORDER=1
    ;;
  Wave-Gauss)
    POSEIDON_DATASET=wave.Gaussians
    POSEIDON_SOBOLEV_ORDER=1
    ;;
  *) printf 'Unknown dataset: %s\n' "$POSEIDON_LABEL" >&2; exit 1 ;;
esac

POSEIDON_MODEL_PATH="/cs/research/vecg/EasyPDE/model-store/poseidon/$POSEIDON_DATASET"
POSEIDON_DATA_PATH=/cs/research/vecg/EasyPDE/quant/Structural-Aware-Probablistic-Quantization-for-Foundational-Physical-Models/poseidon/camlab-ethz/down_streams
POSEIDON_ARTIFACT_ROOT="$POSEIDON_PROJECT_ROOT/quantization_artifacts/$POSEIDON_LABEL/$POSEIDON_PROTOCOL"
POSEIDON_RESULT_ROOT="$POSEIDON_PROJECT_ROOT/quantization_evaluation/$POSEIDON_LABEL/$POSEIDON_PROTOCOL"
POSEIDON_RUN_SUFFIX="traj16-ft10-bit$POSEIDON_BITS-$POSEIDON_CALIBRATION_TAG"

POSEIDON_COMMON=(
  --model_path "$POSEIDON_MODEL_PATH"
  --dataset "$POSEIDON_DATASET"
  --data_path "$POSEIDON_DATA_PATH"
  --traj_num 16
  --random_calibration "$POSEIDON_RANDOM_CALIBRATION"
  --calibration_seed "$POSEIDON_CALIBRATION_SEED"
  --final_time 10 --time_back 0
  --batch_size 2 --num_workers 0 --activation_bits 32
  --seed "$POSEIDON_TRAIN_SEED" --DEV cuda
)

POSEIDON_LOG_DIR="$POSEIDON_PROJECT_ROOT/logs/poseidon/$POSEIDON_LABEL/$POSEIDON_PROTOCOL"
mkdir -p "$POSEIDON_LOG_DIR"
set -o pipefail
```

四种方法**选一种**执行；每种方法都显式设置版本化保存路径。其日志名与正式运行名对应。

Uniform：

```bash
POSEIDON_RUN_NAME="uniform-$POSEIDON_RUN_SUFFIX"
POSEIDON_RUN_DIR="$POSEIDON_ARTIFACT_ROOT/$POSEIDON_RUN_NAME"
python -u -m UniformQ.uniformq \
  --config configs/uniform/default.json \
  "${POSEIDON_COMMON[@]}" --weight_bits "$POSEIDON_BITS" \
  --save_path "$POSEIDON_RUN_DIR" \
  2>&1 | tee -a "$POSEIDON_LOG_DIR/$POSEIDON_RUN_NAME.log"
```

PPQ layerwise：

```bash
# 0.005 是历史起点；正式运行需填入为本次实际位宽选定、并已记录的 gamma。
# 调 gamma 时另设 tuning 目录，不能覆盖下面的正式目录。
POSEIDON_PPQ_GAMMA=0.005
POSEIDON_RUN_NAME="ppq-layerwise-$POSEIDON_RUN_SUFFIX"
POSEIDON_RUN_DIR="$POSEIDON_ARTIFACT_ROOT/$POSEIDON_RUN_NAME"
python -u -m PPQ.ppq \
  --config configs/ppq/layerwise.json --variant layerwise \
  "${POSEIDON_COMMON[@]}" --target_bits "$POSEIDON_BITS" \
  --init_bits 8 --mc_samples 10 --epochs 5 \
  --learning_rate 9.1e-4 --eta 1e-6 --percentile 1e-4 \
  --gamma "$POSEIDON_PPQ_GAMMA" --save_every 1000 \
  --save_path "$POSEIDON_RUN_DIR" \
  2>&1 | tee -a "$POSEIDON_LOG_DIR/$POSEIDON_RUN_NAME.log"
```

BRECQ：

```bash
POSEIDON_RUN_NAME="brecq-$POSEIDON_RUN_SUFFIX"
POSEIDON_RUN_DIR="$POSEIDON_ARTIFACT_ROOT/$POSEIDON_RUN_NAME"
python -u -m BRECQ.brecq \
  --config configs/brecq/default.json \
  "${POSEIDON_COMMON[@]}" --weight_bits "$POSEIDON_BITS" \
  --reconstruction_iterations 20000 --rec_batchsize 32 \
  --reconstruction_lr 1e-3 --save_every 1000 \
  --save_path "$POSEIDON_RUN_DIR" \
  2>&1 | tee -a "$POSEIDON_LOG_DIR/$POSEIDON_RUN_NAME.log"
```

SBPQ：

```bash
POSEIDON_RUN_NAME="sbpq-$POSEIDON_RUN_SUFFIX"
POSEIDON_RUN_DIR="$POSEIDON_ARTIFACT_ROOT/$POSEIDON_RUN_NAME"
python -u -m SBPQ.sbpq \
  --config configs/sbpq/default.json \
  "${POSEIDON_COMMON[@]}" --target_bits "$POSEIDON_BITS" \
  --init_bits 8 --mc_samples 10 --total_steps 800 \
  --learning_rate 3.75e-5 --eta 1e-6 --percentile 1e-4 \
  --prior_scale 1 --likelihood_scale 1 --beta_kappa 150 --delta_bits 2 \
  --minimum_bits 1 --maximum_bits 16 \
  --sobolev_order "$POSEIDON_SOBOLEV_ORDER" --sobolev_norm l1 \
  --sobolev_order_weights 1 --task_loss_weight 1 --sobolev_loss_weight 1 \
  --save_every 100 --save_path "$POSEIDON_RUN_DIR" \
  2>&1 | tee -a "$POSEIDON_LOG_DIR/$POSEIDON_RUN_NAME.log"
```

PPQ 的 5 epochs 指每个被选中 Linear 都使用全部校准 batch 训练 5 轮，不是全模型合计 5 次更新。BRECQ 不使用 MC 参数，保留其块级重构配置。其余专用参数、待实施的选层与数值精度建议见设置文档。

修改 gamma、初始化、Sobolev order 或预算时要记录有效配置；同一个短目录名不会编码这些变化，已有目录的配置校验仍会拒绝混用不同实验。旧 PPQ 的 resume 会恢复其原来的 MC5／init9／21 epochs，不自动变成本轮新实验。

## 9. 独立评估与 resume

### 9.1 v2 模型的独立评估

以下接第 8 节中**已完成量化的那一种方法**，沿用 `POSEIDON_RUN_DIR`、`POSEIDON_RUN_NAME`、`POSEIDON_DATASET` 等变量。评估终点 10 是命令示例；正式比较前需统一各数据集的评估终点、协议与 batch 设置，并记录实际命令。它不是根据量化文件名推断的旧评估范围。

```bash
POSEIDON_EVAL_FINAL_TIME=10
POSEIDON_EVAL_OUTPUT="$POSEIDON_RESULT_ROOT/$POSEIDON_RUN_NAME.json"
mkdir -p "$POSEIDON_RESULT_ROOT"
python -u -m evaluation.evaluate \
  --protocol safe-svd \
  --model_path "$POSEIDON_RUN_DIR" \
  --checkpoint "$POSEIDON_MODEL_PATH" \
  --dataset "$POSEIDON_DATASET" \
  --data_path "$POSEIDON_DATA_PATH" \
  --final_time "$POSEIDON_EVAL_FINAL_TIME" \
  --eval_obj ongt --batch_size 8 --physics_batch_size 10 \
  --num_workers 0 --DEV cuda \
  --output_path "$POSEIDON_EVAL_OUTPUT" \
  2>&1 | tee -a "$POSEIDON_LOG_DIR/$POSEIDON_RUN_NAME-eval.log"
```

`--checkpoint` 用于同时计算原模型与压缩模型的指标。**必须保留版本化的 `--output_path`**；默认推断会落回旧的非版本目录。改变评估协议或终点时另存明确命名的评估文件，并登记其设置。

### 9.2 补做历史 D02 评估

D02 的量化已经完成，以下仅为历史模型补评估模板，不会计入 v2 主表。示例显式采用 `final_time=10`；不反推其他历史 JSON 的评估终点。

```bash
POSEIDON_D02_MODEL=/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_artifacts/NS-PwC/ppq-layerwise-traj16-ft10-bit8-default
POSEIDON_D02_OUTPUT=/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_evaluation/NS-PwC/ppq-layerwise-traj16-ft10-bit8-default.json
python -u -m evaluation.evaluate \
  --protocol safe-svd --model_path "$POSEIDON_D02_MODEL" \
  --checkpoint /cs/research/vecg/EasyPDE/model-store/poseidon/fluids.incompressible.PiecewiseConstants \
  --dataset fluids.incompressible.PiecewiseConstants \
  --data_path /cs/research/vecg/EasyPDE/quant/Structural-Aware-Probablistic-Quantization-for-Foundational-Physical-Models/poseidon/camlab-ethz/down_streams \
  --final_time 10 --eval_obj ongt --batch_size 8 --physics_batch_size 10 \
  --num_workers 0 --DEV cuda --output_path "$POSEIDON_D02_OUTPUT"
```

### 9.3 恢复未完成的量化

**D02 已完成，不需要重做量化。** 恢复时填写真实的未完成目录，选择对应方法入口；以下占位路径须替换。

```bash
POSEIDON_RESUME_DIR="/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_artifacts/<数据集>/sbpq-led-v2/<真实未完成实验目录>"
python -u -m PPQ.ppq --variant layerwise --resume "$POSEIDON_RESUME_DIR"
```

SBPQ 和 BRECQ 同理使用 `python -u -m SBPQ.sbpq --resume <真实目录>`、`python -u -m BRECQ.brecq --resume <真实目录>`。恢复入口读取原实验配置，勿叠加新 JSON 或不同超参数；要使用新预算就新建实验目录。Uniform 不使用训练式 resume。

历史运行也从其原目录恢复。修改本仓库的 layerwise 默认文件不改变旧 checkpoint 中的训练预算。只有 checkpoint／`work` 目录不代表模型已完成，需要确认导出的 `compressed_model.pt` 和完成记录。

## 10. 后续 agents 更新记录的方法

1. 启动前核对版本、数据集、配置位宽、方法、训练 seed 与校准组；严格使用 1/default、7/seed7、42/seed42 三组，检查共享目录与运行登记，避免重复启动。
2. 从真实目录读取 `experiment.json`／`metadata.json`，核对有效配置；PPQ v2 必须为 init8／MC10／每层 5 epochs。旧模型的训练 seed42/default 不能计入新表。
3. 确认设置文档中尚未实施的选层、精度与搜索边界差异；若改变算法实现，登记源码提交并另开协议版本，勿与当前设置混记。
4. 选择 GPU，创建独立 tmux。启动后把对应实验格改为 `🔵 量化中`，登记主机、GPU、tmux、PID、日志、配置与时间。
5. 量化导出后若没有评估 JSON，改为 `🟨 待评估`；独立评估启动时改为 `🔵 评估中`，记录完整评估命令、版本化输出路径与评估终点。
6. 模型和评估均完成后改为 `✅ 已完成`，补上模型、结果路径和实际 `avg_weight_bits`；PPQ 还需记录 gamma、初始化、MC 与 epochs。
7. 进程消失但状态不完整时标为 `⏸ 中断／失败`，记录 checkpoint 和错误；不要仅凭 tmux 会话存在标为在跑。
8. 每次实际核查更新文首状态时间与运行表；只修改设置不冒充重新核查。未核实的主机保持“待核实”。三组均值与标准差应注明它们联合改变训练和校准随机设置。

本文件位于本地仓库：`/Users/hcj/Project/poseidon-quantized/PoseidonExperiments.md`。上述运行路径均为服务器共享仓库路径。
