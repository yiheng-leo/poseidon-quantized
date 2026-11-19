# Poseidon 实验操作与进度

状态核查：2026-11-19 11:14（Asia/Shanghai；跨主机扫描时间及覆盖范围见“正在运行的任务”）。

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
tmux new-session -s poseidon-nspwc-sbpq-bit8-seed1 /bin/bash
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
tmux attach-session -t poseidon-nspwc-sbpq-bit8-seed1
```

离开会话而保持任务运行：按 `Ctrl+b`，松开后按 `d`。重连时先 SSH 到原来的主机，再 attach 原会话。tmux 会话属于具体主机，不能在另一台主机上 attach 同名会话来查看原任务。

## 2. 主实验设置与命名

| 项目 | 约定 |
| --- | --- |
| 数据集 | NS-PwC、CE-RM、Wave-Gauss |
| 配置位宽 | 2、3、4、8 bits |
| 方法 | Uniform、PPQ **layerwise**、BRECQ、SBPQ |
| 主表随机重复 | `random_calibration=y`，`calibration_seed=1/2/3`，分别记作 seed1/seed2/seed3 |
| 固定校准对照 | `random_calibration=n`，记作 default；与随机 seed1/2/3 分开 |
| 量化校准轨迹 | `traj_num=16` |
| 量化校准时间 | `final_time=10`、`time_back=0` |
| 内存 batch | `batch_size=2` |
| 训练随机种子 | 当前已有运行均为 `seed=42`；复用这些设置时显式传 `--seed 42` |
| 激活精度 | 32；当前四种方法按 weight-only 设置比较 |
| 模型文件 | 实验目录内的 `compressed_model.pt` |

**seed1/seed2/seed3 指校准抽样种子，不是训练 `--seed`。** 当 `random_calibration=n` 时，无论训练种子或 `calibration_seed` 是多少，目录后缀都是 `default`；完整数值设置仍保存在配置与元数据中。

服务器已启用下面的短路径规则。省略 `--save_path` 和 `--output_path` 即自动生成；显式传入的路径直接使用，不追加长参数子目录。

```text
模型目录：
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_artifacts/<数据集>/<方法>-traj16-ft10-bit<B>-<default或seedN>/

评估 JSON：
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_evaluation/<数据集>/<方法>-traj16-ft10-bit<B>-<default或seedN>.json
```

方法标识：Uniform → `uniform`，PPQ layerwise → `ppq-layerwise`，BRECQ → `brecq`，SBPQ → `sbpq`。
SBPQ/PPQ 的 `B` 使用 `target_bits`；Uniform/BRECQ 使用 `weight_bits`。**目录位宽是配置位宽；实际平均位宽必须读取结果 JSON 的 `avg_weight_bits`。**

当前 PPQ layerwise 的 default 模型配置为 8 bits，但实际平均约为 **2.237410 bits**。其 `target_bits` 不会让 MDL prior 以目标位宽为中心；只改 `target_bits` 不能保证得到不同实际位宽。后续 PPQ 运行应明确记录 `gamma`、初始化位宽与实际平均位宽，避免把相同数值配置仅改标签的运行当成不同实际位宽结果。

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

本次发现 **7 个已保存模型、6 个已有评估 JSON、1 个待评估模型**。它们均使用 `random_calibration=n`，因此不能填入随机 seed1/2/3 的完成格。

## 5. 主实验表：随机校准 seed1 / seed2 / seed3

共 **4 位宽 × 4 方法 × 3 seeds × 3 数据集 = 144 个实验格**。表中的 PPQ 全部指 **layerwise**。
本次未发现这 144 个随机校准配置的启动记录或产物，先标为未做；已有 default 结果见下一节。

| 配置位宽 | 方法 | 校准 seed | NS-PwC | CE-RM | Wave-Gauss |
| --- | --- | --- | --- | --- | --- |
| 2 bits | Uniform | seed1 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 2 bits | Uniform | seed2 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 2 bits | Uniform | seed3 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 2 bits | PPQ | seed1 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 2 bits | PPQ | seed2 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 2 bits | PPQ | seed3 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 2 bits | BRECQ | seed1 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 2 bits | BRECQ | seed2 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 2 bits | BRECQ | seed3 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 2 bits | SBPQ | seed1 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 2 bits | SBPQ | seed2 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 2 bits | SBPQ | seed3 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | Uniform | seed1 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | Uniform | seed2 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | Uniform | seed3 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | PPQ | seed1 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | PPQ | seed2 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | PPQ | seed3 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | BRECQ | seed1 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | BRECQ | seed2 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | BRECQ | seed3 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | SBPQ | seed1 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | SBPQ | seed2 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | SBPQ | seed3 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | Uniform | seed1 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | Uniform | seed2 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | Uniform | seed3 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | PPQ | seed1 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | PPQ | seed2 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | PPQ | seed3 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | BRECQ | seed1 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | BRECQ | seed2 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | BRECQ | seed3 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | SBPQ | seed1 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | SBPQ | seed2 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | SBPQ | seed3 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 8 bits | Uniform | seed1 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 8 bits | Uniform | seed2 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 8 bits | Uniform | seed3 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 8 bits | PPQ | seed1 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 8 bits | PPQ | seed2 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 8 bits | PPQ | seed3 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 8 bits | BRECQ | seed1 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 8 bits | BRECQ | seed2 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 8 bits | BRECQ | seed3 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 8 bits | SBPQ | seed1 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 8 bits | SBPQ | seed2 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 8 bits | SBPQ | seed3 | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |

## 6. 已有 default 实验进度

本表独立记录固定校准结果，不将 default 自动当作 seed1。

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
- 变体：**layerwise**；`init_bits=9`、`gamma=0.005`。
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

其余 47 台因 SSH 连接、超时或 host key 校验问题未完成核查，运行状态为 **❔ 待核实**；不能据此断言所有服务器均无任务。共享的三个数据集实验目录也未发现新增 seed1/2/3 启动记录。

对最近执行这些实验的服务器还检查了 tmux：

| 主机 | 已有 tmux 会话 | 当前观察 | 本项目任务状态 |
| --- | --- | --- | --- |
| ruddy | 无 | 未发现本人 tmux server | 未发现量化／评估进程 |
| canada | `quant` | 窗口停在 csh；PPQ layerwise 已导出后返回 shell | 未发现量化／评估进程 |
| dory | `poseidon-quant-RM`、`poseidon-quant-gauss` | 窗口均停在 csh；已有 CE-RM、Wave-Gauss 结果文件 | 未发现量化／评估进程 |

GPU 上有其他进程并不代表本项目在跑；不要将其他用户的 GPU 占用填入本实验的“运行中”状态。

后续 agent 启动后在此登记；阶段应明确写“量化中”或“评估中”，不能只因为存在 tmux 就标为正在跑。

| 实验键（数据集／位宽／方法／校准） | 阶段 | 主机 | GPU | tmux | PID | 日志绝对路径 | 最近核查时间 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 当前已核实主机上未发现本项目在跑任务 | — | — | — | — | — | — | 见上方快照时间 |

当前可优先补做 **D02 的独立评估**，无需重新训练该模型。

在对应服务器上复查进程与 tmux：

```bash
ps -u "$(id -u)" -o pid,etime,args
tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} pid=#{pane_pid} command=#{pane_current_command} cwd=#{pane_current_path}'
```

对运行中的具体实验，再核对其目录中的 `run_state.json`、`costs.json` 和已登记日志。成本文件包含历史启动会话，早期 interrupted 记录不代表最后一次已完成运行仍然中断。

## 8. 启动量化模板

本节命令均在服务器仓库根目录、已激活的 `poseidon-quant` 环境中执行。使用第 1 节的 Bash tmux。
先填写本次实验变量；下面仅给出 NS-PwC、配置 8 bits、随机 seed1 的模板，没有自动启动整张实验表。

```bash
POSEIDON_LABEL=NS-PwC
POSEIDON_BITS=8
POSEIDON_RANDOM_CALIBRATION=y
POSEIDON_CALIBRATION_SEED=1

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

POSEIDON_COMMON=(
  --model_path "$POSEIDON_MODEL_PATH"
  --dataset "$POSEIDON_DATASET"
  --data_path "$POSEIDON_DATA_PATH"
  --traj_num 16
  --random_calibration "$POSEIDON_RANDOM_CALIBRATION"
  --calibration_seed "$POSEIDON_CALIBRATION_SEED"
  --final_time 10 --time_back 0
  --batch_size 2 --activation_bits 32
  --seed 42 --DEV cuda
)

if [ "$POSEIDON_RANDOM_CALIBRATION" = n ]; then
  POSEIDON_CALIBRATION_TAG=default
else
  POSEIDON_CALIBRATION_TAG="seed$POSEIDON_CALIBRATION_SEED"
fi

POSEIDON_LOG_DIR="/cs/research/vecg/EasyPDE/quant/poseidon-quantized/logs/poseidon/$POSEIDON_LABEL"
mkdir -p "$POSEIDON_LOG_DIR"
set -o pipefail
```

四种方法**选一种**执行。下面省略 `save_path`，由服务器按已确认规则自动生成。

Uniform：

```bash
python -u -m UniformQ.uniformq \
  --config configs/uniform/default.json \
  "${POSEIDON_COMMON[@]}" --weight_bits "$POSEIDON_BITS" \
  2>&1 | tee -a "$POSEIDON_LOG_DIR/uniform-traj16-ft10-bit$POSEIDON_BITS-$POSEIDON_CALIBRATION_TAG.log"
```

PPQ layerwise：

```bash
# 历史 default 运行使用 gamma=0.005；这是数值参数，不是“自动达到目标 bits”的开关。
# 为本次实际位宽实验确定 gamma 后，在这里填写并记录。
POSEIDON_PPQ_GAMMA=0.005

python -u -m PPQ.ppq \
  --config configs/ppq/layerwise.json --variant layerwise \
  "${POSEIDON_COMMON[@]}" --target_bits "$POSEIDON_BITS" \
  --gamma "$POSEIDON_PPQ_GAMMA" \
  2>&1 | tee -a "$POSEIDON_LOG_DIR/ppq-layerwise-traj16-ft10-bit$POSEIDON_BITS-$POSEIDON_CALIBRATION_TAG.log"
```

BRECQ：

```bash
python -u -m BRECQ.brecq \
  --config configs/brecq/default.json \
  "${POSEIDON_COMMON[@]}" --weight_bits "$POSEIDON_BITS" \
  2>&1 | tee -a "$POSEIDON_LOG_DIR/brecq-traj16-ft10-bit$POSEIDON_BITS-$POSEIDON_CALIBRATION_TAG.log"
```

SBPQ：

```bash
python -u -m SBPQ.sbpq \
  --config configs/sbpq/default.json \
  "${POSEIDON_COMMON[@]}" --target_bits "$POSEIDON_BITS" \
  --sobolev_order "$POSEIDON_SOBOLEV_ORDER" \
  2>&1 | tee -a "$POSEIDON_LOG_DIR/sbpq-traj16-ft10-bit$POSEIDON_BITS-$POSEIDON_CALIBRATION_TAG.log"
```

这些模板保留当前配置文件中的其余优化参数。若修改 `gamma`、初始化位宽、Sobolev order、优化预算等，应记录实际配置；同一个短目录名不会编码这些变化，已有目录的配置校验仍会拒绝混用不同实验。

## 9. 独立评估与 resume

### 9.1 对应模型的独立评估

先填写真实模型目录与本次评估终点。以下示例为 **D02**，显式以 `final_time=10` 评估；这只是待执行模板，不代表其评估已经完成，也不反推旧 JSON 的评估终点。

```bash
POSEIDON_EVAL_MODEL=/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_artifacts/NS-PwC/ppq-layerwise-traj16-ft10-bit8-default
POSEIDON_EVAL_DATASET=fluids.incompressible.PiecewiseConstants
POSEIDON_EVAL_CHECKPOINT=/cs/research/vecg/EasyPDE/model-store/poseidon/fluids.incompressible.PiecewiseConstants
POSEIDON_EVAL_FINAL_TIME=10
POSEIDON_EVAL_DATA_PATH=/cs/research/vecg/EasyPDE/quant/Structural-Aware-Probablistic-Quantization-for-Foundational-Physical-Models/poseidon/camlab-ethz/down_streams

python -u -m evaluation.evaluate \
  --protocol safe-svd \
  --model_path "$POSEIDON_EVAL_MODEL" \
  --checkpoint "$POSEIDON_EVAL_CHECKPOINT" \
  --dataset "$POSEIDON_EVAL_DATASET" \
  --data_path "$POSEIDON_EVAL_DATA_PATH" \
  --final_time "$POSEIDON_EVAL_FINAL_TIME" \
  --eval_obj ongt --batch_size 8 --physics_batch_size 10 \
  --num_workers 0 --DEV cuda
```

没有填写 `output_path` 时，服务器从模型配置推断同名评估 JSON，D02 的目标位置见索引。上述 `--checkpoint` 用于同时计算原模型与压缩模型的指标。先确认评估范围，再将所有主实验按同一口径评估；改变评估协议或终点时应另存显式 `output_path` 并登记。

### 9.2 恢复未完成的量化

**D02 已完成，不需要重做量化。** 恢复时填写对应的未完成实验目录，并选择正确的方法入口；下面的占位路径必须替换为真实目录。

```bash
POSEIDON_RESUME_DIR="/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_artifacts/<数据集>/<真实未完成实验目录>"
python -u -m PPQ.ppq --variant layerwise --resume "$POSEIDON_RESUME_DIR"
```

SBPQ 和 BRECQ 同理使用 `python -u -m SBPQ.sbpq --resume <真实目录>`、`python -u -m BRECQ.brecq --resume <真实目录>`。
恢复入口读取原实验配置，目录与超参数不应另起一套；Uniform 不使用训练式 resume。
只有 checkpoint／`work` 目录不代表模型已完成；需要确认导出的 `compressed_model.pt` 和完成记录。

## 10. 后续 agents 更新记录的方法

1. 启动前核对主表中的数据集、配置位宽、方法和校准 seed；检查共享目录与“正在运行的任务”，避免重复启动。
2. 从本文的服务器绝对路径读取现有 `experiment.json`／`metadata.json`；不要从旧父目录名称猜测实际参数。
3. 选择 GPU，创建独立 tmux。启动后把对应实验格改为 `🔵 量化中`，登记主机、GPU、tmux、PID、日志和时间。
4. 量化导出后，若没有评估 JSON，改为 `🟨 待评估`；独立评估启动时改为 `🔵 评估中`。
5. 模型和评估均完成后改为 `✅ 已完成`，补上模型、结果路径和实际 `avg_weight_bits`；PPQ 还需记录该次 `gamma`。
6. 进程消失但状态不完整时标为 `⏸ 中断／失败`，记录已有 checkpoint 和错误；不要仅凭 tmux 会话存在标为在跑。
7. 每次核查更新文首时间与运行表。未核实的远端主机保持“待核实”；实际完成与否优先查看共享产物及完成记录。

本文件位于本地仓库：`/Users/hcj/Project/poseidon-quantized/PoseidonExperiments.md`。上述运行路径均为服务器共享仓库路径。
