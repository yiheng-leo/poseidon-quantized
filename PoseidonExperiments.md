# Poseidon 实验操作与进度

最近状态核查：2026-11-21 00:14（Asia/Shanghai；本次仅实查 dory 的 gamma1e-6 与 shark 的 gamma3e-7：进程、心跳、逐层进度、checkpoint 及最终产物；其余主机／任务未重新核查，见第 5.4、7 节）。

实验设置更新：2026-11-19。完整配置及待实施的对齐、提速建议见 [PoseidonExpSetting.md](PoseidonExpSetting.md)；协议为 `sbpq-led-v2`。初始队列登记 63 项正式实验和 18 项 PPQ 扫描，11 月 20 日追加 NS-PwC 8-bit 的 `gamma=1e-6`、`3e-7`，共 83 项。**11 月 21 日 00:14 两项均仍运行：1e-6 完成 469/533 层（87.99%），3e-7 完成 372/533 层（69.79%），均未导出最终模型或最终实际位宽。** 其余任务沿用 11 月 20 日的核查记录（10 项完成、4 项旧任务中断、67 项未启动），本次未重新核验；详情见第 5.3、5.4、7 节。

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
| ⏳ 已入队／等待 GPU | 任务已登记且有存活的等待进程，尚未领取／启动 | 共享队列 `plan.json`、`states/`、`workers/`、PID 和 tmux；不记成量化中 |
| ⏳ 等待 gamma | PPQ 正式任务已登记，待本数据集的 gamma 匹配后释放 | `states/` 为 `waiting_gamma`，对应 `selections/` 尚未选定 |
| ⬜ 未做 | 共享目录未发现该配置的启动记录、模型或结果 | 启动前还需复核已有任务登记 |
| ⏸ 中断／失败 | 已启动但进程已结束，存在未完成状态 | 核对 `run_state.json`、checkpoint 和日志后决定 resume |
| ⏸ 队列停止 | 已登记但尚未启动，负责领取任务的等待进程已不存在 | 队列记录仍保留；不等同于该实验运行失败，也不会自动推进 |
| ❔ 待核实 | 主机不可访问或文件与进程证据不一致 | 不直接当作空闲或完成 |

历史核查发现 **7 个已保存模型、6 个已有评估 JSON、1 个待评估模型**。它们均为训练 `seed=42`、`random_calibration=n`，不符合本轮“训练 1＋固定校准／训练 7＋随机校准 7／训练 42＋随机校准 42”中的任何一组；保留历史状态，不计入 v2 主表。

## 5. 主实验表：训练与校准三组配对（v2）

共 **4 位宽 × 4 方法 × 3 组配对 × 3 数据集 = 144 个实验格**。PPQ 全部指 **layerwise、MC=10、init_bits=8、每层 5 epochs**。
下表 63 个正式实验格已登记共享队列：8 格完成、1 格模型完成但评估中断、2 格准备阶段中断、52 格未启动且队列停止（其中 PPQ 8-bit 的 9 格尚待 gamma）。其余 81 格未入本次队列。第 5.1 节的 tuning 运行不计入正式完成格。
NS-PwC 的 PPQ 正式运行前，先完成下方 [gamma sweep 待办](#nspwc-ppq-gamma-sweep)，为 2、3、4、8 bits 分别选定 gamma。

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
| 2 bits | SBPQ | 1 | default（n） | ⏸ 队列停止 | ⏸ 队列停止 | ⏸ 队列停止 |
| 2 bits | SBPQ | 7 | seed7（y） | ⏸ 队列停止 | ⏸ 队列停止 | ⏸ 队列停止 |
| 2 bits | SBPQ | 42 | seed42（y） | ⏸ 队列停止 | ⏸ 队列停止 | ⏸ 队列停止 |
| 3 bits | Uniform | 1 | default（n） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | Uniform | 7 | seed7（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | Uniform | 42 | seed42（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | PPQ | 1 | default（n） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | PPQ | 7 | seed7（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | PPQ | 42 | seed42（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | BRECQ | 1 | default（n） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | BRECQ | 7 | seed7（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | BRECQ | 42 | seed42（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 3 bits | SBPQ | 1 | default（n） | ⏸ 队列停止 | ⏸ 队列停止 | ⏸ 队列停止 |
| 3 bits | SBPQ | 7 | seed7（y） | ⏸ 队列停止 | ⏸ 队列停止 | ⏸ 队列停止 |
| 3 bits | SBPQ | 42 | seed42（y） | ⏸ 队列停止 | ⏸ 队列停止 | ⏸ 队列停止 |
| 4 bits | Uniform | 1 | default（n） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | Uniform | 7 | seed7（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | Uniform | 42 | seed42（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | PPQ | 1 | default（n） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | PPQ | 7 | seed7（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | PPQ | 42 | seed42（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | BRECQ | 1 | default（n） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | BRECQ | 7 | seed7（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | BRECQ | 42 | seed42（y） | ⬜ 未做 | ⬜ 未做 | ⬜ 未做 |
| 4 bits | SBPQ | 1 | default（n） | ⏸ 队列停止 | ⏸ 队列停止 | ⏸ 队列停止 |
| 4 bits | SBPQ | 7 | seed7（y） | ⏸ 队列停止 | ⏸ 队列停止 | ⏸ 队列停止 |
| 4 bits | SBPQ | 42 | seed42（y） | ⏸ 队列停止 | ⏸ 队列停止 | ⏸ 队列停止 |
| 8 bits | Uniform | 1 | default（n） | ✅ 已完成 | ✅ 已完成 | ✅ 已完成 |
| 8 bits | Uniform | 7 | seed7（y） | ✅ 已完成 | ✅ 已完成 | ✅ 已完成 |
| 8 bits | Uniform | 42 | seed42（y） | ✅ 已完成 | 🟨 待评估（中断） | ✅ 已完成 |
| 8 bits | PPQ | 1 | default（n） | ⏸ 待 gamma，队列停止 | ⏸ 待 gamma，队列停止 | ⏸ 待 gamma，队列停止 |
| 8 bits | PPQ | 7 | seed7（y） | ⏸ 待 gamma，队列停止 | ⏸ 待 gamma，队列停止 | ⏸ 待 gamma，队列停止 |
| 8 bits | PPQ | 42 | seed42（y） | ⏸ 待 gamma，队列停止 | ⏸ 待 gamma，队列停止 | ⏸ 待 gamma，队列停止 |
| 8 bits | BRECQ | 1 | default（n） | ⏸ 队列停止 | ⏸ 队列停止 | ⏸ 队列停止 |
| 8 bits | BRECQ | 7 | seed7（y） | ⏸ 队列停止 | ⏸ 队列停止 | ⏸ 队列停止 |
| 8 bits | BRECQ | 42 | seed42（y） | ⏸ 队列停止 | ⏸ 队列停止 | ⏸ 队列停止 |
| 8 bits | SBPQ | 1 | default（n） | ⏸ 队列停止 | ⏸ 准备阶段中断 | ⏸ 队列停止 |
| 8 bits | SBPQ | 7 | seed7（y） | ⏸ 队列停止 | ⏸ 队列停止 | ⏸ 队列停止 |
| 8 bits | SBPQ | 42 | seed42（y） | ⏸ 队列停止 | ⏸ 准备阶段中断 | ⏸ 队列停止 |

<a id="nspwc-ppq-gamma-sweep"></a>

### 5.1 NS-PwC 的 PPQ layerwise gamma sweep

登记日期：2026-11-19；状态更新：2026-11-21 00:14。**8-bit 首轮两个候选完成、一个中断；追加的 `1e-6`、`3e-7` 分别完成 469/533、372/533 层，仍在运行。** 2/3/4-bit 的九个首轮候选沿用 11 月 20 日“未启动，原共享等待进程已停止”的记录，本次未重新核查。用户已明确低位宽 PPQ sweep 仅做 NS-PwC（`fluids.incompressible.PiecewiseConstants`）；本节目标平均位宽为 2、3、4、8 bits。

依据用户报告的约 3.7 avg bits、gamma 暂按默认 `0.005`，设置以下首轮搜索点。该参照运行的有效配置和最终位宽仍需核对；表中 gamma 是初始搜索范围，尚未选定正式 gamma。8-bit 已完成候选的最终位宽见第 5.1.1 节。

2026-11-19 本次复核：共享目录内已完成的旧 layerwise 为 `2.237409993` bits，旧 global 为 `3.548605535` bits；均为 gamma `0.005`、seed42、init9、MC5、21 epochs。尚未定位到用户报告约 3.7 bits 的同配置参照，两个旧结果均未复用到本轮 sweep。

| 目标平均位宽 B | 首轮 gamma 候选 | 粗扫状态 | 细扫状态 | 选定 gamma | 选定运行的实际 avg bits |
| --- | --- | --- | --- | --- | --- |
| 2 bits | `2.5e-2`、`5e-2`、`1e-1` | ⏸ 三个候选未启动，队列停止 | 未启动 | 待定 | 待记录 |
| 3 bits | `7.5e-3`、`1.25e-2`、`2e-2` | ⏸ 三个候选未启动，队列停止 | 未启动 | 待定 | 待记录 |
| 4 bits | `2e-3`、`3e-3`、`5e-3` | ⏸ 三个候选未启动，队列停止 | 未启动 | 待定 | 待记录 |
| 8 bits | `3e-6`、`1e-5`、`3e-5` | 两个完成、一个中断 | 扩展 `1e-6`、`3e-7` 运行中，分别完成 469/533、372/533 层 | 待定 | 已完成候选为 7.469173、6.467118；两个新候选尚无最终位宽 |

固定设置：原模型与数据路径沿用第 3 节；训练 `seed=1`、`random_calibration=n`（default；`calibration_seed=42` 仅占位），`traj_num=16`、`final_time=10`、`time_back=0`、`batch_size=2`、`num_workers=0`、`activation_bits=32`；PPQ 使用 `init_bits=8`、`mc_samples=10`、每层 `epochs=5`、`learning_rate=9.1e-4`、`eta=1e-6`、`percentile=1e-4`、`save_every=1000`。同一个 B 内只改变 gamma；`target_bits=B` 作为该组实验的配置标签，不会自动把实际平均位宽拉到 B。

待办顺序与完成标准：

- [ ] 核对 3.7 avg bits 参照运行的真实 gamma、变体、seed、初始化、MC、epochs 及最终状态；只复用有效配置完全一致的已有运行。
- [ ] 使用训练 seed1＋固定校准 default，完成上表 12 个首轮候选的粗扫；每次记录最终 `avg_weight_bits`、模型目录、gamma、源码提交、运行成本、主机／GPU、PID、日志和状态。
- [ ] 对每个 B 找到把目标平均位宽夹住的两个正 gamma；通常较小 gamma 对应较高位宽、较大 gamma 对应较低位宽。若未夹住目标，向对应方向扩展范围；若实测顺序反常，先检查优化是否完成及配置是否一致。
- [ ] 在两点之间用几何中点 `gamma_mid = sqrt(gamma_low * gamma_high)` 细扫，按新的实测平均位宽更新区间。已有候选达到匹配标准时可直接进入选择阶段。
- [ ] 为 NS-PwC 的每个 B 选定一个 gamma。按“不超过 B bits”的预算比较时，优先匹配 `B - 0.1 <= avg_weight_bits <= B`，并保留实际位宽；未匹配的组继续定位，不能仅按目录中的配置位宽登记达标。
- [ ] 固定各 B 的选定 gamma，再按第 8 节模板运行三组正式实验：训练／校准配对 **1/default、7/seed7、42/seed42**。三组共享该 B 的 gamma，分别报告实际平均位宽及评估结果；seed 间的位宽变化如实记录。
- [ ] 将选定 gamma 回填上表及对应正式实验记录；仅在正式模型和评估完成后更新主实验格。gamma sweep 的完成状态与正式实验完成状态分别登记。

不同 gamma 必须使用独立的 tuning 目录，并显式传入 `--save_path`；如做独立评估，还需显式传入 `--output_path`。例如 G 统一使用无空格的科学计数法 `5e-2`，沿用下列目录格式：

```text
模型目录：
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_artifacts/NS-PwC/sbpq-led-v2/tuning/ppq-layerwise-traj16-ft10-bit<B>-default-gamma<G>/

评估 JSON（如执行评估）：
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_evaluation/NS-PwC/sbpq-led-v2/tuning/ppq-layerwise-traj16-ft10-bit<B>-default-gamma<G>.json

日志：
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/logs/poseidon/NS-PwC/sbpq-led-v2/tuning/ppq-layerwise-traj16-ft10-bit<B>-default-gamma<G>.log
```

定位 gamma 以模型元数据中的最终参数量加权 `avg_weight_bits` 为依据，不需要对所有候选都做完整测试集评估。若在满足位宽预算的候选间按误差择优，使用校准／验证数据；正式 gamma 确定后再做统一测试评估。正式输出仍使用第 2 节的标准短路径，tuning 产物保留来源记录。

#### 5.1.1 8-bit 首轮启动记录（2026-11-19）

本节保留最先启动的三个 8-bit gamma 的来源记录，固定训练 seed1/default；状态已按 2026-11-20 实查更新。校准为 **533 个 Linear、528 个校准 batch、1056 个样本**，完整预算为 1,407,120 次优化更新。两个候选已完成量化，第三个保留优化 checkpoint；三个原量化进程和监督进程均已不存在。扫描候选不要求完整测试集评估，不计入正式主表完成格。

| gamma | 11 月 20 日核查状态 | 原主机 / GPU | 原 tmux（已不存在） | 旧量化 PID | 旧监督 PID | 11 月 19 日启动／排队时间（Asia/Shanghai） |
| --- | --- | --- | --- | --- | --- | --- |
| `3e-6` | 调参量化完成，533/533 层；1,407,120 updates；最终 **7.469172856 bits**；11 月 20 日 02:07 完成 | dory / 0 | `poseidon-nspwc-ppq-bit8-g3e-6` | 882559 | 882521 | 15:07:05 |
| `1e-5` | 调参量化完成，533/533 层；1,407,120 updates；最终 **6.467118252 bits**；11 月 20 日 03:03 完成 | albacore / 0 | `poseidon-nspwc-ppq-bit8-g1e-5` | 1534082 | 1532884 | 15:10:01 排队，15:14:01 自动启动 |
| `3e-5` | ⏸ 中断；464/533 层（87.05%）；1,224,960 updates；无最终模型，保留 `ppq_checkpoint.pt` | shark / 0 | `poseidon-nspwc-ppq-bit8-g3e-5` | 3398846 | 3398805 | 15:07:46 |

三个主机均为 NVIDIA GeForce RTX 4070 Ti SUPER（16376 MiB）。GPU UUID 分别为 dory `GPU-1b56f0c6-7954-f002-25cc-a299dcd240b9`、albacore `GPU-fef16809-0629-84cd-b499-be1d7635a42f`、shark `GPU-d036df2f-28f4-973f-7605-ff61bdc917e1`；显式按 UUID 绑定设备。三个主机上的正式环境均通过 CUDA Adam 检查；已启动 checkpoint 记录 PyTorch `2.15.0+cu132`、matmul precision `highest`，运行时探测 CUDA build `13.2`、matmul TF32=false、cuDNN TF32=true。

源码提交：`27dca632801f9b094e29b9ea1e34fe35a6d9b941`。量化实现未修改；增加了操作层监督脚本 [scripts/run_ppq_nspwc_bit8_gamma.py](scripts/run_ppq_nspwc_bit8_gamma.py)。脚本持有每个候选的独占锁，保存完整命令、源码／脚本哈希、设备和 PID；训练仍调用现有 `PPQ.ppq` 入口。监督脚本仅在已保存历史中确认后续层也完成 epoch5 后，清理更早完成层的可重建 IO 缓存；保留校准、当前层、最近完成层及 checkpoint，清理记录写入 `*.supervisor.json`。此清理在训练进程外执行，耗时不计入 PPQ 的 `costs.json`。

albacore 在全主机扫描时空闲，但 15:07:29 启动前被其他任务占用（4455 MiB）；首个监督进程在创建实验前主动退出。15:10:01 在原 tmux 中设置 `--wait_for_idle`，每 30 秒查询一次显存，等待期间不分配 CUDA 内存。**15:14:01 GPU 空闲后已自动启动**，量化 PID 为 1534082，`*.queue.json` 已更新为 `started`；15:43 复查确认量化进程存活。共享队列只登记这三个既有运行，不重复启动或中断它们。

令 `G` 为 `3e-6`、`1e-5` 或 `3e-5`；每个候选的精确路径为：

```text
模型目录：
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/quantization_artifacts/NS-PwC/sbpq-led-v2/tuning/ppq-layerwise-traj16-ft10-bit8-default-gamma<G>/

量化日志：
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/logs/poseidon/NS-PwC/sbpq-led-v2/tuning/ppq-layerwise-traj16-ft10-bit8-default-gamma<G>.log

启动清单／运行监督／等待记录：
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/logs/poseidon/NS-PwC/sbpq-led-v2/tuning/ppq-layerwise-traj16-ft10-bit8-default-gamma<G>.launch.json
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/logs/poseidon/NS-PwC/sbpq-led-v2/tuning/ppq-layerwise-traj16-ft10-bit8-default-gamma<G>.supervisor.json
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/logs/poseidon/NS-PwC/sbpq-led-v2/tuning/ppq-layerwise-traj16-ft10-bit8-default-gamma1e-5.queue.json
```

成本与完成依据仍读取各模型目录的 `costs.json`、`run_state.json`、`metadata.json` 和 `compressed_model.pt`；最终平均位宽位于 `metadata.json` 的 `quantization.avg_weight_bits`。当前代码的 `run_state.json` 在逐层优化期间仍可能显示 `preparing`，需结合实际进程、`costs.json` 中的 optimization 计数及 checkpoint cursor 判断阶段。日志中的阶段性全模型平均位宽不能当作最终结果。

查看已有日志示例（原 tmux 已不存在）：

```bash
ssh dory
tail -n 40 /cs/research/vecg/EasyPDE/quant/poseidon-quantized/logs/poseidon/NS-PwC/sbpq-led-v2/tuning/ppq-layerwise-traj16-ft10-bit8-default-gamma3e-6.log
```

两个已完成候选均低于 `7.9 <= avg_weight_bits <= 8`，尚未选定 gamma。原共享队列逻辑要求粗扫全部完成后再选择／补点，其 worker 仍不存在；用户随后指定的两个扩展候选已独立启动，见第 5.4 节。正式 PPQ 尚无模型或评估；其他方法的正式完成情况见第 5.3 节。

### 5.2 共享等待队列（2026-11-19）

用户指令范围：**8-bit 全部正式实验；2/3/4-bit 的 SBPQ 三组 seeds；2/3/4-bit 的 PPQ sweep 仅 NS-PwC**。正式实验覆盖三个数据集，严格使用 1/default、7/seed7、42/seed42 配对；2/3/4-bit 的 PPQ、Uniform、BRECQ 正式实验未入本次队列。

本节下表保留**初始 81 项在 11 月 20 日 14:00 的核查快照**。14:42 追加启动的两个 NS-PwC bit8 候选登记在 `additional/`，使总登记数变为 83；新增运行与实时证据见第 5.4 节。

| 任务组 | 范围 | 初始任务数 | 核查状态 |
| --- | --- | --- | --- |
| 8-bit 正式实验 | 4 方法 × 3 seeds × 3 数据集 | 36 | 8 项完成、3 项中断（含 1 项已有模型待评估）、25 项未启动 |
| 2/3/4-bit SBPQ | 3 位宽 × 3 seeds × 3 数据集 | 27 | 均未启动，等待进程已停止 |
| 2/3/4-bit PPQ 粗扫 | 仅 NS-PwC；每个位宽 3 个 gamma | 9 | 均未启动，等待进程已停止 |
| 8-bit PPQ 前置粗扫 | CE-RM、Wave-Gauss 各 3 个 gamma | 6 | 均未启动，等待进程已停止 |
| 已有 8-bit PPQ 粗扫 | NS-PwC 的三个既有运行 | 3 | 两个完成、一个中断，无进程存活 |
| 合计 | 新增 78 项，登记既有 3 项 | **81** | **10 项完成、4 项中断、67 项未启动、0 项运行** |

上表为 2026-11-20 的实际核查结果。原始 `states/` 仍计数为 `completed=10, queued=58, waiting_gamma=9, running=3, external_running=1`；后两项合计 4 个标签已经过期。此次保留原始状态与认领记录，没有把历史成本文件改写为失败，也没有重新领取任务。`status` 命令只汇总这些 JSON，不能替代跨主机实际进程检查。

每项正式任务依次执行量化和独立评估，故另有 **63 条评估命令**随任务串接，不单独计入上述 81 项。评估统一采用 `safe-svd`、test、`final_time=10`、`eval_obj=ongt`、`batch_size=8`、`physics_batch_size=10`、`num_workers=0`，显式指定 v2 的输出路径；实际命令和评估协议均落盘。扫描候选只训练和记录最终位宽，不做完整测试集评估。

CE-RM 和 Wave-Gauss 的 8-bit 正式 PPQ 分别依赖自己的 gamma 选择，各先排入 `3e-6`、`1e-5`、`3e-5` 三个候选，不直接套用 NS-PwC 的 gamma。六个 gamma 组为 NS-PwC 的 bit2/3/4/8，以及 CE-RM、Wave-Gauss 的 bit8。每组等已登记候选全部完成后，优先选择 `B-0.1 <= avg_weight_bits <= B` 且最接近 B 的候选；未匹配时按 10 倍扩展或几何中点补点。每组最多增加 12 个自适应候选；候选失败、顺序异常或达到上限时写入 `needs_attention`，正式 PPQ 保持等待，不随意填 gamma。

监督程序为 [scripts/poseidon_experiment_queue.py](scripts/poseidon_experiment_queue.py)。11 月 19 日曾在以下 7 台主机、各 GPU 0 上创建独立 Bash tmux，统一会话名 `poseidon-v2-wait`；当天 15:48 七个心跳均新鲜。**11 月 20 日实查这些等待进程与 tmux 均已不存在。** 下表 PID 是历史登记值，不是当前进程；最近心跳时间均转换为 Asia/Shanghai。

| 主机 | GPU 型号 | GPU UUID | 旧等待 PID | 最后心跳（Asia/Shanghai） |
| --- | --- | --- | --- | --- |
| canada | RTX 3090 Ti，24 GB | `GPU-4e8a6d4c-bd03-cd8f-1a3f-d139f9d0c466` | 879539 | 11-20 03:55:49 |
| ruddy | RTX 3090 Ti，24 GB | `GPU-658c1d8c-2242-e044-a2ef-771b17dbad16` | 1363784 | 11-20 03:55:36 |
| goosander | RTX 3090 Ti，24 GB | `GPU-b3c57d22-7c43-7552-f9be-d239423a9730` | 1463340 | 11-20 03:56:02 |
| scaup | RTX 3090 Ti，24 GB | `GPU-81296a4a-c223-af82-4f61-3fa27e662930` | 1400430 | 11-19 22:35:40 |
| albacore | RTX 4070 Ti SUPER，16 GB | `GPU-fef16809-0629-84cd-b499-be1d7635a42f` | 1543917 | 11-20 03:55:53 |
| dory | RTX 4070 Ti SUPER，16 GB | `GPU-1b56f0c6-7954-f002-25cc-a299dcd240b9` | 887753 | 11-20 03:55:32 |
| shark | RTX 4070 Ti SUPER，16 GB | `GPU-d036df2f-28f4-973f-7605-ff61bdc917e1` | 3402318 | 11-20 03:56:00 |

等待进程存活时的工作方式：每 30 秒检查 GPU，显存已用量低于 1000 MiB 才领取任务；共享任务认领目录防止跨主机重复，GPU 锁保证本队列同卡串行。等待已有 PPQ 子进程退出后才使用相应 GPU。设置的优先顺序为 Uniform 8-bit → 已解锁的 PPQ 8-bit → SBPQ 8-bit → 8-bit PPQ 前置扫描 → BRECQ 8-bit → 低位宽 SBPQ → 低位宽 PPQ 扫描；依赖未满足的任务不阻塞其他就绪任务。该等待机制没有向其他用户预留 GPU，且没有配置开机自启；服务器重启后不会自行恢复等待。

量化实现与优化配置保持不变。队列冻结源码／配置指纹，若后续文件改变则暂停新任务并标记 `source_changed`；共享盘可用空间少于 50 GiB 时等待空间恢复。PPQ 继续清理已完成早期层的可重建 IO 缓存，保留校准、当前层、最近完成层及 checkpoint。失败任务记录错误并保留 checkpoint，队列继续其他任务。历史实测 SBPQ 峰值 reserved 为约 8.9 GB、BRECQ 约 11.7 GB，任务要求显卡总显存至少 14000 MiB。

共享队列的精确路径：

```text
/cs/research/vecg/EasyPDE/quant/poseidon-quantized/logs/poseidon/queues/sbpq-led-v2-20261119/
  plan.json             # 全部 81 项初始任务、实际参数、命令模板、源码指纹和依赖
  states/               # 逐任务排队／运行／完成／失败状态；运行时含主机、GPU、PID
  claims/               # 跨主机唯一认领记录
  workers/              # 每张 GPU 的等待进程 PID、设备占用、最近心跳
  selections/           # 每个数据集／位宽组的候选和选定 gamma
  additional/           # 补充候选；目前含用户指定的 gamma1e-6、gamma3e-7 两项
  job_records/          # 实际执行的量化／评估命令、PID、源码、日志与耗时阶段
  worker-<主机别名>.log # 等待进程日志
```

本地已保存同名目录下的 `plan.json` 和 `verified-startup.json`；本地快照不自动刷新，实时状态读服务器。各实验的模型、量化成本和评估仍按第 2、5.1 节的 v2 标准路径独立保存；评估另附 `.protocol.json` 记录实际协议。尚未启动的任务不会伪造 `compressed_model.pt`、评估 JSON 或完成格。

检查队列（在服务器仓库根目录和 `poseidon-quant` 环境执行）：

```bash
python scripts/poseidon_experiment_queue.py status \
  --queue /cs/research/vecg/EasyPDE/quant/poseidon-quantized/logs/poseidon/queues/sbpq-led-v2-20261119
# 原 poseidon-v2-wait 会话已消失，当前不能 attach。
```

验证记录：本地和服务器的 6 个 CPU 队列检查均通过，覆盖实验矩阵、并发唯一认领、数据集／位宽 gamma 依赖、候选扩展／几何中点、粗扫完成条件及量化后评估串接。全部 81 条量化命令和 63 条评估命令均通过生产入口参数解析与有效设置核验；检查期间未分配 GPU。

### 5.3 2026-11-20 14:00 历史核查：完成结果与中断位置

本轮 v2 目录已保存 **11 个量化模型、8 个正式评估 JSON**：9 个 Uniform 模型（其中 1 个待评估）和 2 个 PPQ tuning 模型。8 个评估 JSON 均可解析，含原模型／压缩模型指标及实际平均位宽。旧 default 的 D01–D07 不计入本节。

| 8-bit Uniform | 训练 1/default | 训练 7/seed7 | 训练 42/seed42 | 完成评估数 |
| --- | --- | --- | --- | --- |
| NS-PwC | ✅ 模型＋评估 | ✅ 模型＋评估 | ✅ 模型＋评估 | 3 |
| CE-RM | ✅ 模型＋评估 | ✅ 模型＋评估 | 🟨 模型完成，评估中断 | 2 |
| Wave-Gauss | ✅ 模型＋评估 | ✅ 模型＋评估 | ✅ 模型＋评估 | 3 |

上述 Uniform 最终平均位宽均为 8。评估 `base_loss` 的原模型→压缩模型为 NS-PwC `0.0017527144 → 0.0021628004`、CE-RM `0.4042667449 → 0.4046984911`、Wave-Gauss `0.0628187880 → 0.0634434968`；各数据集已完成 seeds 的这一指标相同，完整物理指标见各自 JSON。正式模型与评估路径按第 2 节组合即可定位。

| 中断任务 | 原主机／旧子进程 PID | 已保存的真实进度 | 恢复时可利用的产物 |
| --- | --- | --- | --- |
| NS-PwC／PPQ layerwise／bit8／1/default／gamma3e-5 | shark／3398846 | 完成 464/533 层；1,224,960 updates；停在 `decoder.layers.3.blocks.0.output.dense` 的 IO 缓存阶段 | `ppq_checkpoint.pt` 存在，6,391,574 bytes；CPU 实读 cursor 为 unit464、epoch1、batch0；原 checkpoint 的 Torch 为 `2.15.0+cu132` |
| CE-RM／SBPQ／bit8／1/default | albacore／1883566 | 准备／敏感度阶段；已有 528 条 FP 校准缓存，日志最后为 sensitivity batch164；尚无优化计数 | 保留 `work` 校准缓存与配置；未发现优化 checkpoint 或最终模型，不能声称可从已完成优化 step 接续 |
| CE-RM／SBPQ／bit8／42/seed42 | dory／995300 | 准备／敏感度阶段；已有 528 条 FP 校准缓存，日志最后为 sensitivity batch525；尚无优化计数 | 保留 `work` 校准缓存与配置；未发现优化 checkpoint 或最终模型 |
| CE-RM／Uniform／bit8／42/seed42 | scaup／1449485（评估） | 量化完成、实际 8 bits；评估日志有进度但未输出最终 JSON | `compressed_model.pt` 与元数据完整；若继续，只需补独立评估，无需重做量化 |

PPQ 中断 checkpoint 的 `phase=optimization`，更新数与完整层数一致；日志当时刚开始为下一层缓存 IO，不用阶段性的 avg bits 充当最终位宽。两个已完成 PPQ 候选各有完整 533 层历史、最终模型与元数据，位宽见第 5.1.1 节；六个 gamma 组均未生成正式选择结果，也没有新增自适应候选。

剩余 **67 项未启动**包括 25 项 8-bit 正式实验、27 项低位宽 SBPQ、9 项 NS-PwC 低位宽 PPQ 粗扫和 6 项 CE-RM／Wave-Gauss 8-bit PPQ 粗扫。冻结的 137 个源码／配置文件复核无变化，共享盘仍约有 1.4 TiB 可用。此次未启动、恢复或更改任何实验，也未修改原始 queue states、claims、成本文件或 checkpoint。

核查快照（本地与服务器同名目录）：`logs/poseidon/queues/sbpq-led-v2-20261119/audits/20261120-status.json`。该文件记录实际进程证据、过期队列标签、已完成任务、checkpoint cursor 和已解析的评估指标，不是运行器的任务状态文件。

### 5.4 2026-11-20：NS-PwC 8-bit PPQ 扩展两个 gamma

用户指定继续扫描 `gamma=1e-6`、`3e-7`，有空闲 GPU 则启动，否则排队等待。14:36 实查七台已部署主机均有显存占用低于 1000 MiB 的 GPU；选择与首轮相同型号的 dory、shark，各自的正式 `poseidon-quant` 环境均通过 CUDA／Adam 检查，PyTorch 为 `2.15.0+cu132`、CUDA build 为 `13.2`。137 个冻结源码／配置文件与首轮一致。

| gamma | 当前阶段（11-21 00:14 实查） | 主机／GPU | tmux | 量化 PID／监督 PID | 启动时间（Asia/Shanghai） |
| --- | --- | --- | --- | --- | --- |
| `1e-6` | 🔵 量化中，完成 469/533 层（87.99%），1,238,160 updates；正在缓存下一层 IO | dory／0，RTX 4070 Ti SUPER | `poseidon-nspwc-ppq-bit8-g1e-6` | 76819／76815 | 11-20 14:42:15 |
| `3e-7` | 🔵 量化中，完成 372/533 层（69.79%），已记录 983,000 updates；下一层优化中 | shark／0，RTX 4070 Ti SUPER | `poseidon-nspwc-ppq-bit8-g3e-7` | 41459／41455 | 11-20 14:42:24 |

11 月 21 日复核：两个量化 PID 与监督 PID 均存活，tmux 和心跳正常，日志持续进行逐层 IO 缓存／优化；均已有 `ppq_checkpoint.pt`，尚无 `compressed_model.pt` 或 `metadata.json`。日志最近保存的阶段性全模型平均位宽分别为 **8.964073614 bits（1e-6）**、**10.057904248 bits（3e-7）**，仅供观察，不能当作最终 `avg_weight_bits` 或用于确认 7.9–8 bits 达标。两项运行约 9.54 小时，137 个冻结源码／配置指纹仍一致；未修改或重启实验。核查快照：`logs/poseidon/queues/sbpq-led-v2-20261119/audits/20261121-gamma-sweep-status.json`。

已核对 `/proc/<PID>/cmdline`、`CUDA_VISIBLE_DEVICES`、GPU 上的实际 PID、tmux、模型目录的 `experiment.json` 和日志。两项均使用 seed1/default、traj16、ft10、batch2、workers0、activation32、init8、MC10、每层 5 epochs、lr9.1e-4、save_every1000；仅 gamma 不同。目录与日志沿用第 5.1 节格式，令 `G=1e-6` 或 `3e-7`。新增实验不计入正式主表完成格，仍以最终参数量加权位宽是否落在 7.9–8 bits 作为选择依据。

启动包装脚本为各候选日志目录下的 `*.start.sh`，先创建独立 Bash tmux、激活环境，再持有原队列 `gpu_locks/<UUID>.lock` 调用监督脚本；候选本身另有独占锁。统一使用 `--wait_for_idle`，如果启动前显存占用达到 1000 MiB，则每 30 秒检查一次，等待时不创建 CUDA 上下文，空闲后自动启动。本次两项均直接启动，无需等待。该机制不包含开机自启。

监督脚本只增加了两个允许的 gamma 参数，量化实现没有改动；继续保留 checkpoint，并清理更早完成层的可重建 IO 缓存。两个候选以 `external=true` 登记到原队列的 `additional/`，由各自监督进程运行；原七个共享 worker 和旧中断任务保持原状态。新增候选的 `.launch.json`、`.queue.json`、`.supervisor.json`、`.waiter.log` 均与量化日志同目录。

登记与启动核验记录：`logs/poseidon/queues/sbpq-led-v2-20261119/audits/20261120-gamma-extension-registration.json`、`20261120-gamma-extension-startup.json`。实际完整命令、源码与脚本指纹、主机、GPU UUID 和 PID 均已落盘。

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

**当前运行：NS-PwC／PPQ layerwise／bit8／seed1/default／gamma1e-6（dory GPU0，PID76819）与 gamma3e-7（shark GPU0，PID41459），分别由独立 tmux 中的监督进程管理。** 11 月 21 日 00:14 实查这两个主机的进程、心跳、有效配置、逐层进度与日志，分别完成 469/533、372/533 层，尚无最终模型或最终平均位宽，完整登记见第 5.4 节。本次未重新核查其他主机或原共享 worker；此前七台主机的检查时间和历史状态见下文。

此前 2026-11-20 13:51–14:00（Asia/Shanghai）重新连接 `albacore`、`canada`、`dory`、`goosander`、`ruddy`、`scaup`、`shark` 七台已部署队列的服务器。**当时各主机当前用户的 Python／tmux 进程均为空，实际运行数和存活等待进程数均为 0。** 以下重启时间及显存表是该次历史快照。共享文件的过期 running 标签不代表当前仍在计算。

七台主机的 `/proc/stat` 启动时间均在今天早上 08:01–08:13；原 tmux 和等待进程不会跨服务器重启保留。多数等待器最后心跳在 03:55–03:56，scaup 更早为昨晚 22:35；停止心跳早于当前系统启动时间，仅凭这些记录不能判定最初停止的具体原因。此次查看的日志末尾未见明确的异常退出 traceback，未将中断推断为 OOM 或算法报错。

| 主机 | 系统启动时间（Asia/Shanghai） | 13:59 实测 GPU0 已用显存 | 13:59 本项目进程／tmux |
| --- | --- | --- | --- |
| canada | 11-20 08:03:00 | 44 MiB | 无／无 |
| ruddy | 11-20 08:13:31 | 9438 MiB | 无／无 |
| goosander | 11-20 08:04:23 | 4625 MiB | 无／无 |
| scaup | 11-20 08:03:23 | 4788 MiB | 无／无 |
| albacore | 11-20 08:01:40 | 8866 MiB | 无／无 |
| dory | 11-20 08:11:36 | 173 MiB | 无／无 |
| shark | 11-20 08:12:25 | 170 MiB | 无／无 |

上述显存数据仅来自当时的 `nvidia-smi`，占用会变化；14:00 那次核查未执行 CUDA 小算子检查或分配 GPU。14:42 的新启动已另在 dory、shark 上执行正式环境 CUDA／Adam 检查并绑定 GPU，见第 5.4 节。其他用户的占用不计入本实验运行状态。

11 月 19 日早前的 `check_ucl_gpus` 曾尝试配置的 54 台主机，只有上述七台可连接；其余 47 台因 SSH、超时或 host key 问题未完成核查。本次只复查已部署队列的七台，其他主机仍为 **❔ 待核实**，不能宣称已检查全 UCL 的当前空闲资源。

已完成和中断任务的详细索引见第 5.1.1、5.3 节。历史结果还可补做 **D02 的独立评估**；这不会使其成为 v2 的完成记录。后续如恢复任务，应先检查共享认领记录和 checkpoint，避免把“评估中断但模型已完成”的 Uniform 重做量化。

复查进程与 tmux：

```bash
ps -u "$(id -u)" -o pid,etime,args
tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} pid=#{pane_pid} command=#{pane_current_command} cwd=#{pane_current_path}'
```

后续实际启动实验时，在本节登记实验键、阶段、主机、GPU、tmux、PID、日志绝对路径和检查时间，再更新对应主实验格。成本文件包含历史启动会话，早期 interrupted 记录不代表最后一次已完成运行仍然中断。

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

以下接第 8 节中**已完成量化的那一种方法**，沿用 `POSEIDON_RUN_DIR`、`POSEIDON_RUN_NAME`、`POSEIDON_DATASET` 等变量。第 5.2 节本次队列已明确统一采用以下 `safe-svd`、test、评估终点 10 和 batch 设置，并保存实际命令。该选择不是根据量化文件名推断旧评估范围，也不改变历史评估的真实参数。

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
