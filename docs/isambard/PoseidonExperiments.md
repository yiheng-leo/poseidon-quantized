# Poseidon 实验进度 · Isambard

2026-11-23 11:28（北京时间）已向 Slurm 提交 NS-PwC PPQ 粗扫、PPQ 补点、SBPQ batch 64 和批量评估。设置与评估协议见 [PoseidonExpSetting.md](PoseidonExpSetting.md)。

11:30 复查确认首批 6 个 PPQ 候选已进入逐层缓存与优化，SBPQ batch 64 已进入敏感度计算。其余 18 个粗扫候选受数组并发限制排队，补点和评估等待依赖。

2026-11-23 14:27（北京时间）复查，PPQ 粗扫 24 项中已完成 12 项，另有 6 项运行、6 项排队，运行中的候选完成 299–367／533 层。已实测 `gamma=1e-7` 得到 8.176438 bits、`1.5e-7` 得到 7.609863 bits，夹住 8-bit 目标区间，当前四个目标均未选定。SBPQ batch 64 已完成并保存模型。补点和批量评估仍在等待依赖，尚无统一评估结果。

2026-11-23 23:57（北京时间）核验，24 个 PPQ 粗扫和四个目标的补点作业均退出成功，共得到 32 个新候选和 1 个历史锚点。3、4、8 bits 已达标，2 bits 在计划的四次补点后仍未命中。统一评估已完成，`results/NS-PwC/result.json` 保存了同一 FP 基线和 9 个压缩模型的结果。

## 已提交作业

| 作业 | Slurm ID | 最终状态 | 启动脚本 | 日志 |
| --- | --- | --- | --- | --- |
| PPQ gamma 粗扫 | 8745392 | 24/24 完成 | `launch_scripts/poseidon/PPQ_NSPwC_sweep.sh` | `logs/PPQ-NS-PwC-grid-8745392_<task>.out` |
| SBPQ batch 64 | 8745393 | 已完成，保存模型和元数据 | `launch_scripts/poseidon/SBPQ_NSPwC_batch64.sh` | `logs/SBPQ-NS-PwC-b64-8745393.out` |
| PPQ 目标位宽补点 | 8745394 | 4/4 作业结束，3 个目标达标 | `launch_scripts/poseidon/PPQ_NSPwC_refine.sh` | `logs/PPQ-NS-PwC-refine-8745394_<task>.out` |
| SAFE-SVD 批量评估 | 8745395 | 已完成，51 分 53 秒，FP 基线与 9 个压缩模型 | `launch_scripts/poseidon/EVAL_NSPwC_batch.sh` | `logs/EVAL-NS-PwC-batch-8745395.out` |

补点与评估使用 `afterany` 依赖，可以利用成功完成的候选，即使同组有任务失败。提交回执在 `logs/poseidon-isambard-nspwc-20261123-jobs.tsv`。查状态可执行 `squeue -j 8745392,8745393,8745394,8745395`，已离队作业用 `sacct -j 8745392,8745393,8745394,8745395`。

## PPQ 目标位宽

各候选保存于 `results/NS-PwC/ppq-layerwise-traj16-ft10-b32-sweep/gamma-<值>/`。原 `gamma=1e-6` 的 5.889101705 bits 数据点参与选参。状态以 `run_state.json`、最终模型和 `metadata.json` 为准。完整响应曲线可随时用 `python scripts/isambard_nspwc_ppq_sweep.py report` 查看。

| 目标 bits | 达标区间 | 选定 gamma | 实际平均位宽 | 统一测试 `base_loss` |
| ---: | --- | --- | ---: | ---: |
| 2 | [1.9, 2.0] | 未匹配 | 最低 2.397478413（gamma `100`） | — |
| 3 | [2.9, 3.0] | `2.3656939e-3` | 2.935657507 | 0.0238910280 |
| 4 | [3.9, 4.0] | `3.3782515e-4` | 3.949470538 | 0.0128705883 |
| 8 | [7.9, 8.0] | `1.1759174e-7` | 7.954565285 | 0.0037704324 |

2-bit 补点的 gamma 依次为 `0.1`、`1`、`10`、`100`，实际平均位宽为 2.413217、2.399026、2.397619、2.397478 bits。本轮观测表明增大 gamma 的收益在约 2.3975 bits 附近趋缓。选中的三个模型已纳入统一评估，PPQ 扫描目录内的 `sweep-summary.json` 记录了选参结果。

## SBPQ batch 对比

| 实验 | Batch／步数 | 优化样本数 | 量化总耗时 | 峰值显存（分配／预留） | 实际平均位宽 |
| --- | --- | ---: | --- | --- | ---: |
| `sbpq-traj16-ft10-steps200` | 32／200 | 6,400 | 49 分 08 秒 | 24.94／25.94 GiB | 7.840136 |
| `sbpq-traj16-ft10-b64-steps103` | 64／103 | 6,400 | 26 分 28 秒 | 46.49／47.87 GiB | 8.122477 |

两组共用同一个 SAFE-SVD `ongt` 测试协议，`final_time=20`，55,440 个样本。推理 batch 为 256，物理指标 batch 为 10，共同 FP `base_loss=0.002329166513`。评估文件为 `results/NS-PwC/result.json`，profile ID 为 `ebea9b16915799168380b7c0d5d519ec637c9e79d9b887c840981d4b1be31ed0`。

| 测试指标 | steps200（batch 32） | batch 64／steps103 | batch 64 相对 steps200 |
| --- | ---: | ---: | ---: |
| `base_loss`（括号为相对 FP 的增幅） | 0.0023845383（+2.3773%） | 0.0024093920（+3.4444%） | +1.04% |
| `rel_continuity` | 0.008381661 | 0.008344403 | −0.44% |
| `rel_h1_error` | 0.020055045 | 0.020034024 | −0.10% |
| `rel_sobolev_012` | 0.065153214 | 0.064777222 | −0.58% |
| `rel_sobolev_0` | 0.001850165 | 0.001867257 | +0.92% |

batch 64 的量化全程少 22 分 40 秒，峰值显存分配从 24.94 升至 46.49 GiB。`base_loss` 更高，部分物理指标略低。实际平均位宽从 7.8401 升至 8.1225 bits，超过 8-bit 上限。

## 历史模型与统一评估

历史结果保留在各自实验目录。PPQ、BRECQ 的旧评估与 SBPQ 的旧评估使用了不同的 FP 基线数值，正式对比采用本轮的统一评估文件。

| 目录（相对于 `results/NS-PwC/`） | 已保存的平均位宽 | 历史评估 |
| --- | ---: | --- |
| `ppq-traj16-ft10` | 5.889102 | `result.json` |
| `brecq-traj16-ft10` | 8.000000 | `result.json` |
| `sbpq-traj16-ft10` | 8.204547 | `test.json` |
| `sbpq-traj16-ft10-default` | 7.844169 | `test.json` |
| `sbpq-traj16-ft10-steps200` | 7.840136 | `test.json` |

统一评估的原模型指标和 9 个压缩模型结果已写入 `results/NS-PwC/result.json`。同一协议已登记的模型及原模型指标可复用，修改产物或评估条件会触发相应重评。
