# Poseidon 实验进度 · Isambard

2026-11-23 11:28（北京时间）已向 Slurm 提交 NS-PwC PPQ 粗扫、PPQ 补点、SBPQ batch 64 和批量评估。设置与评估协议见 [PoseidonExpSetting.md](PoseidonExpSetting.md)。

11:30 复查确认首批 6 个 PPQ 候选已进入逐层缓存与优化，SBPQ batch 64 已进入敏感度计算。其余 18 个粗扫候选受数组并发限制排队，补点和评估等待依赖。

## 已提交作业

| 作业 | Slurm ID | 提交时状态 | 启动脚本 | 日志 |
| --- | --- | --- | --- | --- |
| PPQ gamma 粗扫 | 8745392 | 排队，24 个候选，最多 6 个并行 | `launch_scripts/poseidon/PPQ_NSPwC_sweep.sh` | `logs/PPQ-NS-PwC-grid-8745392_<task>.out` |
| SBPQ batch 64 | 8745393 | 排队 | `launch_scripts/poseidon/SBPQ_NSPwC_batch64.sh` | `logs/SBPQ-NS-PwC-b64-8745393.out` |
| PPQ 目标位宽补点 | 8745394 | 排队，等粗扫数组全部结束 | `launch_scripts/poseidon/PPQ_NSPwC_refine.sh` | `logs/PPQ-NS-PwC-refine-8745394_<task>.out` |
| SAFE-SVD 批量评估 | 8745395 | 排队，等补点数组和 SBPQ batch 64 都结束 | `launch_scripts/poseidon/EVAL_NSPwC_batch.sh` | `logs/EVAL-NS-PwC-batch-8745395.out` |

补点与评估使用 `afterany` 依赖，可以利用成功完成的候选，即使同组有任务失败。提交回执在 `logs/poseidon-isambard-nspwc-20261123-jobs.tsv`。查状态可执行 `squeue -j 8745392,8745393,8745394,8745395`，已离队作业用 `sacct -j 8745392,8745393,8745394,8745395`。

## PPQ 目标位宽

各候选保存于 `results/NS-PwC/ppq-layerwise-traj16-ft10-b32-sweep/gamma-<值>/`。原 `gamma=1e-6` 的 5.889101705 bits 数据点参与选参。状态以 `run_state.json`、最终模型和 `metadata.json` 为准。完整响应曲线可随时用 `python scripts/isambard_nspwc_ppq_sweep.py report` 查看。

| 目标 bits | 达标区间 | 选定 gamma | 最终位宽 |
| ---: | --- | --- | --- |
| 2 | [1.9, 2.0] | 待粗扫和补点 | 待记录 |
| 3 | [2.9, 3.0] | 待粗扫和补点 | 待记录 |
| 4 | [3.9, 4.0] | 待粗扫和补点 | 待记录 |
| 8 | [7.9, 8.0] | 待粗扫和补点 | 待记录 |

## SBPQ batch 对比

| 实验 | Batch／步数 | 优化样本数 | 量化总耗时 | 峰值显存（分配／预留） | 实际平均位宽 |
| --- | --- | ---: | --- | --- | ---: |
| `sbpq-traj16-ft10-steps200` | 32／200 | 6,400 | 49 分 08 秒 | 24.94／25.94 GiB | 7.840136 |
| `sbpq-traj16-ft10-b64-steps103` | 64／103 | 6,400（计划） | 待记录 | 待记录 | 待记录 |

## 历史模型与统一评估

历史结果保留在各自实验目录。PPQ、BRECQ 的旧评估与 SBPQ 的旧评估使用了不同的 FP 基线数值，正式对比采用本轮的统一评估文件。

| 目录（相对于 `results/NS-PwC/`） | 已保存的平均位宽 | 历史评估 |
| --- | ---: | --- |
| `ppq-traj16-ft10` | 5.889102 | `result.json` |
| `brecq-traj16-ft10` | 8.000000 | `result.json` |
| `sbpq-traj16-ft10` | 8.204547 | `test.json` |
| `sbpq-traj16-ft10-default` | 7.844169 | `test.json` |
| `sbpq-traj16-ft10-steps200` | 7.840136 | `test.json` |

统一评估将原模型指标和已完成模型的评估写入 `results/NS-PwC/result.json`。同一协议已登记的模型及原模型指标可复用，修改产物或评估条件会触发相应重评。评估脚本同时写出 PPQ 的 `sweep-summary.json`。
