# Isambard Agent 指南

用 [连接说明](IsambardConnection.md) 的双跳命令从本机进入项目。路径依次是 `ssh canada`、`cd MyHome`、`ssh u6abp.aip2.isambard`、`cd cjhome/poseidon-quantized`。以 `hostname` 和 `pwd -P` 核对当前主机与仓库。

## 实验资料

- [实验进度](PoseidonExperiments.md) 记录 Isambard 作业 ID、依赖、日志和产物路径。
- [实验设置](PoseidonExpSetting.md) 记录训练配方、PPQ 目标位宽和统一评估协议。
- 仓库根目录的 `PoseidonExperiments.md` 和 `PoseidonExpSetting.md` 记录 UCL 实验。Isambard 使用本目录的两份文档。
- GitHub 的 `ritchie` 分支同步源码、脚本和文档。模型、数据、输出、Slurm 日志及 MPP 独立工作树保存在服务器本地。

## 核对作业

在 Isambard 项目根目录查当前用户的队列。对已经离队的作业使用 `sacct`，并核对最终产物。

```bash
squeue -u "$(id -un)" -o '%i %j %t %M %l %R'
cat logs/poseidon-isambard-nspwc-20261123-jobs.tsv
sacct -j 8745392,8745393,8745394,8745395 \
  --format=JobID,JobName,State,Elapsed,ExitCode
```

实际 ID 从最新进度文档或提交回执读取。对数组按元素统计完成、运行、等待、失败。 `PD (Dependency)` 等上游作业结束，`PD (JobArrayTaskLimit)` 等并发名额。只查状态时使用只读命令。继续训练或重新提交前先核对原任务、checkpoint 和输出目录。

## 核对成果

| 目标 | 实际依据 |
| --- | --- |
| PPQ 扫描 | `results/NS-PwC/ppq-layerwise-traj16-ft10-b32-sweep/gamma-*/`。运行中的层数读 `history.json` 中完成最后一轮的层，最终位宽读 `metadata.json` 的 `quantization.avg_weight_bits` |
| PPQ 选参 | `python3 scripts/isambard_nspwc_ppq_sweep.py report` 汇总已完成点和各目标匹配。 `sweep-summary.json` 是批量评估启动时保存的快照 |
| SBPQ batch 64 | `results/NS-PwC/sbpq-traj16-ft10-b64-steps103/`。用 `costs.json` 的累计计数和 Slurm 日志核对敏感度、优化步数及最终成本 |
| 统一评估 | `results/NS-PwC/result.json` 的 `protocols` 保存评估参数、共同原模型及各压缩模型指标。不同协议的数值分别报告 |

量化完成需检查 `run_state.json` 为 `completed`，并确认 `compressed_model.pt` 和 `metadata.json` 存在。正在优化时，日志里的平均位宽仅是阶段值。耗时用 `costs.json` 的 `cumulative.wall_seconds`，显存用 `gpu_peak_allocated_bytes` 和 `gpu_peak_reserved_bytes`，按 `2**30` 换算 GiB。对续跑或失败的任务注明累计耗时与各会话状态。

报告时写清核查时间、每组的完成数和当前阶段。已有最终结果时，附上 gamma、实际位宽、模型路径和评估协议。更新 Isambard 的 [实验进度](PoseidonExperiments.md) 时保留历史快照和原始产物。
