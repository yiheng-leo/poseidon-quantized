# SBPQ / BRECQ / PPQ：实验目录、资源成本与断点

适用于 ritchie 的正式入口 `SBPQ.sbpq`、`BRECQ.brecq` 和 `PPQ.ppq`（global/layerwise）。
量化目标、校准轨迹选择和最终独立评估协议保持不变。

## 自动目录命名

这三个入口的 `--save_path` 现在表示**父目录**。例如：

```bash
python -u -m BRECQ.brecq --config configs/brecq/default.json \
  --model_path /path/to/checkpoint --data_path /path/to/down_streams \
  --traj_num 64 --batch_size 2 --seed 42 --DEV cuda \
  --save_path quantization_artifacts/NS-PwC
```

输出为下面形式的自动命名子目录，精确路径会在启动时打印：

```text
quantization_artifacts/NS-PwC/
  brecq_w8a32_recbs32_iter20000_lr0p001_traj64_fix_cs42_ft20_tb0_bs2_s42_cfg-<20位指纹>/
```

SBPQ 名称包括初始/目标 bits、kappa、delta、MC、eta、lr、prior scale、Sobolev、优化步数等；
PPQ 名称包括初始/目标 bits、epochs、MC、eta、lr 等。所有名称都包括轨迹选择、时间、batch 和 seed。
名称过长时截短可读部分，配置指纹始终保留。

`experiment.json` 保存完整有效 CLI 配置与完整 SHA256。指纹覆盖整个有效配置，包括没有写入可读名称的参数，
模型/数据路径规范化为绝对路径；JSON 顺序、配置文件位置和输出父目录不影响指纹。
`save_every`、`profile_interval`、resume、overwrite、keep_work_dir 是操作选项，不改变实验编号，
每次启动的操作选项会写入成本报告。目录已有不同配置会拒绝混用；已有相同实验须显式 resume 或 overwrite。
`--output_root` 也可指定父目录。原来的其他方法入口保留旧 save_path 语义。

## 断点恢复

新实验可直接指定实际实验目录，模型、数据及超参数会从 experiment.json 恢复：

```bash
python -u -m BRECQ.brecq --resume /path/to/actual-brecq-experiment
python -u -m PPQ.ppq --resume /path/to/actual-ppq-experiment
python -u -m SBPQ.sbpq --resume /path/to/actual-sbpq-experiment
```

也可传实验目录中的主 checkpoint 文件。恢复继续使用原实验目录，不再次追加名字。
为保证名称和训练定义一致，新实验 resume 时不允许更改训练超参数（包括 batch、epochs/total_steps 和迭代预算）；
可以更改保存间隔与采样间隔，例如 `--save_every 1000 --profile_interval 0.2`。
需要不同预算或其他训练配置时，作为新的实验启动。

- SBPQ：保留步长、AdamW、完成步数、随机数状态；校验 FP、校准、clean outputs、ranges、Beta 参数和层顺序。
- PPQ：保留全部步长、当前 unit/epoch/batch、Adam、epoch 内未完成的 loss 累计量、history 和 RNG。
  layerwise 恢复跳过已完成层，并恢复当前层内部进度。
- BRECQ：每完成一层 MSE scales 初始化就保存进度；完成块的 quantizer 状态单独分片保存；
  主 checkpoint 保存当前块 alpha、Adam、iteration、温度/损失计数与 RNG。
  恢复跳过已完成 scales/块，继续当前块，不重做整个一小时初始化。

BRECQ 默认配置每 1000 次重构更新保存当前块，PPQ/SBPQ 默认每 100 次；`--save_every` 可覆盖。
PPQ/BRECQ 还会校验 PyTorch 版本、数值精度与方法源码指纹，避免不同实现静默拼接。
checkpoint 采用临时文件 + fsync + 原子替换；被打断的更新从最后一次完整保存的状态重放。
完成块分片通过 SHA256 验证，主 checkpoint 引用这些分片；请保留整个实验目录，不能只复制一个 BRECQ 主文件。

`work/` 下的校准/FP 输出及当前块 IO 缓存使用输入与内容校验，恢复时复用，缺失可验证地重建。
异常退出后保留，成功导出后默认清理；checkpoint 与已完成块分片仍保留。
若在准备阶段尚未产生主 checkpoint，新版实验目录的清单也支持继续准备。

旧 BRECQ/PPQ 运行没有保存这些状态，不能从旧 work-* 推出 optimizer 或已优化权重，也不会因更新代码而自动具有断点。
历史 SBPQ checkpoint 仍可按原目录恢复；缺少 experiment.json 时需要原命令参数或原 metadata.json。
新增的训练配置必须保持与旧断点一致。历史 SBPQ 的旧成本无法倒推，不会填造历史优化时间。

## 成本记录

每个实验独立生成 `costs.json`，运行时每 5 秒和每次 checkpoint 刷新，包含每次启动与累计统计。
完成后的 metadata.json 同时包含 costs_summary。命令行查看：

```bash
python -m quantization.costs /path/to/actual-experiment
python -m quantization.costs /path/to/actual-experiment --json
```

分阶段记录 calibration_loading、model_loading、identity_verification、calibration/FP cache、
range/scale 初始化、sensitivity、optimizer_setup、optimization、checkpoint_saving、materialization、artifact_saving、cache_cleanup 等。
实际阶段随方法不同而不同；纯 optimization 时间包含每步加载 batch、前后向和 optimizer 更新，
**排除准备和 checkpoint 写入**，嵌套阶段使用独占时间，不重复累计。

| 字段 | 定义 |
|---|---|
| wall_seconds | 阶段墙钟秒数；CUDA 阶段边界同步 |
| cpu_seconds | 采样累计的当前进程与可访问子进程 user+system CPU 秒数 |
| cpu_peak_rss_bytes | 当前进程和活跃子进程 RSS 之和的采样峰值 |
| cpu_peak_utilization_percent | CPU 时间/墙钟时间的采样峰值，单核 100%，多核可超过 100% |
| cpu_process_lifetime_peak_rss_bytes | OS 报告的父进程整个生命周期 RSS 峰值，排除子进程，不代表单阶段峰值 |
| gpu_peak_allocated_bytes | 该阶段 PyTorch allocator 中张量分配的显存峰值 |
| gpu_peak_reserved_bytes | 该阶段 PyTorch allocator 保留的显存峰值 |
| counters.optimization_updates / samples | 实际执行的更新数及样本处理数，包括发生过的重放工作 |
| counters.checkpoint_bytes_written | 各次 checkpoint 实际写入字节数之和 |
| counters.cache_bytes_written / artifact_bytes_written | 实际缓存/最终产物写入量 |

内存字段原始单位为 bytes，查看命令转换为 GiB。CPU 模式 GPU 字段为 null。
GPU 指标包含模型等常驻分配，不包含 CUDA context 等 allocator 外开销，不等同于 nvidia-smi 整卡使用量。
CPU RSS 是采样值，可能错过很短的峰值，共享页求和可能重复计数；采样间隔默认 0.1 秒。
权限限制时退化为可访问进程统计，并明确标记 cpu_scope_complete=false，不静默声称完整进程树。
CUDA 设备、CPU 核数、主机、版本等硬件信息和每次启动的采样/保存间隔一并保存。

累计时间为各次**实际记录**的耗时之和，峰值取最大值；不把峰值相加。
恢复时重新准备或重放的成本也是真实成本，按对应阶段累计。
SIGKILL/主机故障只能保留最后一次刷新之前的记录，最后几秒未知；没有记录的历史不伪装成 0 秒。
资源记录不会回填旧实验的纯优化时间。评估 JSON 仍只输出平均 bits、原指标和百分比变化。
