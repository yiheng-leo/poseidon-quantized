# Poseidon Quantization — ritchie

Poseidon 的 SBPQ 与各个 baseline 使用独立入口，共用完整轨迹校准、模型导出和独立评估接口。
SBPQ 的默认优化配置来自本次 NSPwC Table 3 复现实验；校准采样改为旧 SAPQ 仓库 ritchie 的完整轨迹协议。

| 方法 | 量化入口 |
|---|---|
| SBPQ | `python -m SBPQ.sbpq`（global） |
| UniformQ | `python -m UniformQ.uniformq` |
| DynamicQ | `python -m DynamicQ.dynamicq` |
| StaticQ | `python -m StaticQ.staticq` |
| PPQ | `python -m PPQ.ppq --variant global` / `layerwise` |
| SAPQ | `python -m SAPQ.sapq --variant global` / `network` / `block` / `layerwise` |
| BRECQ | `python -m BRECQ.brecq` |
| RepQ | `python -m RepQ.repq` |
| PTQ4ViT | `python -m PTQ4ViT.ptq4vit` |

在仓库根目录执行；显式命令行参数覆盖 JSON 配置。新入口使用 CLI/JSON 设置 SBPQ，不读取旧的 `SBPQ_*` 实验环境变量。

```bash
python -m pip install -r setting_environment/quantization_requirements.txt

python -u -m SBPQ.sbpq --config configs/sbpq/default.json \
  --model_path /path/to/poseidon/fluids.incompressible.PiecewiseConstants \
  --dataset fluids.incompressible.PiecewiseConstants \
  --data_path /path/to/down_streams --DEV cuda \
  --traj_num 64 --random_calibration n --calibration_seed 42 \
  --final_time 20 --time_back 0 --batch_size 2 \
  --save_path quantization_artifacts/NS-PwC/sbpq-traj64

python -m evaluation.evaluate \
  --model_path quantization_artifacts/NS-PwC/sbpq-traj64 \
  --checkpoint /path/to/poseidon/fluids.incompressible.PiecewiseConstants \
  --dataset fluids.incompressible.PiecewiseConstants \
  --data_path /path/to/down_streams --DEV cuda \
  --final_time 20 --batch_size 8 --physics_batch_size 10 \
  --output_path quantization_evaluation/NS-PwC/sbpq-traj64.json
```

SBPQ 默认：init/reference bits=8、bounds=1..16、κ=150、Δbits=2、prior scale=1、MC=10、eta=1e-6、
AdamW lr=3.75e-5、weight decay=0、无梯度裁剪、800 个累计优化步；敏感度使用 task MSE + 二阶 Sobolev L1，
各阶权重均为 1，weight range 使用经验分位数 p=1e-4。目标 bits 是先验中心，不是硬性平均位宽约束。

默认选取训练集前 64 条完整轨迹，所有时间对参与校准快照与敏感度计算；`--random_calibration y`
改为由 calibration seed 决定的无放回轨迹选择。不会再默认截取 512 个打乱的 minibatch。
Wave-Gaussians 使用 `--final_time 15`。`--batch_size` 同时影响敏感度的 batch 均值，应作为实验设置记录。

**800 步是优化预算，不是完整校准遍历次数。** 优化沿校准 batch 顺序循环；若总 batch 数超过 800，
likelihood 只使用前 800 个 batch，敏感度仍使用全部选中样本。日志和 metadata 会报告覆盖量；
需要完整优化遍历时，将 `--total_steps` 设为校准 batch 数或其倍数。

每次压缩保存 `compressed_model.pt`、`metadata.json`、`quantization_state.pt`；SBPQ 还保存
`sbpq_checkpoint.pt`、`sbpq_trainer_state.pt`、`history.json` 和准备缓存。
最终权重按步长一次性 round，推理不需要量化 hooks、MC 或旧缓存。这些是浮点 QDQ 产物，未做整数打包。

恢复时保留原命令的模型、数据、采样和优化设置，追加：

```bash
# 在上面的量化命令中追加；total_steps 是包含已有进度的累计目标。
--resume quantization_artifacts/NS-PwC/sbpq-traj64 --total_steps 800
```

默认每 100 步原子保存。新校准集不复用作者旧的 basename 缓存；准备缓存验证模型、数据内容、层集合、
归一化与损失实现。改变这些设置必须使用新的输出目录。恢复会校验校准数据及优化状态。

量化入口不加载验证集或测试集。独立评估沿用旧 ritchie 与 SAFE-SVD 对齐的测试集全时间对协议，
JSON 仅输出平均 bits、指标原值与相对变化百分比；这不是论文历史的 256 样本验证口径。

作者入口 `python -u run_sbpq_poseidon.py` 仍保留原校准和验证流程，默认优化参数已更新为上述配置，
继续支持旧 `SBPQ_*` 环境变量及旧 checkpoint。需要复查历史 Table 3 时使用该入口；
新 `SBPQ.sbpq` 的校准数据已改变，不承诺逐值复现历史结果。

- [迁移来源、校准与方法范围](docs/ritchie_migration.md)
- [代码目录](docs/layout.md)
- [SBPQ 方法](SBPQ/methodology.md)
- [独立评估协议](docs/poseidon_evaluation.md)
- [公共量化协议及旧 SAPQ origin](quantization/README.md)

BRECQ、PPQ、SAPQ、SBPQ 启动时会先用一个小张量检查对应 Adam/AdamW 优化器，
在模型加载、校准缓存和 scale 初始化之前暴露内部 PyTorch 导入或 backend 错误。
这项检查不改变量化算法或随机种子。也可单独在新进程检查环境：

```bash
python -m quantization.runtime --DEV cuda
# SBPQ 对应的优化器：
python -m quantization.runtime --DEV cuda --optimizer AdamW
```

如果新进程通过、旧长任务曾报内部缺失符号，要检查任务运行期间共享 Python 环境是否被更新；
若新进程也失败，应先修复环境。此检查不会重装 PyTorch，也不会让旧 BRECQ 工作目录变成可恢复 checkpoint。

验证：`python -m pytest`，只收集 `tests/` 和 `SBPQ/tests/`。
