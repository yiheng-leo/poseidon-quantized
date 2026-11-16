# SBPQ on SafeSVD VICON

在本仓库根目录运行。模型、数据和校准采样使用 [SafeSVD](https://github.com/realcrane/SafeSVD)
仓库内的 VICON 实验版本；`--safe_svd_root` 指向同时含有 `VICON/` 和 `safe_svd/` 的完整仓库。
不需要重新下载已准备好的 checkpoint 或 HDF5 数据。

```bash
python -m pip install -r setting_environment/vicon_requirements.txt

python -u -m SBPQ.sbpq \
  --config configs/sbpq/vicon.json \
  --safe_svd_root /path/to/SafeSVD \
  --ckpt_path /path/to/SafeSVD/models/VICON/model.pth \
  --data_root /path/to/SafeSVD/data/processed_data/2D/CFD/converted \
  --config_override datasets.types.NS2D.folder=/path/to/SafeSVD/data/raw_data/NavierStokes-2D-conditoned \
  --calibration_num 256 --batch_size 64 \
  --random_calibration n --calibration_seed 42 \
  --DEV cuda \
  --save_path quantization_artifacts/VICON/sbpq-cali256
```

`vicon.json` 已设置 `model_type=vicon`；也可以使用 `--model vicon` 显式选择。
`--ckpt_path` / `--data_root` 是 `--model_path` / `--data_path` 的别名。
`--safe_svd_root` 也可由 `SAFE_SVD_ROOT` 环境变量提供。

请将 `--data_root` 设为 **SafeSVD 对比实验实际使用的同一个目录**。
如果 SafeSVD 使用整理后的 `data/VICON`，这里也使用该目录。
目录默认包含 `2D_Train_Rand/`、`2D_Train_Turb/` 和 `NavierStokes-2D-conditoned/`；
NS2D 放在别处时可单独覆盖：

```bash
--config_override datasets.types.NS2D.folder=/absolute/path/to/NavierStokes-2D-conditoned
```

## 校准对齐

直接调用 SafeSVD 的 `build_calibration_loaders` 和 `apply_vicon_file_order`，
加载 COMPRESSIBLE2D、EULER2D、NS2D 三个类别的 **train** 数据。
时间对、类别顺序、channel mask、bc 归一化、`loss.min_ex` 与 VICON 保持一致。
不创建验证集或测试集。

`calibration_num` 是**每个类别**的样本上限；与 SafeSVD 一样，每类最多使用
`floor(calibration_num / batch_size)` 个 batch。256/64 时，每类 4 个 batch，
三个类别合计 768 个样本、12 个 batch。`traj_num`、`final_time`、`time_back` 属于 Poseidon，
不控制 VICON 校准。若 SafeSVD 实验使用 2048，请同时改为 `--calibration_num 2048`。
配置中的 256 对应 SafeSVD README 的 VICON 示例，而 SafeSVD CLI 的默认值为 2048。

对比实验保持 checkpoint、数据内容/划分、Hydra 配置、calibration seed、batch size 和 worker 数相同。
默认使用 SafeSVD 自带的 `VICON/configs/research_calibration_order.json`，文件集合不匹配会报错。
自定义数据划分可传 `--vicon_file_order /path/to/order.json`；
`--vicon_file_order filesystem` 对应 SafeSVD 的同名选项，应在双方一致使用。
文件顺序清单只校验路径集合与顺序，不能确认不同目录里的 HDF5 内容相同。

校准快照写入 CPU 磁盘缓存；metadata 记录实际各类样本数、时间对与 batch 内容指纹。
恢复时重新核对输入内容、模型权重、源码和敏感度设置，数据变化不会静默复用旧缓存。

## SBPQ 适配

- 严格加载 `ICON_CROPPED` / `ICON_UNCROPPED` 原 checkpoint，支持裸 state dict、
  `model_state_dict` / `state_dict` 包装及 `module.` 前缀。
- 复用 SafeSVD 的 attention 转换，将 Q/K/V/out 投影显式变为 Linear，关闭绕过 hooks 的融合路径。
- `pre_proj`、`post_proj` 各为一个结构块，每个 TransformerEncoderLayer 为一个结构块。
  默认 10 层模型共 12 个块、62 个候选 Linear；量化其 weight，bias/LayerNorm/位置嵌入保持原值。
- 敏感度为各块输出梯度平方的均值，损失为有效 query 上的归一化 task MSE + 物理单位 Sobolev L1。
  Sobolev 沿用 SBPQ 的分阶权重与混合导数，每阶内平均；mask 排除无效通道、边界和元数据通道。
- likelihood 对相同有效 query 的 FP 输出计算 MC log-mean-exp **平方误差之和**，
  保留原 SBPQ 的 Uniform 权重噪声、参数量加权 Beta 中心、步长参数化及 AdamW。
  演示对与 node-type 通道不进入 query 重建损失。

默认优化配方与 `configs/sbpq/default.json` 相同：初始/目标 8 bits、κ=150、Δbits=2、
MC=10、eta=1e-6、lr=3.75e-5、800 个累计更新。尚未对 VICON 调参；target bits 是先验中心，
实际平均位宽由 metadata 报告。batch 按类别与源顺序循环，`total_steps` 控制更新预算。

## 产物、恢复与评估

`--save_path` 是父目录，实际参数子目录在启动日志中打印，其中包含：

- `compressed_model.pt`：完整模型，已将量化权重 round 写入，无量化 hooks。
- `compressed_model.pth`：相同模型的 state dict，供 **SafeSVD 原 VICON evaluator** 加载。
- `quantization_state.pt`、`sbpq_preparation.pt`、`sbpq_beta_parameters.pt`。
- `sbpq_checkpoint.pt`、`sbpq_trainer_state.pt`、`history.json`。
- `experiment.json`、`metadata.json`、`costs.json`。

这些是浮点 fake-quantized 权重，未做整数打包。完整 `.pt` 需要本仓库与 SafeSVD 的 VICON 源码；
使用 `quantization.artifacts.load_quantized_artifact` 时会按 metadata 注册源码路径。
独立评估推荐 `.pth`，它不依赖 SBPQ 的模型类。

```bash
python -u -m SBPQ.sbpq --resume /absolute/path/to/actual-parameter-directory

python /path/to/SafeSVD/evaluation/vicon/evaluate.py \
  --vicon-root /path/to/SafeSVD/VICON \
  --checkpoint /path/to/SafeSVD/models/VICON/model.pth \
  --data-root /path/to/SafeSVD/data/processed_data/2D/CFD/converted \
  --config-override datasets.types.NS2D.folder=/path/to/SafeSVD/data/raw_data/NavierStokes-2D-conditoned \
  --model /absolute/path/to/actual-parameter-directory/compressed_model.pth \
  --batch-size 64 --device cuda \
  --output /absolute/path/to/actual-parameter-directory/evaluation.json
```

结束时会打印带完整路径、配置与 overrides 的独立评估命令。
该命令沿用 SafeSVD evaluator 默认的 research 测试协议；对比 SafeSVD 结果时使用相同评估协议。
若量化时使用了自定义 Hydra 配置或文件夹覆盖，评估时也传对应的
`--config-dir` / `--config-name` / `--config-override`。

小规模检查可用新输出目录，加 `--total_steps 4 --mc_samples 2 --save_every 1`。
测试命令使用微型真实 VICON 网络与合成 HDF5，检查采样/归一化逐值对齐、attention 等价、
MC 梯度、量化导出、SafeSVD 加载兼容性与精确断点恢复：

```bash
SAFE_SVD_ROOT=/path/to/SafeSVD python -m pytest tests/test_sbpq_vicon.py -q
```
