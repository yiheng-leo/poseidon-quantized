# Poseidon 环境迁移

核查日期：2026-11-19。已读取当前 UCL 服务器的 `poseidon-quant` 环境，并检查本仓库实际 Python 导入。
根目录 `requirements.txt` 覆盖当前 Poseidon 各量化方法、独立物理评估、scOT 推理／训练工具、绘图、
测试以及已内置的 VICON 配置依赖；`setting_environment/quantization_requirements.txt` 转发到同一份清单。

## 来源环境与安装

来源环境为 Linux x86_64、Python **3.11.17**；关键版本为 PyTorch **2.15.0+cu132**、
Transformers **4.35.2**、Accelerate **0.31.0**、NumPy **2.4.6**。
其中 Transformers 与 Accelerate 保留旧 API 版本，以匹配仓库自定义的 ScOT 模型和 Trainer。

新服务器创建独立的 Python 3.11 环境：

```bash
conda create -n poseidon-quant python=3.11 -y
conda activate poseidon-quant
```

先按新服务器的显卡与驱动选择 PyTorch GPU 构建，参照 [PyTorch 官方安装入口](https://pytorch.org/get-started/locally/)。
此处不提供 CUDA 下载或系统驱动安装命令。配置好 PyTorch 后，在仓库根目录安装其余依赖：

```bash
python -m pip install -r requirements.txt
python -m pip check
```

清单中的 `torch==2.15.0` 固定基础版本，允许已经安装的对应本地构建后缀，如 `+cu132`。
清单不直接列出 `cuda-toolkit`、`cuda-*`、`nvidia-*` 或 `triton`；PyTorch 安装器仍可能按所选
wheel 安装其需要的运行库，这与手动安装系统 CUDA Toolkit／驱动是两回事。

主要依赖及其非 CUDA Python 依赖采用来源环境的实际版本。`wandb==0.23.1` 是为
`scOT/inference.py`、`scOT/train.py` 和旧模型下载工具补充的兼容依赖，来源量化环境没有安装它，
因此这不是一份未经筛选的 `pip freeze`。使用已有本地模型和数据进行量化／评估，无需先执行 W&B 登录。

## 验证环境

本次已核验 48 个固定版本在官方 PyPI 可用且支持 Python 3.11；其中 47 个与来源环境一致，
另补 `wandb`。验证使用基于来源环境创建的临时 venv（`--system-site-packages`），只在该 venv 内
补装 W&B 及对齐构建工具；48 项约束全部满足，离线 pip dry-run 和 `pip check` 通过。
17 个量化／评估／ScOT／VICON 模块导入及 CPU Adam、AdamW 检查通过，CUDA 未初始化。
新服务器的 GPU 运行还需在其驱动和 PyTorch 配置完成后执行下方 CUDA 检查。

先检查纯 Python 导入和 CPU 优化器：

```bash
python -c "import torch, transformers, accelerate, h5py, pandas, wandb; from scOT.model import ScOT; from scOT.trainer import Trainer; print(torch.__version__, transformers.__version__, accelerate.__version__)"
python -m quantization.runtime --DEV cpu --optimizer Adam
python -m quantization.runtime --DEV cpu --optimizer AdamW
```

GPU 配置完成后，再检查实际 CUDA 算子和两种优化器：

```bash
python -c "import torch; print(torch.cuda.is_available()); x = torch.ones(1, device='cuda'); print((x + 1).item(), torch.cuda.get_device_name(0))"
python -m quantization.runtime --DEV cuda --optimizer Adam
python -m quantization.runtime --DEV cuda --optimizer AdamW
```

## 迁移实验文件

- 模型 checkpoint、HDF5 数据、量化模型和完整未完成实验目录需另外复制，Python 依赖文件不包含这些数据。
- 旧 PPQ／BRECQ optimizer checkpoint 会校验 **完整 PyTorch 版本字符串**、数值精度、源码和有效配置；
  继续当前旧断点时应匹配 `torch.__version__ == '2.15.0+cu132'`，同一基础版本的另一构建也可能被拒绝。
- 实验配置和等待脚本包含 UCL 的绝对路径；新实验需填写新服务器的模型、数据和输出位置。
  旧断点的路径属于被校验配置，不能仅替换 CLI 路径就假定可无缝恢复。
- 等待队列还需要系统提供 Bash、SSH、`tmux` 和 NVIDIA 驱动的 `nvidia-smi`，它们由系统安装，不属于 pip 包。

当前清单不为 `archive/`、旧 ImageNet 演示或 Aurora 工具安装额外的大型框架。
旧 Aurora／NetCDF 数据辅助工具另需 `microsoft-aurora`、`xarray`、`netCDF4`；旧 ImageNet 演示另需匹配
PyTorch 的 `torchvision` 和 `timm`；旧 Trainer 的 Adam8bit 后端另需 `bitsandbytes`。
当前 Poseidon 正式量化入口使用 Adam／AdamW，不依赖这些额外包。
