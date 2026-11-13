# 代码目录

每个量化方法使用 `<Method>/<method>.py` 启动、`quantize.py` 实现，返回
`QuantizationResult(model, metadata, state)`；公共 runner 负责加载校准、调用方法与保存，独立 evaluation 负责测试。

- `SBPQ/sbpq.py`、`SBPQ/quantize.py`：SBPQ 新正式入口与适配。
- `SBPQ/defaults.py`、`configs/sbpq/default.json`：本次 NSPwC 实验优化默认值。
- `SBPQ/poseidon/`：作者的 block、Sobolev、sensitivity、Beta builder、likelihood、range 等内核。
- `SBPQ/trainer.py`、`resume.py`、`storage.py`：优化、checkpoint/RNG 恢复和磁盘数据序列。
- `UniformQ/`、`DynamicQ/`、`StaticQ/`、`PPQ/`、`SAPQ/`、`BRECQ/`、`RepQ/`、`PTQ4ViT/`：迁入的独立 baseline 实现。
- `quantization/`：runner、完整轨迹 calibration、artifact、稳定推理层、缓存和兼容接口；experiments.py 控制参数命名，checkpoints.py/costs.py 控制断点与资源记录。
- `evaluation/evaluate.py`：独立评估；`evaluation/poseidon/` 为 SAFE-SVD 对齐协议。
- `scOT/`、`physical/`：模型与物理运行依赖。
- `configs/<method>/`：JSON 配置；原 finetuning YAML 保持原路径。
- `tests/`、`SBPQ/tests/`：当前测试；历史实验和上游测试不自动收集。
- `archive/main/`：被新 baseline 接口替换的原目录，保留追溯，不能依赖其旧路径继续直接运行。

作者 SBPQ 入口 `run_sbpq_poseidon.py` 保留历史协议，详见 [迁移说明](ritchie_migration.md)。
原 Aurora、finetuning、下载与绘图材料保持原位置；本次新增的统一量化/评估入口适用于 Poseidon。
