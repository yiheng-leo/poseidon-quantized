# 方法配置

`<method>/default.json` 提供各方法配置；显式 CLI 参数覆盖 JSON。
SBPQ 使用 `python -m SBPQ.sbpq --config configs/sbpq/default.json`，默认参数来自本次 NSPwC 复现实验，
校准选择独立通过 traj_num、random_calibration、calibration_seed、final_time、time_back 控制。
详情见 [仓库说明](../README.md)。新入口不从 SBPQ_* 环境变量继承作者实验设置。

`sbpq/vicon.json` 切换为 SafeSVD 的 VICON 模型与校准协议，默认每类 256 个校准样本、batch size 64。
使用 `--safe_svd_root`、`--ckpt_path`、`--data_root` 指向已有源码、模型和数据，详见 [VICON 说明](../SBPQ/vicon/README.md)。

PPQ 支持 global/layerwise；SAPQ 支持 global/network/block/layerwise，保留旧 ritchie 的 --origin。
SBPQ 当前只支持 global。raw/normalized 等均为参数，不产生额外入口。
DynamicQ/StaticQ/RepQ/PTQ4ViT 的 W8A8 模板与 SBPQ 的 A32 设置需要区分。
原 run*.yaml 和 sweep.yaml 是 finetuning 配置，保持原路径。

SBPQ、BRECQ、PPQ 的 save_path 是父目录，自动追加参数子目录；断点与成本见 [说明](../docs/costs_and_resume.md)。
