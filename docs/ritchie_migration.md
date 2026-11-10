# ritchie 迁移记录

迁移基于 poseidon-quantized/main `24354f9dc1d5c767e1f004a958516177efb9802b`，
公共框架、baseline 实现和独立评估取自
Structural-Aware-Probablistic-Quantization-for-Foundational-Physical-Models/ritchie
`b9b5daa720330d9a143b1b2fc7e84506916d36f5`。
模型实现继续使用本仓库的 `scOT/`，没有替换模型或数据文件。

## SBPQ

`SBPQ/sbpq.py` 启动公共 runner，`SBPQ/quantize.py` 适配作者的方法模块。
沿用 Beta parameter-weighted centering、每通道步长、网络输出 SUM squared MC likelihood、
eta、AdamW、固定步数循环、连续有效位宽以及 round(W/S)*S 推理网格。
MC checkpoint replay、原子 checkpoint、RNG 恢复沿用 main 的实现。
优化参数默认来自本次实验：B*=8、init=8、1..16、Δ=2、κ=150、MC=10、eta=1e-6、
prior/likelihood scale=1、lr=3.75e-5、800 steps、weight decay=0、无 clipping；敏感度 task MSE + H2 L1，所有阶权重=1。

新入口只选结构块内、排除 norm/time 的 Linear；Poseidon-L 为本次确认的 524 层。
层选择从模型结构生成，不依赖缺失的 L_quantize_layers.pt，也不会回退到包含 norm/time 的 792 层。
经验 percentile weight ranges 与 main 相同。SBPQ 是 weight-only，训练中未使用的 activation ranges 不再计算。
导出时按作者评估 hook 相同的 step.clamp_min(1e-12) 网格一次性写回权重，不保留 hooks。

## 校准改变

新入口调用公共 `quantization.calibration.build_calibration_loader`：训练 split、默认前 64 条轨迹，
或由 calibration_seed 决定的随机无放回轨迹选择；time_step_size=1，时间跨度 final_time-time_back，
保留完整轨迹内所有时间对及最后一个不满 batch。与迁移来源保持一致。
不使用 main 的“2048 条训练轨迹中 shuffle 后截取 512 batch”策略，不加载 val/test 参与优化。

例如 NSPwC 的 final_time=20、time_back=0 包含每条轨迹 231 个时间对；64 条为 14784 个样本，
batch size=2 时为 7392 个 batch。敏感度遍历全部数据，默认 800 个 likelihood 更新只覆盖前 800 个 batch。
若希望完整遍历，显式设置 total_steps=7392 或其倍数。这是预算和数据规模的区别，不是静默截断校准加载器。
样本/FP 输出逐 batch 保存在临时磁盘序列中，避免完整轨迹数据常驻 RAM。

metadata 保存轨迹索引、时间对、split、归一化、数据标识、batch size、校准 fingerprint 和优化覆盖量。
准备缓存保存在当前输出目录，验证 FP 权重、校准张量及顺序、层集合、损失设置和实现；不读取旧的匿名共享缓存。
断点恢复重新建立相同的数据快照与 FP 输出，再校验签名并恢复 optimizer/RNG。
更换数据、层或损失需创建新输出目录；历史 main 的 checkpoint 用作者入口恢复。

## Baselines 与公平比较

迁入 UniformQ、DynamicQ、StaticQ、PPQ(global/layerwise)、SAPQ(global/network/block/layerwise)、
BRECQ、RepQ 和 PTQ4ViT，保留旧 ritchie 实现和方法配置。各自需要校准时使用同一个完整轨迹选择接口。
FP/Uniform/Dynamic 和 A32 Static 不需要离线校准，不为它们加载无用的校准数据。

方法固有选层范围保持原实现：SBPQ 为结构块内的选中 Linear；Uniform/PPQ/SAPQ 等为排除 norm/time 的 Linear，
可能还包含块外投影；BRECQ Poseidon 包装包含 Conv2d、Linear 和条件归一化投影。
W8A8 baseline 与 SBPQ W≈8A32 不是相同 activation 配置。metadata 的层集合和 W/A 设置必须一起检查。
本次迁移不把不同算法的量化范围伪装成完全相同，也不改写它们的网格或重构目标。

## 评估与历史结果

正式量化与评估分离。统一产物可直接用于 `python -m evaluation.evaluate`；默认沿用 SAFE-SVD 全测试时间对物理指标，
输出紧凑 JSON。原始 checkpoint 仅在请求 FP 对比时需要。
该测试协议不同于本次论文复现实验的 val batch size=128、val steps=2；改变校准和评估后不应继续逐值对照 Table 3。

`run_sbpq_poseidon.py` 和 `SBPQ/poseidon/run_sbpq_network_global.py` 保留作者校准、缓存和验证生命周期，
默认优化配置更新为上述实验值；默认层列表使用仓库已有 JSON。它们继续用 SBPQ_* 环境变量。
旧 SAPQ/PPQ/BRECQ/DynamicQ/StaticQ 文件保存在 `archive/main/`，用于追溯，正式入口不导入这些历史文件。
模型、dataset、ppq_artifacts 等已有符号链接保持 main 原状；新实验应显式指定仓库外的稳定模型/数据路径。
