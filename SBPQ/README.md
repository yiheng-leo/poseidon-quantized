# SBPQ on ritchie

新的正式压缩入口是 `python -m SBPQ.sbpq --config configs/sbpq/default.json`。
校准、产物、独立评估与恢复命令见 [仓库说明](../README.md) 和 [迁移说明](../docs/ritchie_migration.md)。
新入口只执行压缩；默认使用完整轨迹校准，最终导出不需要 hooks 的 compressed_model.pt。
新入口的 save_path 是自动参数子目录的父目录，并记录纯优化与资源成本；详见 [成本与恢复](../docs/costs_and_resume.md)。

下文描述保留的作者入口 `python -u run_sbpq_poseidon.py`，它继续使用原始采样/缓存/验证协议和 SBPQ_* 环境变量，
保存路径不等同于新入口的 save_path。该入口默认优化参数也已更新为本次 NSPwC 实验值。

# SBPQ Poseidon memory and saving behavior

Run the complete experiment from the repository root with:

```bash
python -u run_sbpq_poseidon.py
```

`SBPQ_SOBOLEV_ORDER_WEIGHTS=1` is a scalar and broadcasts to every derivative
order (for order 2: `[1, 1, 1]`). Explicit comma-separated values must have
exactly `order + 1` entries. Invalid lengths fail during configuration, before
model loading or sensitivity computation. The `SBPQ_SOB_WEIGHTS` alias follows
the same rules. Scalar `1` retains the historical `sw1` cache/run tag.

The runner prints the selected Linear count before preparing caches. A cached
range file may contain a superset (e.g. 792 entries) while only 524 layers are
selected; it filters to the current selection without modifying the cache and
rejects missing selected entries. The cache-entry count is not the number of
layers being optimized.

The MC likelihood uses non-reentrant PyTorch checkpointing by default. Noisy
weights and activations are recomputed during backward with the original RNG
states. The original **sum** squared-error energy, log-mean-exp, MC count, eta,
Beta prior and optimizer configuration are preserved. Direct diagnostic calls
can pass `memory_efficient=False` to compare with the stacked implementation.
The trainer logs `MC backward: checkpoint replay with original RNG states.`

The trainer also releases FP weight gradients left by sensitivity estimation.
Validation forwards use at most eight samples at a time by default; predictions
are reassembled before computing the original loader-batch metrics. This does
not reduce the configured number of validation samples.

After optimization completes, `sbpq_trainer_state.pt` and `history.json` are
saved under `SBPQ/artifacts/poseidon/<model>/runs/<run_name>/` **before** FP,
SBPQ and fixed-bit validation. A validation failure therefore leaves the final
steps available.

## Periodic checkpoints and resume

New runs save `sbpq_checkpoint.pt` in the run directory before optimization,
every 100 optimizer updates by default, and after the final update. Writes use
a temporary file and atomic replacement so an interrupted save retains the
last completed checkpoint. Set the interval with `--save_every` or
`SBPQ_SAVE_EVERY`:

```bash
python -u run_sbpq_poseidon.py --save_every 100
```

Keep the original model, data and SBPQ training environment settings, then pass
either a run directory or a checkpoint file:

```bash
python -u run_sbpq_poseidon.py --resume /path/to/run
python -u run_sbpq_poseidon.py --resume /path/to/run/sbpq_checkpoint.pt
```

In tcsh/csh the equivalent environment option is:

```tcsh
setenv SBPQ_RESUME /path/to/run
setenv SBPQ_SAVE_EVERY 100
python -u run_sbpq_poseidon.py
```

A directory prefers `sbpq_checkpoint.pt`, then falls back to
`sbpq_trainer_state.pt`. Resume restores step sizes, AdamW state, history,
calibration batch position, and Python/NumPy/CPU/CUDA RNG states. Initialization
still reconstructs calibration batches and FP reference outputs, reusing range
and sensitivity caches; their fingerprints and the training settings are
checked before restoring the checkpoint. Same-environment CPU tests match
uninterrupted training exactly; GPU reproducibility also depends on hardware,
libraries and deterministic kernels.

The step budget is cumulative. A checkpoint at step 400 with a target of 1000
runs the remaining 600 updates. By default resume uses the saved target if
`SBPQ_NUM_OPTIMIZATION_STEPS` is unset. To extend training explicitly:

```bash
python -u run_sbpq_poseidon.py --resume /path/to/run --total_steps 2000
```

A completed target performs no further optimization and proceeds to evaluation.
Resumed output stays in the original run directory. Starting a fresh run in a
directory already containing a checkpoint is refused: use `--resume` or a new
`SBPQ_RUN_GROUP`. Unset `SBPQ_RESUME` for subsequent fresh experiments.

Legacy trainer states can continue from their stored optimizer/history with a
warning, but their original RNG and calibration fingerprints were not saved,
so they cannot promise identical continuation. A crashed old run with no saved
trainer state has no recoverable optimization progress. Changes since the most
recent periodic checkpoint are recomputed, at most 99 completed updates with
the default interval. Calibration batches/reference outputs are not duplicated
into every checkpoint.

Run numerical, memory-retention and save-order checks with:

```bash
python -m pytest SBPQ/tests -q
```

CUDA comparisons are skipped on machines without CUDA. Full Poseidon-L peak
memory must be measured on the target GPU; checkpointing adds recomputation.
