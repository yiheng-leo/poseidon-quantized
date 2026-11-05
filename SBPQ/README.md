# SBPQ Poseidon memory and saving behavior

Run the complete experiment from the repository root with:

```bash
python -u run_sbpq_poseidon.py
```

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
steps available. This does not add periodic training checkpoints or automatic
resume; an optimization failure before this save still requires a new run.

Run numerical, memory-retention and save-order checks with:

```bash
python -m pytest SBPQ/tests -q
```

CUDA comparisons are skipped on machines without CUDA. Full Poseidon-L peak
memory must be measured on the target GPU; checkpointing adds recomputation.
