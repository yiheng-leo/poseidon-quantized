"""Original main/db1d702 global SAPQ experiment behind ``SAPQ.sapq --origin``.

Defaults come from PPQ/config.py + SAPQ/run_sapq_network_global.py, rather
than the separate *_raw or *_all experiments. Keep the original MC estimator,
label/mask forward, legacy bit statistic and early stopping of that trainer.
Replay original RNG states to backpropagate one MC graph at a time.
Reuse trusted original caches when present; report every regenerated input.
"""
from __future__ import annotations

from itertools import islice
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from quantization.calibration import dataset_metadata, _fingerprint
from quantization.common import QuantizationResult, move_to_device, output_tensor, select_quantizable_layers
from quantization.probabilistic import (
    _clamp, _cpu, _dataset_constants, _load, _reference_quantizers, _seed,
    _weight_hooks, get_lr_for_epoch,
)
from quantization.progress import progress, progress_write
from .loss import _prior
from .reconstruction import _block_units
from .sensitivity import _sensitivity


DEFAULTS = {
    "method": "sapq", "variant": "global", "traj_num": 2048,
    "random_calibration": "n", "final_time": 14, "time_back": 0,
    "batch_size": 2, "calib_steps": 512, "weight_bits": 8, "activation_bits": 32,
    "init_bits": 9, "target_bits": 8.0, "epochs": 21, "mc_samples": 5,
    "learning_rate": 9.1e-4, "eta": 1e-6, "prior_mode": "block_sens",
    "prior_scale": 1e-4, "sigma0": 0.5, "alpha": 1.0,
    "sensitivity": "sobolev", "sensitivity_normalization": "normalized",
    "percentile": 1e-4, "bmax_bits": 20, "min_epochs": 10, "early_stop_bits": 4.0,
}
REFERENCE = "main/db1d702:SAPQ/run_sapq_network_global.py + PPQ/config.py"


def apply_defaults(args):
    """Origin selects a complete experiment, overriding modern config/CLI knobs.

    Keep model/data/dataset/output paths, device, seed and operational switches.
    Record displaced settings so an old traj16 command cannot silently retain
    its trajectory selection, raw sensitivity, or shortened time horizon.
    """
    if getattr(args, "origin_profile", None) == REFERENCE:
        return args
    args.origin_overrides = {
        key: getattr(args, key) for key, value in DEFAULTS.items()
        if hasattr(args, key) and getattr(args, key) != value
    }
    for key, value in DEFAULTS.items():
        setattr(args, key, value)
    args.origin_profile = REFERENCE
    if args.model_path is None:
        args.model_path = "models/NS-PwC-L"
    return args


def _find_cache(args, kind):
    explicit = getattr(args, f"origin_{kind}", None)
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"--origin_{kind} file not found: {path}")
        return path
    root = Path(getattr(args, "origin_root", None) or Path(__file__).resolve().parents[1])
    tags = [Path(args.model_path).name]
    # Support relocated checkpoints whose directory is now the dataset name.
    if args.dataset == "fluids.incompressible.PiecewiseConstants" and args.model_name == "Poseidon-L":
        if "NS-PwC-L" not in tags:
            tags.append("NS-PwC-L")
    percentile = f"p{args.percentile:.0e}"
    if kind == "frozen_batches":
        paths = [root / "ppq_artifacts" / "frozen_calibration_batches.pt"]
    elif kind == "sensitivity":
        paths = [root / "SAPQ" / "prior_sensitivity_sobo" / tag / "prior_sensitivity.pt" for tag in tags]
    elif kind == "ranges":
        paths = [root / "precalculated_ranges" / tag / percentile / "ranges.pt" for tag in tags]
    elif kind == "initial_steps":
        paths = [root / "initial_step_sizes" / tag / percentile / f"{args.init_bits}bit.pt" for tag in tags]
    else:
        paths = [root / "inspect_layers" / "L_quantize_layers.pt",
                 root / "inspect_layers" / "L_quantize_layers.json"]
    return next((path.resolve() for path in paths if path.is_file()), None)


def _read(path):
    # Original caches are trusted PyTorch files supplied by this experiment.
    progress_write(f"[origin] Load original cache: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


class _CalibrationBatches:
    def __init__(self, source, batches, length, cached=False):
        self.dataset, self.batches, self.length, self.cached = source, batches, length, cached

    def __len__(self):
        return self.length

    def __iter__(self):
        return iter(self.batches) if self.cached else islice(self.batches, self.length)


def build_calibration_loader(args, dataset_factory=None):
    """Original frozen batches, or the first 512 shuffled batches of size two.

    Do not supply time overrides: scOT.get_dataset defaults to seven time steps
    of size two for ordinary downstream time-dependent datasets, as in main.
    Construct only the training data; evaluation remains a separate command.
    """
    apply_defaults(args)
    if dataset_factory is None:
        from scOT.problems.base import get_dataset
        dataset_factory = get_dataset
    source = dataset_factory(dataset=args.dataset, which="train", num_trajectories=2048,
                             data_path=str(args.data_path))
    path = _find_cache(args, "frozen_batches")
    if path:
        batches = _read(path)
        if not isinstance(batches, list) or not batches:
            raise ValueError("Original frozen calibration cache must be a nonempty list of batches.")
        if any(not isinstance(batch, dict) or "pixel_values" not in batch for batch in batches):
            raise ValueError("Original calibration batches must contain pixel_values.")
        loader = _CalibrationBatches(source, batches, len(batches), cached=True)
        samples = sum(len(batch["pixel_values"]) for batch in batches)
    else:
        progress_write("[origin] Original frozen batches unavailable; rebuilding 512 shuffled "
                       "training batches (1024 samples). This is a new calibration draw.")
        batches = DataLoader(source, batch_size=2, shuffle=True, num_workers=args.num_workers,
                             pin_memory=True, drop_last=False)
        count = min(args.calib_steps, len(batches))
        samples = min(len(source), count * 2)
        loader = _CalibrationBatches(source, batches, count)
    metadata = {
        "schema": "sapq-main-origin-calibration-v1", "dataset": args.dataset, "split": "train",
        "reference": REFERENCE, "selection": "original_frozen_batches" if path else "shuffled_samples",
        "frozen_batches_file": str(path) if path else None,
        "frozen_batches_sha256": _file_hash(path) if path else None,
        "num_samples": samples, "num_batches": len(loader), "train_capacity": 2048,
        "seed": args.seed, "source": dataset_metadata(source),
        "time_pairs": _cpu(getattr(source, "time_indices", None)),
        "batch_size": 2, "max_batches_without_cache": 512,
    }
    metadata["fingerprint"] = _fingerprint(metadata)
    return source, loader, metadata


def _file_hash(path):
    from quantization.artifacts import _sha256
    return _sha256(path)


def _forward(model, batch, device):
    if torch.is_tensor(batch):
        return output_tensor(model(batch.to(device)))
    return output_tensor(model(**{
        key: move_to_device(batch[key], device)
        for key in ("pixel_values", "time", "pixel_mask", "labels") if batch.get(key) is not None
    }))


def _freeze(model, loader, directory, device):
    paths, samples = [], 0
    with torch.no_grad():
        for index, batch in enumerate(progress(loader, desc="Origin cache FP outputs")):
            clean = _forward(model, batch, device)
            path = directory / f"batch_{index:06d}.pt"
            torch.save({"batch": _cpu(batch), "clean": _cpu(clean)}, path)
            paths.append(path)
            samples += len(batch if torch.is_tensor(batch) else batch["pixel_values"])
    if not paths:
        raise ValueError("Origin calibration loader is empty.")
    return paths, samples


def _prepare(model, args, device):
    """Original candidate filtering, range-cache and initialization-cache policy."""
    modules = dict(model.named_modules())
    paths = {kind: _find_cache(args, kind) for kind in ("layers", "ranges", "initial_steps", "sensitivity")}
    layer_path = paths["layers"]
    if layer_path:
        names = json.loads(layer_path.read_text()) if layer_path.suffix == ".json" else _read(layer_path)
        names = names["quantize_layers"] if isinstance(names, dict) else names
        layers = {name: modules[name] for name in names if isinstance(modules.get(name), nn.Linear)}
    else:
        progress_write("[origin] Original layer list unavailable; selecting Linear layers excluding norm/time.")
        layers = select_quantizable_layers(model)
    if paths["ranges"]:
        cached = _read(paths["ranges"])["ranges_dict"]
        layers = {name: layer for name, layer in layers.items() if name in cached
                  and cached[name]["weight_ranges"].numel() == layer.out_features
                  and cached[name]["activation_ranges"].numel() == layer.in_features}
        ranges = {name: {"weight_ranges": cached[name]["weight_ranges"].to(device)} for name in layers}
    else:
        progress_write("[origin] Original ranges unavailable; recomputing Gaussian ranges at p=1e-4.")
        z = torch.sqrt(torch.tensor(2.0, device=device)) * torch.special.erfinv(
            torch.tensor(1.0 - 2.0 * args.percentile, device=device))
        ranges = {}
        for name, layer in layers.items():
            mean, std = layer.weight.detach().mean(1), layer.weight.detach().std(1, unbiased=False)
            upper = mean + std * float(z)
            ranges[name] = {"weight_ranges": (upper - (2 * mean - upper)).clamp_min(1e-8)}
    cached_steps = _read(paths["initial_steps"])["step_sizes"] if paths["initial_steps"] else None
    if cached_steps is not None:
        layers = {name: layer for name, layer in layers.items() if name in cached_steps}
        ranges = {name: ranges[name] for name in layers}
    if not layers:
        raise ValueError("Origin found no compatible Linear layers in the model/original caches.")
    steps = {}
    for name, layer in layers.items():
        value = (cached_steps[name][0].to(device) if cached_steps is not None
                 else ranges[name]["weight_ranges"] / (2 ** args.init_bits))
        if value.shape != (layer.out_features,):
            raise ValueError(f"Original initial steps have incompatible shape for {name}.")
        steps[name] = nn.Parameter(value.detach().clone())
    _clamp(steps, ranges, args.bmax_bits)
    return layers, ranges, steps, paths


def _average_bits(steps, ranges, layers):
    # Keep main's additive epsilon, including in the early-stopping decision.
    total = sum(float((torch.log2((ranges[name]["weight_ranges"] + 1e-8)
                                 / (step + 1e-8)) * layers[name].in_features).sum())
                for name, step in steps.items())
    return total / sum(layer.weight.numel() for layer in layers.values())


def _rng_state(device):
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["device"] = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        state["device"] = torch.mps.get_rng_state()
    return state


def _restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["device"], device)
    elif device.type == "mps":
        torch.mps.set_rng_state(state["device"])


def _backward_likelihood(forward, target, args, device):
    """Original log-mean-exp value/gradient with one resident MC graph.

    Preserve the sequential noise draws of main, rather than generating new
    per-sample seeds. The no-grad pass obtains each sample's responsibility;
    replaying its saved RNG state computes the same weighted energy gradient.
    Restore the post-forward RNG state so the next optimizer update sees the
    original random sequence too. The MC count and batch size stay unchanged.
    """
    states, scores = [], []
    with torch.no_grad():
        for _ in range(args.mc_samples):
            states.append(_rng_state(device))
            pred = forward()
            dims = tuple(range(1, pred.ndim))
            scores.append(-(pred - target).square().mean(dim=dims) / (2 * args.eta))
            del pred
        scores = torch.stack(scores)
        probabilities = scores.softmax(0)
        loss_value = -(scores.logsumexp(0)
                       - torch.log(torch.tensor(float(args.mc_samples), device=device))).sum().item()
    next_state = _rng_state(device)
    try:
        for index, state in enumerate(states):
            _restore_rng(state, device)
            pred = forward()
            dims = tuple(range(1, pred.ndim))
            energy = (pred - target).square().mean(dim=dims) / (2 * args.eta)
            (probabilities[index] * energy).sum().backward()
            del pred, energy
    finally:
        _restore_rng(next_state, device)
    return loss_value


def _optimize(model, layers, steps, ranges, sensitivity, paths, args, device):
    optimizer = torch.optim.Adam(list(steps.values()), lr=args.learning_rate)
    history = []
    for epoch in progress(range(1, args.epochs + 1), desc="Origin SAPQ", unit="epoch"):
        lr = get_lr_for_epoch(epoch, args.learning_rate, args.epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        totals = {"likelihood": 0.0, "prior": 0.0}
        for path in progress(paths, desc=f"Origin epoch {epoch}/{args.epochs}", leave=False):
            record = _load(path)
            optimizer.zero_grad(set_to_none=True)
            def forward():
                with _weight_hooks(layers, steps, noisy=True):
                    return _forward(model, record["batch"], device)
            likelihood = _backward_likelihood(forward, record["clean"].to(device), args, device)
            prior = _prior(steps, ranges, sensitivity, args)
            if not math.isfinite(likelihood) or not torch.isfinite(prior):
                raise FloatingPointError("Non-finite original SAPQ objective.")
            prior.backward()
            if any(step.grad is None or not torch.isfinite(step.grad).all() for step in steps.values()):
                raise FloatingPointError("Missing or non-finite original SAPQ step gradients.")
            optimizer.step()
            _clamp(steps, ranges, args.bmax_bits)
            totals["likelihood"] += likelihood
            totals["prior"] += float(prior.detach())
        with torch.no_grad():
            bits = _average_bits(steps, ranges, layers)
        history.append({"epoch": epoch, "learning_rate": lr, "avg_bits": bits,
                        **{key: value / len(paths) for key, value in totals.items()}})
        progress_write(f"[origin] epoch {epoch}/{args.epochs} bits={bits:.6f} "
                       f"likelihood={history[-1]['likelihood']:.6g} prior={history[-1]['prior']:.6g}")
        if epoch >= args.min_epochs and bits < args.early_stop_bits:
            progress_write(f"[origin] EARLY STOP: epoch={epoch}, average bits < {args.early_stop_bits}")
            break
    return history


def quantize(model, calibration_loader, args, device, work_dir):
    apply_defaults(args)
    device = torch.device(device)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    progress_write(f"[origin] Experiment: {REFERENCE}")
    progress_write("[origin] Defaults: global, batch=2, 512 shuffled batches if no frozen cache, "
                   "normalized Sobolev sensitivity, init=9, target=8, MC=5, epochs<=21, "
                   "early stop after epoch 10 when average bits < 4.")
    progress_write("[origin] MC backward: replay original RNG states, one sample graph at a time.")
    if args.origin_overrides:
        progress_write("[origin] Replaced config/CLI experiment settings: " + json.dumps(args.origin_overrides))
    flags = {name: param.requires_grad for name, param in model.named_parameters()}
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")  # main's load_poseidon_model
        model.to(device).eval().requires_grad_(False)
        from quantization.probabilistic import _scratch_directory
        with _seed(args.seed, device), _scratch_directory(work_dir, "origin-", args.keep_work_dir) as scratch:
            paths, samples = _freeze(model, calibration_loader, Path(scratch), device)
            layers, ranges, steps, caches = _prepare(model, args, device)
            initial_steps = _cpu({name: value.detach().clone() for name, value in steps.items()})
            raw, normalized = {}, {}
            if caches["sensitivity"]:
                normalized = _read(caches["sensitivity"])["layer_sensitivity_norm"]
            else:
                progress_write("[origin] Original sensitivity unavailable; recomputing Sobolev sensitivity "
                               "on this calibration draw, with per-block min-max normalization.")
                reference_layers, reference_steps = _reference_quantizers(model, args.weight_bits)
                raw, normalized = _sensitivity(
                    model, _block_units(model, layers), reference_layers, reference_steps, paths,
                    device, args, _dataset_constants(calibration_loader), forward_fn=_forward)
            # main defaults to zero sensitivity for layers absent from its file.
            sensitivity = {name: normalized.get(name, torch.zeros_like(step)).to(device)
                           for name, step in steps.items()}
            if any(value.shape != steps[name].shape for name, value in sensitivity.items()):
                raise ValueError("Original sensitivity shapes do not match the selected model layers.")
            history = _optimize(model, layers, steps, ranges, sensitivity, paths, args, device)
            with torch.no_grad():
                avg_bits = _average_bits(steps, ranges, layers)
                for name, layer in layers.items():
                    scale = steps[name].to(layer.weight.dtype).view(-1, 1)
                    layer.weight.copy_(torch.round(layer.weight / scale) * scale)
            state = {"steps": _cpu(steps), "initial_steps": initial_steps, "ranges": _cpu(ranges),
                     "sensitivity": _cpu(sensitivity), "sensitivity_raw": _cpu(raw),
                     "sensitivity_normalized": _cpu(normalized), "history": history}
            metadata = {
                "method": "sapq", "origin": True, "reference": REFERENCE,
                "representation": "materialized_float_fake_quantized_weights", "weight_only": True,
                "activation_bits": 32, "layer_names": list(layers), "num_layers": len(layers),
                "avg_weight_bits": avg_bits, "initial_bits": args.init_bits, "target_bits": args.target_bits,
                "epochs": len(history), "max_epochs": args.epochs,
                "min_epochs": args.min_epochs, "early_stop_bits": args.early_stop_bits,
                "stopped_early": len(history) < args.epochs,
                "calibration_samples": samples, "calibration_batches": len(paths),
                "sensitivity_normalization": "normalized", "sensitivity": "sobolev",
                "likelihood": "original_mc_log_mean_exp_output_mse",
                "mc_backward": "single_graph_replay_original_rng_states",
                "variance_convention": "sigma0^2 * (1 + alpha * sensitivity)^2",
                "origin_caches": {key: {"path": str(path), "sha256": _file_hash(path)} if path else None
                                  for key, path in caches.items()},
                "evaluated_test_data": False,
            }
            return QuantizationResult(model, metadata, state)
    finally:
        torch.set_float32_matmul_precision(precision)
        for name, param in model.named_parameters():
            param.requires_grad_(flags[name])
