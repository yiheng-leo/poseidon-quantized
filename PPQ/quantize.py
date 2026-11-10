"""PPQ network and layer-observation workflows with unchanged update order."""
from __future__ import annotations
import math
from pathlib import Path
import torch
from torch import nn
from quantization.common import (
    QuantizationResult, forward_model, move_to_device, select_quantizable_layers,
)
from quantization.probabilistic import (
    _Unit, _average_bits, _clamp, _cpu, _dataset_constants, _first_tensor,
    _freeze_calibration, _load, _scratch_directory, _seed, _validate,
    _weight_hooks, _weight_ranges, get_lr_for_epoch,
)
from quantization.probabilistic import _cache_layer
from .loss import backward_likelihood, prior as compute_prior
from quantization.progress import progress, progress_write


def _optimize(model, layers, steps, ranges, paths, args, device,
              *, unit=None):
    local_layers = unit.layers if unit is not None else layers
    local_steps = {name: steps[name] for name in local_layers}
    optimizer = torch.optim.Adam(list(local_steps.values()), lr=args.learning_rate)
    history = []
    unit_name = unit.name if unit is not None else "network"
    for epoch in progress(range(1, args.epochs + 1),
                          desc=f"{args.method} optimize: {unit_name}", unit="epoch"):
        lr = get_lr_for_epoch(epoch, args.learning_rate, args.epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        totals = {"likelihood": 0.0, "prior": 0.0, "total": 0.0}
        batches = progress(paths, desc=f"{args.method} epoch {epoch}/{args.epochs}",
                           leave=False)
        for path in batches:
            record = _load(path)
            target = record["clean" if unit is None else "target"].to(device)
            if unit is None:
                batch = move_to_device(record["batch"], device)
                def forward():
                    with _weight_hooks(local_layers, local_steps, noisy=True):
                        return forward_model(model, batch, device)
            else:
                inputs = move_to_device(record["args"], device)
                kwargs = move_to_device(record["kwargs"], device)
                def forward():
                    with _weight_hooks(local_layers, local_steps, noisy=True):
                        return _first_tensor(unit.module(*inputs, **kwargs))
            optimizer.zero_grad(set_to_none=True)
            likelihood = backward_likelihood(
                forward, target, args, device,
                layer_count=len(layers) if args.method == "ppq-layerwise" else 1,
            )
            prior = compute_prior(local_steps, ranges, local_layers, args)
            prior.backward()
            if not math.isfinite(likelihood) or not torch.isfinite(prior):
                raise FloatingPointError("Non-finite probabilistic quantization loss.")
            if any(step.grad is None or not torch.isfinite(step.grad).all() for step in local_steps.values()):
                raise FloatingPointError("Missing or non-finite step-size gradients.")
            optimizer.step()
            _clamp(local_steps, ranges, args.bmax_bits)
            totals["likelihood"] += likelihood
            totals["prior"] += float(prior.detach())
            totals["total"] += likelihood + float(prior.detach())
            batches.set_postfix(likelihood=likelihood, prior=float(prior.detach()),
                                loss=likelihood + float(prior.detach()), lr=lr, refresh=False)
        history.append({"unit": unit.name if unit is not None else "network",
                        "epoch": epoch, "learning_rate": lr, "num_batches": len(paths),
                        "avg_bits": _average_bits(steps, ranges, layers),
                        **{key: value / len(paths) for key, value in totals.items()}})
        progress_write(f"[{args.method}] {history[-1]['unit']} epoch {epoch}/{args.epochs} "
                       f"loss={history[-1]['total']:.6g} bits={history[-1]['avg_bits']:.4f}")
    return history


def quantize(model, calibration_loader, args, device, work_dir):
    """Learn steps from all supplied calibration samples and materialize weights.

    Model parameters are frozen during calibration; only channel steps are
    optimized. Returned weights are floating point fake-quantized values, not a
    packed integer representation. Inference needs no hooks or calibration files.
    """
    if args.method not in {"ppq", "ppq-layerwise"}:
        raise ValueError(f"Unknown PPQ method: {args.method}")
    _validate(args)
    device = torch.device(device)
    model = model.to(device).eval()
    layers = select_quantizable_layers(model)
    if not layers:
        raise ValueError("No eligible Linear layers were found for probabilistic quantization.")
    original_flags = {name: param.requires_grad for name, param in model.named_parameters()}
    for param in model.parameters():
        param.requires_grad_(False)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        keep = getattr(args, "keep_work_dir", False)
        with _seed(args.seed, device), _scratch_directory(work_dir, "probabilistic-", keep) as scratch:
            paths, num_samples = _freeze_calibration(model, calibration_loader, scratch, device)
            ranges = _weight_ranges(layers, args.percentile)
            steps = {name: nn.Parameter(rec["weight_ranges"].clone() / (2.0 ** args.init_bits))
                     for name, rec in ranges.items()}
            _clamp(steps, ranges, args.bmax_bits)
            initial_steps = {name: value.detach().cpu().clone() for name, value in steps.items()}
            raw, normalized, sensitivity = {}, {}, {}
            constants = _dataset_constants(calibration_loader)
            history = []
            if args.method == "ppq":
                history = _optimize(model, layers, steps, ranges, paths, args, device)
            else:
                units = [_Unit(name, module, {name: module}) for name, module in layers.items()]
                for unit in progress(units, desc="PPQ reconstruction layers", unit="layer"):
                    with _scratch_directory(scratch, "unit-", keep) as unit_dir:
                        unit_paths = _cache_layer(model, unit, paths, Path(unit_dir), device)
                        history.extend(_optimize(model, layers, steps, ranges,
                                                 unit_paths, args, device, unit=unit))
            avg_bits = _average_bits(steps, ranges, layers)
            with torch.no_grad():
                for name, module in progress(layers.items(), desc="PPQ materialize weights", unit="layer"):
                    scale = steps[name].to(module.weight.dtype).view(-1, 1)
                    module.weight.copy_(torch.round(module.weight / scale) * scale)
            state = {"steps": _cpu(steps), "initial_steps": initial_steps, "ranges": _cpu(ranges),
                     "sensitivity": sensitivity, "sensitivity_raw": raw,
                     "sensitivity_normalized": normalized, "history": history}
            metadata = {
                "method": args.method, "representation": "materialized_float_fake_quantized_weights",
                "weight_only": True, "activation_bits": 32, "num_layers": len(layers),
                "layer_names": list(layers), "calibration_samples": num_samples,
                "calibration_batches": len(paths), "calibration_storage": "temporary_disk",
                "initial_bits": args.init_bits, "target_bits": args.target_bits,
                "avg_weight_bits": avg_bits, "epochs": args.epochs, "mc_samples": args.mc_samples,
                "likelihood": "mc_mean_mse",
                "prior_mode": "weighted_average_bit_cap" if args.method == "ppq" else "ppq",
                "gamma_used": args.method == "ppq-layerwise",
                "avg_cap_bits": getattr(args, "avg_cap_bits", 8.0) if args.method == "ppq" else None,
                "sensitivity": args.sensitivity if sensitivity else None,
                "sensitivity_normalization": getattr(args, "sensitivity_normalization", "raw"),
                "sensitivity_reference_bits": args.weight_bits,
                "sensitivity_reference_quantizer": "per_tensor_affine_linear_and_conv2d",
                "sobolev_denormalized": args.sensitivity == "sobolev" and constants is not None,
                "sobolev_scaling_protocol": "legacy_dataset_std_broadcast_all_channels",
                "sobolev_differences": "forward_difference_without_grid_spacing",
                "variance_convention": "sigma0^2 * (1 + alpha * sensitivity)^2",
                "block_prior_multiplier": 1.0,
                "block_fisher_damping": None,
                "layerwise_likelihood_divisor": len(layers) if args.method == "ppq-layerwise" else 1,
                "evaluated_test_data": False,
            }
            return QuantizationResult(model=model, metadata=metadata, state=state)
    finally:
        for name, param in model.named_parameters():
            param.requires_grad_(original_flags[name])
