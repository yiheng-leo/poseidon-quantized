"""SAPQ workflows with separate update scopes and reconstruction observations.

Modes retain the current method identifiers: sapq (global/output), sapq-network
(block/output), sapq-block (block/local), sapq-layerwise (layer/local).
"""
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
from quantization.probabilistic import _reference_quantizers
from .loss import _prior, backward_likelihood
from .reconstruction import _block_units, _cache_unit, _reference_prefix
from .sensitivity import _sensitivity
from quantization.progress import progress, progress_write


def _optimize(model, layers, steps, ranges, sensitivity, paths, args, device,
              *, unit=None, block=False, network_unit=None):
    local_layers = (unit.layers if unit is not None else
                    network_unit.layers if network_unit is not None else layers)
    local_steps = {name: steps[name] for name in local_layers}
    optimizer = torch.optim.Adam(list(local_steps.values()), lr=args.learning_rate)
    history = []
    unit_name = unit.name if unit is not None else network_unit.name if network_unit is not None else "network"
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
                fisher=record["fisher"].to(device) if block else None,
            )
            prior = _prior(local_steps, ranges, sensitivity, args)
            # Preserve the original SAPQ block objective's additional factor.
            prior_weight = 1e-4 if block else 1e-10 if network_unit is not None else 1.0
            (prior_weight * prior).backward()
            if not math.isfinite(likelihood) or not torch.isfinite(prior):
                raise FloatingPointError("Non-finite probabilistic quantization loss.")
            if any(step.grad is None or not torch.isfinite(step.grad).all() for step in local_steps.values()):
                raise FloatingPointError("Missing or non-finite step-size gradients.")
            optimizer.step()
            _clamp(local_steps, ranges, args.bmax_bits)
            totals["likelihood"] += likelihood
            totals["prior"] += float(prior.detach())
            totals["total"] += likelihood + prior_weight * float(prior.detach())
            batches.set_postfix(likelihood=likelihood, prior=float(prior.detach()),
                                loss=likelihood + prior_weight * float(prior.detach()),
                                lr=lr, refresh=False)
        history.append({"unit": unit.name if unit is not None else (
                            network_unit.name if network_unit is not None else "network"),
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
    if getattr(args, "origin", False):
        from .origin import quantize as quantize_origin
        return quantize_origin(model, calibration_loader, args, device, work_dir)
    if args.method not in {"sapq", "sapq-block", "sapq-layerwise", "sapq-network"}:
        raise ValueError(f"Unknown SAPQ method: {args.method}")
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
            units = _block_units(model, layers)
            reference_layers, reference_steps = _reference_quantizers(model, args.weight_bits)
            raw, normalized, sensitivity = {}, {}, {}
            constants = _dataset_constants(calibration_loader)
            if args.prior_mode == "block_sens":
                raw, normalized = _sensitivity(model, units, reference_layers, reference_steps, paths,
                                                device, args, constants)
                sensitivity = normalized if getattr(args, "sensitivity_normalization", "raw") == "normalized" else raw
            history = []
            if args.method == "sapq":
                history = _optimize(model, layers, steps, ranges, sensitivity, paths, args, device)
            else:
                if args.method not in {"sapq-block", "sapq-network"}:
                    units = [_Unit(name, module, {name: module}) for name, module in layers.items()]
                for unit in progress(units, desc=f"{args.method} reconstruction units", unit="unit"):
                    if args.method == "sapq-network":
                        history.extend(_optimize(model, layers, steps, ranges, sensitivity,
                                                 paths, args, device, network_unit=unit))
                        continue
                    with _scratch_directory(scratch, "unit-", keep) as unit_dir:
                        prefix = _reference_prefix(model, unit, reference_layers)
                        unit_paths = _cache_unit(model, unit, reference_layers, prefix, reference_steps,
                                                 paths, Path(unit_dir), device, block=args.method == "sapq-block")
                        history.extend(_optimize(model, layers, steps, ranges, sensitivity,
                                                 unit_paths, args, device, unit=unit,
                                                 block=args.method == "sapq-block"))
            avg_bits = _average_bits(steps, ranges, layers)
            with torch.no_grad():
                for name, module in progress(layers.items(), desc="SAPQ materialize weights", unit="layer"):
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
                "likelihood": "mc_log_mean_exp_fisher" if args.method == "sapq-block" else (
                    "mc_log_mean_exp_output_mse" if args.method == "sapq" else (
                        "mc_log_mean_exp_output_sum_squared" if args.method == "sapq-network" else "mc_mean_mse")),
                "prior_mode": args.prior_mode,
                "gamma_used": args.prior_mode == "ppq",
                "avg_cap_bits": None,
                "sensitivity": args.sensitivity if sensitivity else None,
                "sensitivity_normalization": getattr(args, "sensitivity_normalization", "raw"),
                "sensitivity_reference_bits": args.weight_bits,
                "sensitivity_reference_quantizer": "per_tensor_affine_linear_and_conv2d",
                "sobolev_denormalized": args.sensitivity == "sobolev" and constants is not None,
                "sobolev_scaling_protocol": "legacy_dataset_std_broadcast_all_channels",
                "sobolev_differences": "forward_difference_without_grid_spacing",
                "variance_convention": "sigma0^2 * (1 + alpha * sensitivity)^2",
                "block_prior_multiplier": 1e-4 if args.method == "sapq-block" else (
                    1e-10 if args.method == "sapq-network" else 1.0),
                "block_fisher_damping": 1.0 if args.method == "sapq-block" else None,
                "layerwise_likelihood_divisor": 1,
                "evaluated_test_data": False,
            }
            return QuantizationResult(model=model, metadata=metadata, state=state)
    finally:
        for name, param in model.named_parameters():
            param.requires_grad_(original_flags[name])
