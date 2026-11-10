"""Shared probability, step-size, functional-weight and calibration primitives.

Method orchestration lives in PPQ.quantize and SAPQ.quantize. This module does
not choose methods or own their training schedules.
"""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path
import tempfile
import torch
from torch import nn
import torch.nn.functional as F
from quantization.common import forward_model, move_to_device, output_tensor
from quantization.progress import progress

_EPS = 1e-8

def add_quantization_noise(tensor, step_sizes, channel_axis):
    """
    Add uniform quantization noise to a tensor, with one step size per channel.

    Args:
        tensor: input tensor
        step_sizes: 1D tensor of shape [num_channels]
        channel_axis: which axis is the channel axis
            - weights: usually 0
            - activations: usually -1

    Returns:
        tensor + uniform noise in [-0.5 * step, 0.5 * step]
    """
    if step_sizes.numel() != tensor.shape[channel_axis]:
        raise ValueError(
            f"step_sizes ({step_sizes.numel()}) does not match "
            f"tensor.shape[channel_axis] ({tensor.shape[channel_axis]})"
        )

    shape = [1] * tensor.dim()
    shape[channel_axis] = tensor.size(channel_axis)

    step_sizes_broadcast = step_sizes.view(shape)
    noise = (torch.rand_like(tensor) - 0.5) * step_sizes_broadcast
    return tensor + noise


def get_lr_for_epoch(
    epoch: int,
    base_lr: float = 1e-6,
    num_epochs: int = 4,
) -> float:
    """
    Piecewise learning-rate schedule over training epochs.

    Current schedule:
      first 25%   -> 1.0   * base_lr
      second 25%  -> 0.1   * base_lr
      third 25%   -> 0.01  * base_lr
      last 25%    -> 0.001 * base_lr
    """
    if epoch <= num_epochs * 0.25:
        return 1.0 * base_lr
    elif epoch <= num_epochs * 0.50:
        return 0.1 * base_lr
    elif epoch <= num_epochs * 0.75:
        return 0.01 * base_lr
    else:
        return 0.001 * base_lr


def compute_mdl_prior(
    step_sizes_dict,
    ranges_dict,
    gamma: float = 0.001,
    eps: float = 1e-8,
):
    """
    MDL prior on WEIGHT step sizes only.

    prior = gamma * sum_{l,k} log2(R_{l,k} / S_{l,k})

    Args:
        step_sizes_dict:
            {layer_name: (weight_step_sizes, activation_step_sizes)}
        ranges_dict:
            {layer_name: {"weight_ranges": ..., "activation_ranges": ...}}
    """
    device = None
    for pair in step_sizes_dict.values():
        for p in pair:
            if p is not None:
                device = p.device
                break
        if device is not None:
            break

    if device is None:
        device = torch.device("cpu")

    prior_loss = torch.zeros((), device=device)

    for name, (weight_step_sizes, _activation_step_sizes) in step_sizes_dict.items():
        rec = ranges_dict.get(name, None)
        if rec is None or "weight_ranges" not in rec:
            continue

        w_ranges = rec["weight_ranges"].to(device)

        if w_ranges.numel() != weight_step_sizes.numel():
            raise ValueError(
                f"[{name}] weight_ranges.shape={tuple(w_ranges.shape)} "
                f"!= weight_step_sizes.shape={tuple(weight_step_sizes.shape)}"
            )

        w_term = torch.log2(
            torch.clamp(w_ranges, min=eps) /
            torch.clamp(weight_step_sizes, min=eps)
        )
        prior_loss = prior_loss + gamma * torch.sum(w_term)

    return prior_loss


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu(item) for item in value)
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    return value


def _inputs(batch):
    """Calibration targets are never used, even to compute the model's own loss."""
    if torch.is_tensor(batch):
        return batch
    if isinstance(batch, (tuple, list)):
        return batch[0]
    if isinstance(batch, dict):
        return {key: batch[key] for key in ("pixel_values", "time", "pixel_mask")
                if key in batch and batch[key] is not None}
    raise TypeError("Calibration batches must be tensors, ScOT dicts, or (input, target).")


def _num_samples(batch):
    tensor = batch if torch.is_tensor(batch) else batch.get("pixel_values")
    if not torch.is_tensor(tensor) or tensor.ndim == 0:
        raise ValueError("Calibration batch has no batched input tensor.")
    return int(tensor.shape[0])


def _requires_input_grad(batch, device):
    batch = move_to_device(batch, device)
    x = batch if torch.is_tensor(batch) else batch["pixel_values"]
    if not x.is_floating_point():
        raise ValueError("Physical model calibration inputs must be floating point.")
    x = x.detach().requires_grad_(True)
    return x if torch.is_tensor(batch) else {**batch, "pixel_values": x}


def _load(path):
    # These files are tensor-only scratch data created by this invocation.
    return torch.load(path, map_location="cpu", weights_only=True)


@contextmanager
def _scratch_directory(parent, prefix, keep):
    if keep:
        yield Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    else:
        with tempfile.TemporaryDirectory(prefix=prefix, dir=parent) as directory:
            yield Path(directory)


def _freeze_calibration(model, loader, directory, device):
    """Consume the loader once, retaining every batch including its partial tail."""
    paths, samples = [], 0
    with torch.no_grad():
        for index, batch in enumerate(progress(loader, desc="Cache FP calibration outputs")):
            batch = _cpu(_inputs(batch))
            count = _num_samples(batch)
            if count == 0:
                raise ValueError("Empty calibration batches are unsupported.")
            clean = _cpu(forward_model(model, batch, device))
            if not torch.isfinite(clean).all():
                raise ValueError(f"Non-finite FP prediction in calibration batch {index}.")
            path = directory / f"batch_{index:06d}.pt"
            torch.save({"batch": batch, "clean": clean}, path)
            paths.append(path)
            samples += count
    if not paths:
        raise ValueError("Calibration loader is empty.")
    return paths, samples


def _weight_ranges(layers, percentile):
    """Original PPQ Gaussian range: 2 * Phi^-1(1-p) * per-channel std."""
    z = math.sqrt(2.0) * torch.special.erfinv(
        torch.tensor(1.0 - 2.0 * percentile, dtype=torch.float64)).item()
    return {name: {"weight_ranges": (2.0 * z * layer.weight.detach().float().std(
        dim=1, unbiased=False)).clamp_min(_EPS)}
        for name, layer in progress(layers.items(), desc="Estimate channel weight ranges", unit="layer")}


@dataclass
class _Unit:
    name: str
    module: nn.Module
    layers: dict


def _first_tensor(output):
    if isinstance(output, (tuple, list)):
        return _first_tensor(output[0])
    return output_tensor(output)


@contextmanager
def _weight_hooks(layers, steps, *, noisy):
    """Functional weights retain gradients to steps; Parameter replacement does not."""
    handles = []
    try:
        for name, module in layers.items():
            step = steps[name]

            def hook(mod, inputs, output, step=step):
                weight = mod.weight
                if noisy:
                    weight = add_quantization_noise(weight, step.to(weight.dtype), 0)
                else:
                    scale = step.to(weight.dtype).view(-1, 1)
                    weight = torch.round(weight / scale) * scale
                return F.linear(inputs[0], weight, mod.bias)

            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _reference_quantizers(model, bits):
    """Legacy PoseidonQuantModel reference: affine per-tensor Linear/Conv weights.

    This reference is used only for sensitivity/Fisher/asymmetric inputs. The
    learned and exported quantizer remains selected Linear, per-output-channel.
    """
    modules = dict(model.named_modules())
    excluded = [name for name, mod in modules.items() if isinstance(mod, (nn.LayerNorm, nn.BatchNorm2d))
                or type(mod).__name__ in {"LayerNorm", "ConditionalLayerNorm"}]
    layers = {name: mod for name, mod in modules.items() if isinstance(mod, (nn.Linear, nn.Conv2d))
              and not any(name == skip or name.startswith(skip + ".") for skip in excluded)}
    quantizers = {}
    for name, layer in progress(layers.items(), desc="Initialize reference quantizers", unit="layer"):
        weight = layer.weight.detach()
        lower, upper = min(weight.min().item(), 0.0), max(weight.max().item(), 0.0)
        delta = max((upper - lower) / (2 ** bits - 1), _EPS)
        quantizers[name] = (weight.new_tensor(delta), round(-lower / delta), 2 ** bits - 1)
    return layers, quantizers


@contextmanager
def _reference_hooks(layers, quantizers):
    handles = []
    try:
        for name, layer in layers.items():
            delta, zero, maximum = quantizers[name]

            def hook(module, inputs, output, delta=delta, zero=zero, maximum=maximum):
                quantized = ((torch.round(module.weight / delta) + zero).clamp(0, maximum) - zero) * delta
                if isinstance(module, nn.Conv2d):
                    return module._conv_forward(inputs[0], quantized, module.bias)
                return F.linear(inputs[0], quantized, module.bias)

            handles.append(layer.register_forward_hook(hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


def _dataset_constants(loader):
    dataset = getattr(loader, "dataset", None)
    seen = set()
    while dataset is not None and id(dataset) not in seen:
        seen.add(id(dataset))
        constants = getattr(dataset, "constants", None)
        if constants is not None:
            return constants
        dataset = getattr(dataset, "source_dataset", getattr(dataset, "dataset", None))
    return None


def _clamp(steps, ranges, max_bits):
    with torch.no_grad():
        for name, step in steps.items():
            maximum = ranges[name]["weight_ranges"]
            step.clamp_(min=(maximum / (2.0 ** max_bits)).clamp_min(_EPS), max=maximum)


def _average_bits(steps, ranges, layers):
    with torch.no_grad():
        total, parameters = 0.0, 0
        for name, step in steps.items():
            bits = torch.log2(ranges[name]["weight_ranges"].clamp_min(_EPS) / step.clamp_min(_EPS))
            total += float(bits.sum()) * layers[name].in_features
            parameters += layers[name].weight.numel()
    return total / parameters


@contextmanager
def _seed(seed, device):
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(seed)
        mps_state = torch.mps.get_rng_state() if device.type == "mps" else None
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        elif device.type == "mps":
            torch.mps.manual_seed(seed)
        try:
            yield
        finally:
            if mps_state is not None:
                torch.mps.set_rng_state(mps_state)


def _energy(pred, target, eta, fisher=None, *, output_sum=False):
    if pred.shape != target.shape:
        raise ValueError(f"Prediction shape {tuple(pred.shape)} differs from target {tuple(target.shape)}.")
    error = (pred - target).square()
    if fisher is not None:
        if fisher.shape != pred.shape:
            raise ValueError("Fisher gradients must match the reconstruction output.")
        return 0.5 * (error * fisher.square()).reshape(pred.shape[0], -1).sum(dim=1)
    flat = error.reshape(pred.shape[0], -1)
    return (flat.sum(dim=1) if output_sum else flat.mean(dim=1)) / (2.0 * eta)


def _backward_likelihood(forward, target, args, device, *, marginal, fisher=None, loss_scale=1.0, output_sum=False):
    """Exact log-mean-exp gradient with only one MC computation graph resident.

    A no-grad pass obtains mixture responsibilities; replaying each seed gives
    identical noise and the weighted energy gradient. This is algebraically the
    gradient of the legacy stacked MC loss, not a changed likelihood estimator.
    """
    seeds = torch.randint(0, 2 ** 31 - 1, (args.mc_samples,)).tolist()
    if not marginal:
        loss_value = 0.0
        for seed in progress(seeds, desc="MC likelihood", unit="sample", leave=False, delay=0.5):
            with _seed(seed, device):
                loss = loss_scale * _energy(forward(), target, args.eta).mean() / args.mc_samples
            loss.backward()
            loss_value += float(loss.detach())
        return loss_value
    scores = []
    with torch.no_grad():
        for seed in progress(seeds, desc="MC likelihood forward", unit="sample", leave=False, delay=0.5):
            with _seed(seed, device):
                scores.append(-_energy(forward(), target, args.eta, fisher,
                                       output_sum=output_sum))
        scores = torch.stack(scores)
        probabilities = scores.softmax(dim=0)
        loss_value = -loss_scale * (scores.logsumexp(dim=0) - math.log(args.mc_samples)).sum().item()
    for index, seed in enumerate(progress(seeds, desc="MC likelihood backward", unit="sample", leave=False, delay=0.5)):
        with _seed(seed, device):
            energy = _energy(forward(), target, args.eta, fisher,
                             output_sum=output_sum)
        (loss_scale * probabilities[index] * energy).sum().backward()
    return loss_value


class _StopCapture(Exception):
    pass


def _capture_unit(model, unit, batch, device, *, stop=True):
    captured = {}

    def capture(mod, inputs, kwargs, output):
        captured.update(args=_cpu(inputs), kwargs=_cpu(kwargs), target=_cpu(_first_tensor(output)))
        if stop:
            raise _StopCapture

    handle = unit.module.register_forward_hook(capture, with_kwargs=True)
    try:
        with torch.no_grad():
            try:
                forward_model(model, batch, device)
            except _StopCapture:
                pass
    finally:
        handle.remove()
    if not captured:
        raise ValueError(f"Reconstruction unit {unit.name!r} did not execute.")
    return captured


def _cache_layer(model, unit, paths, directory, device):
    """Persist one layer's clean inputs/output, preserving calibration order."""
    result = []
    for index, path in enumerate(progress(paths, desc=f"Cache layer IO: {unit.name}", leave=False)):
        record = _load(path)
        captured = _capture_unit(model, unit, record["batch"], device)
        local_path = directory / f"unit_{index:06d}.pt"
        torch.save(captured, local_path)
        result.append(local_path)
    return result


def _validate(args):
    if args.activation_bits != 32:
        raise ValueError("PPQ/SAPQ adapters support weight-only quantization; activation_bits must be 32.")
    if args.epochs < 1 or args.mc_samples < 1:
        raise ValueError("epochs and mc_samples must be positive.")
    if not (0 < args.percentile < 0.5) or args.eta <= 0 or args.learning_rate <= 0:
        raise ValueError("Require 0 < percentile < 0.5, eta > 0 and learning_rate > 0.")
    if not 0 < args.init_bits <= args.bmax_bits or args.bmax_bits > 30:
        raise ValueError("Require 0 < init_bits <= bmax_bits <= 30.")
    if args.sigma0 <= 0 or args.alpha < 0 or args.prior_scale < 0 or args.gamma < 0:
        raise ValueError("Prior sigma0 must be positive; alpha, prior_scale and gamma must be nonnegative.")
    if getattr(args, "sensitivity_normalization", "raw") not in {"raw", "normalized"}:
        raise ValueError("sensitivity_normalization must be raw or normalized.")
