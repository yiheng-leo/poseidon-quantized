"""Effective bitwidth for artifacts that export fixed-grid weight scales."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from SBPQ.step_sizes import effective_bitwidth_from_step_size


FIXED_GRID_METHODS = {"brecq", "uniform", "dynamic", "static", "repq", "ptq4vit"}


def file_identity(path):
    info = path.stat()
    return {"path": str(path.resolve()), "bytes": info.st_size,
            "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns}


def checkpoint_identity(checkpoint):
    if not checkpoint:
        return None
    path = Path(checkpoint).resolve(strict=True)
    files = [path] if path.is_file() else sorted(p for p in path.iterdir() if p.is_file() and (
        p.name == "config.json" or p.suffix in {".bin", ".safetensors"}
        or p.name.endswith(".index.json")
    ))
    if not files:
        raise ValueError(f"No checkpoint files found in {path}")
    return [file_identity(p) for p in files]


def reference_checkpoint(metadata, fallback=None):
    for value in (metadata.get("source_model_path"), metadata.get("settings", {}).get("model_path"), fallback):
        if value and Path(value).expanduser().exists():
            return Path(value).expanduser().resolve()
    return None


def weight_state_path(model_path):
    path = Path(model_path)
    return (path if path.is_dir() else path.parent) / "quantization_state.pt"


def _reference_range(weight, percentile, input_ratio=None):
    weight = weight.detach().float().flatten(1)
    if input_ratio is not None:
        weight = weight * input_ratio.flatten()
    width = weight.shape[1]
    probabilities = weight.new_tensor([percentile, 1 - percentile])
    ranges = []
    # Bound quantile sorting workspace for large Linear and convolution layers.
    for channels in weight.split(max(1, (1 << 22) // width), dim=0):
        low, high = torch.quantile(channels, probabilities, dim=1)
        ranges.append((high - low).clamp_min(1e-12))
    return torch.cat(ranges), width


@lru_cache(maxsize=4)
def _reference_ranges(checkpoint, _checkpoint_files, names, percentile):
    # Cache channel summaries across artifacts, with checkpoint content identity in the key.
    from quantization.runner import _load_model

    model = _load_model(checkpoint, "cpu")
    return {name: _reference_range(model.get_submodule(name).weight, percentile) for name in names}


@torch.no_grad()
def average_fixed_grid_bits(metadata, model_path, checkpoint=None):
    """Compute parameter-weighted log2(R/S) using FP channel percentiles."""
    quantization = metadata["quantization"]
    method = quantization["method"]
    source = reference_checkpoint(metadata, checkpoint)
    state_path = weight_state_path(model_path)
    if source is None or not state_path.is_file():
        return None

    # BRECQ also exports per-weight AdaRound tensors. Map the sidecar so that
    # only the small scale tensors used here are loaded from it.
    state = torch.load(state_path, map_location="cpu", weights_only=True, mmap=True)
    names = tuple(sorted(quantization.get("quantized_layers", state)))
    if not names or any(name not in state for name in names):
        return None
    percentile = float(metadata.get("settings", {}).get("percentile", 1e-4))
    if not 0 < percentile < 0.5:
        raise ValueError(f"Invalid weight range percentile: {percentile}")
    if method == "repq":
        from quantization.runner import _load_model

        # RepQ applies the input affine before weight QDQ. Measure R in the
        # same transformed coordinates as the saved scale.
        model = _load_model(source, "cpu")
        ranges = {name: _reference_range(model.get_submodule(name).weight, percentile,
                                         state[name]["input_ratio"]) for name in names}
    else:
        files = tuple(tuple(file.values()) for file in checkpoint_identity(source))
        ranges = _reference_ranges(source, files, names, percentile)

    total_bits = 0.0
    num_weights = 0
    for name, (channel_ranges, weights_per_channel) in ranges.items():
        layer = state[name]
        if method == "repq":
            scale = layer["weight_quantizer"]["scale"]
        else:
            scale = layer["scale" if method == "brecq" else "weight_scale"]
        scale = torch.as_tensor(scale).flatten().to(channel_ranges)
        if scale.numel() not in (1, channel_ranges.numel()) or not torch.all(torch.isfinite(scale) & (scale > 0)):
            raise ValueError(f"Invalid weight quantization step for layer {name!r} in {state_path}")
        bits = effective_bitwidth_from_step_size(channel_ranges, scale)
        total_bits += float(bits.sum(dtype=torch.float64)) * weights_per_channel
        num_weights += channel_ranges.numel() * weights_per_channel
    return total_bits / num_weights
