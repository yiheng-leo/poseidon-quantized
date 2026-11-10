"""Sample-wise field and physical structure metrics for ScOT datasets."""
from __future__ import annotations

import re

import torch

from quantization.calibration import _underlying_dataset


def denormalize_output(value, dataset, dataset_name):
    dataset = _underlying_dataset(dataset)
    constants = getattr(dataset, "constants", {})
    channels = value.shape[1]
    if "wave" in dataset_name:
        mean = [constants["mean"], constants["mean_c"]][:channels]
        std = [constants["std"], constants["std_c"]][:channels]
    elif "mean_solution" in constants:
        mean, std = [constants["mean_solution"]], [constants["std_solution"]]
    elif "mean" in constants and "std" in constants:
        mean = torch.as_tensor(constants["mean"]).flatten().tolist()
        std = torch.as_tensor(constants["std"]).flatten().tolist()
        names = [name.strip() for group in re.findall(r"\[([^\]]+)\]", getattr(dataset, "label_description", ""))
                 for name in group.split(",")]
        while channels > len(mean) and len(mean) < len(names):
            name = names[len(mean)]
            if name == "tracer" and "incompressible" in dataset_name:
                mean.append(constants["tracer_mean"])
                std.append(constants["tracer_std"])
            elif name == "tracer":
                # CompressibleBase appends its tracer without normalization.
                mean.append(0.0)
                std.append(1.0)
            elif name == "g" and "mean_forcing" in constants:
                mean.append(constants["mean_forcing"])
                std.append(constants["std_forcing"])
            else:
                break
    elif hasattr(dataset, "mean") and hasattr(dataset, "std"):
        mean, std = [dataset.mean], [dataset.std]
    else:
        raise ValueError("Dataset does not expose output normalization for physical metrics.")
    mean = torch.as_tensor(mean, dtype=value.dtype, device=value.device).flatten()
    std = torch.as_tensor(std, dtype=value.dtype, device=value.device).flatten()
    if mean.numel() not in (1, channels) and mean.numel() < channels:
        raise ValueError("Output normalization does not cover all channels.")
    result = value * std[:channels].view(1, -1, 1, 1) + mean[:channels].view(1, -1, 1, 1)
    if hasattr(dataset, "mean_pressure") and channels > 3:
        # CompressibleBase subtracts a dataset-specific pressure offset *before*
        # applying the shared normalization constants.
        result[:, 3] += float(dataset.mean_pressure)
    return result


def _sample_mean(value):
    return value.flatten(1).mean(1)


def per_sample_metrics(pred, target, dataset, dataset_name):
    """Return vectors, so partial final batches receive correct sample weights."""
    dataset = _underlying_dataset(dataset)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {target.shape}.")
    pred, target = pred.double(), target.double()
    error = (pred - target).abs().flatten(1).sum(1)
    norm = target.abs().flatten(1).sum(1)
    denominator = torch.where(norm == 0, torch.full_like(norm, 1e-10), norm)
    metrics = {"L1": error, "RelL1": 100 * error / denominator}
    p = denormalize_output(pred, dataset, dataset_name)
    y = denormalize_output(target, dataset, dataset_name)
    dx, dy = 1.0 / p.shape[-1], 1.0 / p.shape[-2]
    if min(p.shape[-2:]) < 3:
        raise ValueError("Physical structure metrics require at least a 3 x 3 grid.")
    dx_p, dx_y = torch.diff(p, dim=-1) / dx, torch.diff(y, dim=-1) / dx
    dy_p, dy_y = torch.diff(p, dim=-2) / dy, torch.diff(y, dim=-2) / dy
    sob = _sample_mean((p-y).abs()) + _sample_mean((dx_p-dx_y).abs()) + _sample_mean((dy_p-dy_y).abs())
    scale = _sample_mean(y.abs()) + _sample_mean(dx_y.abs()) + _sample_mean(dy_y.abs())
    metrics.update(sobolev_s01=sob, rel_sobolev_s01=sob / scale.clamp_min(1e-12))
    if "incompressible" in dataset_name:
        desc = getattr(dataset, "label_description", "")
        names = [name.strip() for group in re.findall(r"\[([^\]]+)\]", desc) for name in group.split(",")]
        if "u" not in names or "v" not in names:
            raise ValueError("NS physical metrics require named u,v output channels.")
        # Incompressible HDF5 uses x,y; transpose=True already switches to y,x.
        if not getattr(dataset, "transpose", False):
            p, y = p.transpose(-2, -1), y.transpose(-2, -1)
        dx, dy = 1.0 / p.shape[-1], 1.0 / p.shape[-2]
        def grads(field):
            gx = (field[..., 1:-1, 2:] - field[..., 1:-1, :-2]) / (2*dx)
            gy = (field[..., 2:, 1:-1] - field[..., :-2, 1:-1]) / (2*dy)
            return gx, gy
        ux, uy = grads(p[:, names.index("u")])
        vx, vy = grads(p[:, names.index("v")])
        ux_y, uy_y = grads(y[:, names.index("u")])
        vx_y, vy_y = grads(y[:, names.index("v")])
        div = _sample_mean((ux+vy).abs())
        vort = _sample_mean(((vx-uy)-(vx_y-uy_y)).abs())
        metrics.update(continuity=div,
                       rel_continuity=div/(_sample_mean(ux_y.abs())+_sample_mean(vy_y.abs())).clamp_min(1e-12),
                       vorticity=vort, rel_vorticity=vort/_sample_mean((vx_y-uy_y).abs()).clamp_min(1e-12))
    return metrics
