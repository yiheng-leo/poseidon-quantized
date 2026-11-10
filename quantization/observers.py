"""Full-calibration per-feature activation range observers."""
from __future__ import annotations
import torch
from .common import forward_model
from .progress import progress

@torch.no_grad()
def activation_ranges(model, layers, calibration, device):
    ranges = {}
    handles = []
    for name, layer in layers.items():
        def collect(module, inputs, name=name):
            value = inputs[0].detach().reshape(-1, module.in_features)
            if not torch.isfinite(value).all():
                raise ValueError(f"Non-finite calibration activations in {name}")
            low, high = value.amin(0), value.amax(0)
            if name in ranges:
                low = torch.minimum(low, ranges[name][0])
                high = torch.maximum(high, ranges[name][1])
            ranges[name] = (low, high)
        handles.append(layer.register_forward_pre_hook(collect))
    try:
        for batch in progress(calibration, desc="Collect activation ranges"):
            forward_model(model, batch, device)
    finally:
        for handle in handles:
            handle.remove()
    missing = set(layers) - set(ranges)
    if missing:
        raise ValueError(f"Calibration did not execute selected layers: {sorted(missing)}")
    return ranges
