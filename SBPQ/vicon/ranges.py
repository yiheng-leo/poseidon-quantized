"""Per-output-channel empirical weight ranges for VICON SBPQ."""
import torch

from quantization.progress import progress


@torch.no_grad()
def compute_weight_ranges(layers, percentile=1e-4):
    ranges = {}
    for name, layer in progress(layers.items(), desc="VICON weight ranges", unit="layer"):
        weight = layer.weight.detach().float()
        value = (torch.quantile(weight, 1 - percentile, dim=1)
                 - torch.quantile(weight, percentile, dim=1)).clamp_min(1e-12)
        ranges[name] = {"weight_ranges": value.cpu()}
    return ranges
