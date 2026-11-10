"""PTQ4ViT Linear Hessian-guided interval search and post-GELU twin grid.

The preserved upstream implementation is in reference/third_party/ptq4vit.
The active adaptation uses physical-regression gradients and selected Linear operations.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F
from quantization.progress import progress

def _ptq_weight(weight, scale, bits):
    if bits == 32:
        return weight
    qmax = 2 ** (bits - 1)
    return (weight / scale).round().clamp(-qmax, qmax - 1) * scale


def _ptq_input(value, scale, negative_scale, bits, twin):
    if bits == 32:
        return value
    qmax = 2 ** (bits - 1)
    if twin:
        return ((value / scale).round().clamp(0, qmax - 1) * scale
                + (value / negative_scale).round().clamp(-qmax, 0) * negative_scale)
    return (value / scale).round().clamp(-qmax, qmax - 1) * scale


@torch.no_grad()
def _ptq_search(layer, paths, weight_bits, activation_bits, twin, device,
                search_rounds=3, candidate_count=100):
    weight = layer.weight.detach()
    weight_scale = weight.abs().amax().clamp_min(1e-12) / (2 ** (weight_bits - 1) - 0.5) if weight_bits < 32 else weight.new_tensor(1.)
    maximum = weight.new_tensor(0.)
    gradient_energy = 0.0
    for path in progress(paths, desc="PTQ4ViT initialize ranges", leave=False):
        record = torch.load(path, map_location="cpu", weights_only=False)
        value = record["x"].to(device)
        maximum = torch.maximum(maximum, value.amax() if twin else value.abs().amax())
        gradient_energy += float(record["g"].double().square().sum())
    if gradient_energy == 0:
        raise ValueError("PTQ4ViT calibration gradients are all zero; choose nontrivial labeled calibration samples")
    activation_scale = maximum.clamp_min(1e-12) / (2 ** (activation_bits - 1) - 0.5) if activation_bits < 32 else weight.new_tensor(1.)
    # GELU's negative lobe is bounded by approximately -0.17, as in PTQ4ViT.
    negative_scale = weight.new_tensor(0.1699712425 / (2 ** (activation_bits - 1)))
    initial_weight_scale, initial_activation_scale = weight_scale.clone(), activation_scale.clone()
    factors = torch.linspace(0.01, 1.2, candidate_count, device=device)

    def score_candidates(candidates, search_weight):
        scores = torch.zeros(len(candidates), dtype=torch.float64, device=device)
        # Each record contains a complete loader batch. Never subsample tokens
        # or observations; candidate chunks keep GPU memory bounded.
        kind = "weight" if search_weight else "activation"
        for path in progress(paths, desc=f"PTQ4ViT {kind} search batches", leave=False):
            record = torch.load(path, map_location="cpu", weights_only=False)
            value, target, gradient = (record[key].to(device) for key in ("x", "y", "g"))
            for index, candidate in enumerate(progress(
                    candidates, desc=f"PTQ4ViT {kind} candidates", unit="candidate",
                    leave=False, delay=0.5)):
                ws = candidate if search_weight else weight_scale
                activation_step = activation_scale if search_weight else candidate
                quantized = F.linear(_ptq_input(value, activation_step, negative_scale, activation_bits, twin),
                                     _ptq_weight(weight, ws, weight_bits), layer.bias)
                scores[index] += ((quantized - target) * gradient).double().square().sum()
        return candidates[int(scores.argmin())]

    for _ in progress(range(search_rounds), desc="PTQ4ViT interval search",
                      unit="round", leave=False):
        if weight_bits < 32:
            weight_scale = score_candidates(initial_weight_scale * factors, True)
        if activation_bits < 32:
            activation_scale = score_candidates(initial_activation_scale * factors, False)
    return weight_scale, activation_scale, negative_scale, gradient_energy
