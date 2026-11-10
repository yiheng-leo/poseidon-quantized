"""RepQ Linear scale/shift adaptation with an explicit input affine.

Derived from the preserved RepQ-ViT implementation in reference/third_party/repq_vit.
No LayerNorm folding, attention MatMul or logarithmic softmax quantization is claimed.
"""
from __future__ import annotations
import copy
import torch
from torch import nn
from quantization.common import QuantizationResult
from quantization.progress import progress
from quantization.layers import QuantizedLinear, _replace, _bits, prepare_method
from quantization.cache import calibration_cache, record_calibration
from quantization.observers import activation_ranges as _activation_ranges

def _repq_weight(weight, bits):
    """RepQ percentile/MSE calibration, vectorized over output channels."""
    if bits == 32:
        return weight.detach().clone(), None
    source = weight.detach().float()
    quantiles = torch.tensor([0.001, 0.0001, 0.00001, 0.999, 0.9999, 0.99999], device=source.device)
    values = torch.quantile(source, quantiles, dim=1)
    low, high = values[:3], values[3:]
    scale = (high - low).clamp_min(1e-12) / (2 ** bits - 1)
    zero = (-low / scale).round()
    integer = (source[None] / scale[:, :, None]).round() + zero[:, :, None]
    candidates = (integer.clamp(0, 2 ** bits - 1) - zero[:, :, None]) * scale[:, :, None]
    best = (candidates - source[None]).square().mean(-1).argmin(0)
    index = torch.arange(source.shape[0], device=source.device)
    output = candidates[best, index]
    # Preserve constant channels exactly; their zero range has no quantization error.
    constant = source.amax(1) == source.amin(1)
    output[constant] = source[constant]
    return output.to(weight.dtype), {"scale": scale[best, index].cpu(), "zero_point": zero[best, index].cpu()}


def _repq(model, layers, calibration, weight_bits, activation_bits, device):
    if activation_bits == 32:
        raise ValueError("RepQ scale reparameterization requires activation_bits < 32; use uniform for weight-only quantization")
    ranges = _activation_ranges(model, layers, calibration, device)
    state = {}
    for name, layer in progress(layers.items(), desc="RepQ reparameterize layers", unit="layer"):
        low, high = ranges[name]
        # Full-calibration min/max replaces first-batch percentile initialization
        # in the old script, so every supplied sample contributes to the scales.
        delta = (high - low).clamp_min(1e-8) / (2 ** activation_bits - 1)
        zp = (-low / delta).round()
        target_delta, target_zp = delta.mean(), zp.mean()
        ratio = delta / target_delta
        shift = (-zp * delta) / ratio + target_zp * target_delta
        transformed = copy.deepcopy(layer)
        with torch.no_grad():
            transformed.weight.mul_(ratio[None])
            compensation = transformed.weight @ shift
            if transformed.bias is None:
                transformed.bias = nn.Parameter(compensation, requires_grad=False)
            else:
                transformed.bias.add_(compensation)
        weight, weight_state = _repq_weight(transformed.weight, weight_bits)
        replacement = QuantizedLinear(transformed, weight, activation_bits, "affine",
                                      scale=target_delta, zero_point=target_zp,
                                      input_ratio=ratio, input_shift=shift)
        model = _replace(model, name, replacement)
        state[name] = {"input_ratio": ratio.cpu(), "input_shift": shift.cpu(),
                       "activation_scale": target_delta.cpu(), "activation_zero_point": target_zp.cpu(),
                       "weight_quantizer": weight_state}
    return model, state



def quantize(model, calibration_loader, args, device, work_dir):
    if _bits(getattr(args, "activation_bits", 32), "activation_bits") == 32:
        raise ValueError("RepQ requires activation quantization (activation_bits < 32)")
    model, layers, weight_bits, activation_bits, metadata = prepare_method(model, args, device, "repq")
    with calibration_cache(calibration_loader, work_dir) as (calibration, directory):
        record_calibration(metadata, calibration)
        model, state = _repq(model, layers, calibration, weight_bits, activation_bits, device)
    metadata.update(algorithm="RepQ scale/shift reparameterization with explicit input affine",
                    activation_calibration="full-calibration per-input-channel min/max",
                    norm_folding=False, attention_matmul_quantized=False, softmax_quantized=False)
    return QuantizationResult(model.eval(), metadata, state)
