"""Materialized DynamicQ weight grid and optional runtime activation QDQ."""
from __future__ import annotations
import torch
from quantization.common import QuantizationResult
from quantization.progress import progress
from quantization.layers import QuantizedLinear, _replace, _bits, _symmetric_weight, prepare_method


def quantize(model, calibration_loader, args, device, work_dir):
    model, layers, weight_bits, activation_bits, metadata = prepare_method(model, args, device, "dynamic")
    state = {}
    for name, layer in progress(layers.items(), desc="DynamicQ quantize layers", unit="layer"):
        weight, scale = _symmetric_weight(layer.weight, weight_bits, grid="dynamic_reference")
        mode = "dynamic" if activation_bits < 32 else "none"
        model = _replace(model, name, QuantizedLinear(layer, weight, activation_bits, mode))
        state[name] = {"weight_scale": scale.cpu(), "activation_scale": None}
    metadata.update(weight_scale_rule="per-output-channel symmetric max-abs",
                    quantization_grid="DynamicQ reference: step=(max_abs+1e-8)/(2^(bits-1)-1), signed symmetric QDQ",
                    activation_scale_rule="per-input-channel max-abs over each current input batch")
    return QuantizationResult(model.eval(), metadata, state)
