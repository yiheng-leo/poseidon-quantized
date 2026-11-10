"""Weight-only max-absolute QDQ using the original PPQ reference grid."""
from __future__ import annotations
import torch
from quantization.common import QuantizationResult
from quantization.progress import progress
from quantization.layers import QuantizedLinear, _replace, _bits, _symmetric_weight, prepare_method


def quantize(model, calibration_loader, args, device, work_dir):
    if _bits(getattr(args, "activation_bits", 32), "activation_bits") != 32:
        raise ValueError("uniform is the static weight-only max-abs baseline; use static or dynamic for activation quantization")
    model, layers, weight_bits, activation_bits, metadata = prepare_method(model, args, device, "uniform")
    state = {}
    for name, layer in progress(layers.items(), desc="UniformQ quantize weights", unit="layer"):
        weight, scale = _symmetric_weight(layer.weight, weight_bits, grid="ppq_reference")
        model = _replace(model, name, QuantizedLinear(layer, weight))
        state[name] = {"weight_scale": scale.cpu(), "activation_scale": None}
    metadata.update(weight_scale_rule="per-output-channel symmetric max-abs",
                    quantization_grid="PPQ.metrics reference: step=2*max_abs/(2^bits-1), round(W/step)*step without clipping")
    return QuantizationResult(model.eval(), metadata, state)
