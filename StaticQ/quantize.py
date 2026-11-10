"""Static Linear QDQ; this is not the archived QNNPACK integer backend."""
from __future__ import annotations
import torch
from quantization.common import QuantizationResult
from quantization.progress import progress
from quantization.layers import QuantizedLinear, _replace, _bits, _symmetric_weight, prepare_method
from quantization.cache import calibration_cache, record_calibration
from quantization.observers import activation_ranges


def quantize(model, calibration_loader, args, device, work_dir):
    model, layers, weight_bits, activation_bits, metadata = prepare_method(model, args, device, "static")
    state = {}
    with calibration_cache(calibration_loader, work_dir, required=activation_bits < 32) as (calibration, directory):
        record_calibration(metadata, calibration)
        ranges = activation_ranges(model, layers, calibration, device) if activation_bits < 32 else {}
        for name, layer in progress(layers.items(), desc="StaticQ quantize layers", unit="layer"):
            weight, weight_scale = _symmetric_weight(layer.weight, weight_bits)
            mode, scale = "none", None
            if activation_bits < 32:
                low, high = ranges[name]
                scale = torch.maximum(low.abs(), high.abs()).clamp_min(1e-12) / (2 ** (activation_bits - 1) - 1)
                mode = "symmetric"
            model = _replace(model, name, QuantizedLinear(layer, weight, activation_bits, mode, scale=scale))
            state[name] = {"weight_scale": weight_scale.cpu(), "activation_scale": None if scale is None else scale.cpu()}
    metadata.update(weight_scale_rule="per-output-channel symmetric max-abs",
                    quantization_grid="step=max_abs/(2^(bits-1)-1), signed symmetric QDQ",
                    activation_scale_rule="per-input-channel max-abs over the complete calibration set")
    return QuantizationResult(model.eval(), metadata, state)
