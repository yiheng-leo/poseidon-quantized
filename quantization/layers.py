"""Portable QDQ layers, unchanged numerical grids, and shared method preparation."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from .common import select_quantizable_layers

def _replace(model, name, module):
    if not name:
        return module
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, module)
    return model


def _bits(value, name):
    value = int(value)
    if value != 32 and not 2 <= value <= 16:
        raise ValueError(f"{name} must be 2..16 or 32 (unquantized), got {value}")
    return value


def _symmetric_weight(weight, bits, *, grid="symmetric"):
    if bits == 32:
        return weight.detach().clone(), torch.ones(weight.shape[0], device=weight.device)
    maximum = weight.detach().abs().amax(dim=1)
    if grid == "ppq_reference":
        # Preserve PPQ.metrics.compute_dynamic_stepsizes/evaluate_with_stepsizes:
        # the grid has 2^b - 1 intervals across [-max, max], with no clipping.
        scale = (2 * maximum / (2 ** bits - 1)).clamp_min(1e-12)
        return ((weight / scale[:, None]).round() * scale[:, None]).detach(), scale
    qmax = 2 ** (bits - 1) - 1
    # DynamicQ adds epsilon before division, rather than clamping the maximum.
    maximum = maximum + 1e-8 if grid == "dynamic_reference" else maximum.clamp_min(1e-12)
    scale = maximum / qmax
    quantized = (weight / scale[:, None]).round().clamp(-qmax, qmax) * scale[:, None]
    return quantized.detach(), scale


class QuantizedLinear(nn.Linear):
    """Materialized QDQ weights with optional runtime input quantization.

    Buffers (including the RepQ input transform) participate in .to() and
    state_dict(). Full-module artifacts resolve this stable repository class.
    """

    def __init__(self, source, weight, activation_bits=32, input_mode="none",
                 scale=None, zero_point=None, negative_scale=None,
                 input_ratio=None, input_shift=None, signed_full_range=False):
        super().__init__(source.in_features, source.out_features, source.bias is not None,
                         device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            self.weight.copy_(weight)
            if self.bias is not None:
                self.bias.copy_(source.bias)
        self.requires_grad_(False)
        self.activation_bits = int(activation_bits)
        self.input_mode = input_mode
        self.signed_full_range = bool(signed_full_range)
        for name, value, default in (
            ("input_scale", scale, 1.0), ("input_zero_point", zero_point, 0.0),
            ("negative_scale", negative_scale, 1.0), ("input_ratio", input_ratio, 1.0),
            ("input_shift", input_shift, 0.0),
        ):
            value = torch.as_tensor(default if value is None else value,
                                    device=weight.device, dtype=weight.dtype).detach().clone()
            self.register_buffer(name, value)

    def quantize_input(self, value):
        value = value / self.input_ratio - self.input_shift
        if self.activation_bits == 32 or self.input_mode == "none":
            return value
        qmax = 2 ** (self.activation_bits - 1) - 1
        qmin = -qmax - int(self.signed_full_range)
        if self.input_mode == "dynamic":
            scale = (value.reshape(-1, self.in_features).abs().amax(0) + 1e-8) / qmax
            return (value / scale).round().clamp(-qmax, qmax) * scale
        if self.input_mode == "symmetric":
            return (value / self.input_scale).round().clamp(qmin, qmax) * self.input_scale
        if self.input_mode == "affine":
            integer = (value / self.input_scale).round() + self.input_zero_point
            return (integer.clamp(0, 2 ** self.activation_bits - 1) - self.input_zero_point) * self.input_scale
        if self.input_mode == "twin":
            positive = (value / self.input_scale).round().clamp(0, qmax) * self.input_scale
            negative = (value / self.negative_scale).round().clamp(-qmax - 1, 0) * self.negative_scale
            return positive + negative
        raise RuntimeError(f"Unknown input quantization mode: {self.input_mode}")

    def forward(self, value):
        return F.linear(self.quantize_input(value), self.weight, self.bias)


def prepare_method(model, args, device, method):
    """Shared validation and metadata; this function performs no algorithm dispatch."""
    weight_bits = _bits(getattr(args, "weight_bits", 8), "weight_bits")
    activation_bits = _bits(getattr(args, "activation_bits", 32), "activation_bits")
    torch.manual_seed(int(getattr(args, "seed", 0)))
    model = model.to(device).eval()
    layers = select_quantizable_layers(model)
    if not layers:
        raise ValueError("The model has no eligible Linear layers")
    for name, layer in layers.items():
        if not torch.isfinite(layer.weight).all():
            raise ValueError(f"Non-finite weights in {name}")
    metadata = {"method": method, "weight_bits": weight_bits, "activation_bits": activation_bits,
                "execution": "fake_quantization", "packed_integer_weights": False,
                "scope": "selected Linear", "quantized_layers": list(layers),
                "calibration_samples": 0, "calibration_batches": 0}
    return model, layers, weight_bits, activation_bits, metadata
