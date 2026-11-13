"""Poseidon BRECQ: MSE scales, resumable AdaRound and materialized weights."""
from quantization.common import QuantizationResult
from quantization.layers import _bits, prepare_method
from .resume import run


def quantize(model, calibration_loader, args, device, work_dir):
    weight_bits = _bits(getattr(args, "weight_bits", 8), "weight_bits")
    activation_bits = _bits(getattr(args, "activation_bits", 32), "activation_bits")
    if activation_bits != 32 or not 2 <= weight_bits <= 8:
        raise ValueError("BRECQ supports weight_bits=2..8 and activation_bits=32")
    model, layers, weight_bits, activation_bits, metadata = prepare_method(model, args, device, "brecq")
    model, state, details = run(model, layers, calibration_loader, args, weight_bits,
                                activation_bits, device, work_dir, metadata)
    metadata.update(details)
    metadata["algorithm"] = "BRECQ block reconstruction with AdaRound; materialized QDQ weights"
    return QuantizationResult(model.eval(), metadata, state)
