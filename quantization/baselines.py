"""Legacy full-model pickle path. New methods import quantization.layers directly."""
from .layers import QuantizedLinear

__all__ = ["QuantizedLinear"]
