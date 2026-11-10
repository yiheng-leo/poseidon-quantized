"""Compact evaluation reports: final weight bits, metric values and changes."""
from __future__ import annotations

import math
from numbers import Real
from typing import Any, Mapping


# Bookkeeping entries returned by the SAFE-SVD calculator are not error metrics.
_EXCLUDED_METRICS = {"model_size", "linear_size", "rel_base_loss"}


def final_average_weight_bits(artifact_metadata: Mapping[str, Any]) -> float | None:
    """Read achieved bits for quantized weights, never the requested bit budget.

    PPQ/SAPQ save their parameter-weighted final mean in avg_weight_bits.
    Fixed-bit baselines save weight_bits. Materialized QDQ tensors alone cannot
    identify the original quantizer, so missing metadata yields null.
    """
    quantization = artifact_metadata.get("quantization", {})
    for key in ("avg_weight_bits", "effective_bits", "weight_bits"):
        value = quantization.get(key)
        if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value):
            return float(value)
    return None


def _metric_values(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if key not in _EXCLUDED_METRICS and isinstance(value, Real) and not isinstance(value, bool)
    }


def get_metric_increases(
    metrics: Mapping[str, Any], origin_metrics: Mapping[str, Any],
) -> dict[str, float | None]:
    """Percent change from origin; a zero baseline has no defined percentage."""
    origin = _metric_values(origin_metrics)
    return {
        key: round(100.0 * (value - origin[key]) / origin[key], 6) if origin[key] != 0 else None
        for key, value in _metric_values(metrics).items() if key in origin
    }


def build_results(*, avg_weight_bits: float | None,
                  origin_metrics: Mapping[str, Any] | None = None,
                  compressed_metrics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Keep only the requested final mean bits and one record per metric."""
    origin = _metric_values(origin_metrics or {})
    compressed = _metric_values(compressed_metrics or {})
    increases = get_metric_increases(compressed, origin)
    metrics = {}
    for key in dict.fromkeys([*origin, *compressed]):
        values = {}
        if key in origin:
            values["origin"] = origin[key]
        if key in compressed:
            values["compressed"] = compressed[key]
        if key in increases:
            values["change_percent"] = increases[key]
        metrics[key] = values
    return {"avg_weight_bits": avg_weight_bits, "metrics": metrics}
