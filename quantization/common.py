"""Shared interfaces; method implementations never construct evaluation data."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass
class QuantizationResult:
    model: nn.Module
    metadata: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)


def select_quantizable_layers(model: nn.Module) -> dict[str, nn.Linear]:
    """Match inspect_layers' Linear subset, excluding norm/time conditioning."""
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and not any(part in name.lower() for part in ("norm", "time"))
    }


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def output_tensor(output):
    if torch.is_tensor(output):
        return output
    if hasattr(output, "output"):
        return output.output
    if isinstance(output, dict) and "output" in output:
        return output["output"]
    raise TypeError("Expected a prediction tensor or a ScOT output with .output")


def forward_model(model, batch, device):
    """Reconstruct FP outputs without asking ScOT to calculate supervised loss.

    ScOT uses labels to fill masked, nonpredicted channels. Those values are
    already known in the input for its PDE datasets, so restore them explicitly.
    """
    if torch.is_tensor(batch):
        return output_tensor(model(batch.to(device)))
    kwargs = {
        key: move_to_device(batch[key], device)
        for key in ("pixel_values", "time")
        if key in batch and batch[key] is not None
    }
    prediction = output_tensor(model(**kwargs))
    mask = batch.get("pixel_mask")
    if mask is not None:
        mask = mask.to(device=prediction.device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask[:, :prediction.shape[1], None, None]
        elif mask.ndim == prediction.ndim:
            mask = mask[:, :prediction.shape[1]]
        else:
            raise ValueError("Expected a channel or spatial ScOT pixel_mask")
        prediction = torch.where(mask, kwargs["pixel_values"][:, :prediction.shape[1]], prediction)
    return prediction


def resolve_device(requested: str) -> torch.device:
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --DEV cpu for CPU runs.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable; pass --DEV cpu.")
    return device
