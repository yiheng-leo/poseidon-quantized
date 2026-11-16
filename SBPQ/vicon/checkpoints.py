# Adapted from SafeSVD; see VENDORED.md and LICENSE-SafeSVD.txt.
"""Checkpoint loading and reporting for this standalone evaluator."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TaggedBatch:
    """One raw batch and its physics/dataset category."""

    category: str
    payload: Any


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve a requested device without silently changing experiment semantics."""
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu.")
    return target


def get_module_by_name(model: nn.Module, dotted_name: str) -> nn.Module:
    """Look up a module path, including ``ModuleList``/``Sequential`` indices."""
    if not dotted_name:
        return model
    try:
        return model.get_submodule(dotted_name)
    except AttributeError:  # pragma: no cover - older PyTorch fallback
        module = model
        for part in dotted_name.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        return module


def set_module_by_name(model: nn.Module, dotted_name: str, replacement: nn.Module) -> None:
    """Replace a module path while supporting numerical container components."""
    parent_name, separator, leaf = dotted_name.rpartition(".")
    parent = get_module_by_name(model, parent_name) if separator else model
    if leaf.isdigit():
        parent[int(leaf)] = replacement  # type: ignore[index]
    else:
        setattr(parent, leaf, replacement)


def load_state_dict_payload(path: Path) -> Mapping[str, torch.Tensor]:
    """Read trusted SAFE-SVD state dicts or research full-model checkpoints."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older PyTorch
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, nn.Module):
        # MPP MySVD and safe_svd.main save the complete compressed model.
        # Rebuild its low-rank layers on the current evaluator architecture.
        return payload.state_dict()
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path} does not contain a state dictionary payload")
    state_dict = payload.get("state_dict", payload.get("model_state_dict", payload))
    if not isinstance(state_dict, Mapping) or not all(torch.is_tensor(value) for value in state_dict.values()):
        raise TypeError(f"{path} does not contain a tensor state dictionary")
    return state_dict  # type: ignore[return-value]


def rebuild_low_rank_topology(model: nn.Module, state_dict: Mapping[str, torch.Tensor]) -> nn.Module:
    """Replace base Linear layers for ``<layer>.0/.1`` SAFE-SVD state keys."""
    targets: list[tuple[str, nn.Sequential]] = []
    for key, first_weight in state_dict.items():
        if not key.endswith(".0.weight"):
            continue
        layer_name = key.removesuffix(".0.weight")
        second_weight = state_dict.get(f"{layer_name}.1.weight")
        if not torch.is_tensor(second_weight):
            continue
        try:
            original = get_module_by_name(model, layer_name)
        except (AttributeError, KeyError):
            continue
        if not isinstance(original, nn.Linear):
            continue
        rank = first_weight.shape[0]
        replacement = nn.Sequential(
            nn.Linear(first_weight.shape[1], rank, bias=False),
            nn.Linear(rank, second_weight.shape[0], bias=f"{layer_name}.1.bias" in state_dict),
        )
        targets.append((layer_name, replacement))
    for name, replacement in targets:
        set_module_by_name(model, name, replacement)
    return model


def load_compressed_state(model: nn.Module, path: Path, *, strict: bool = True) -> nn.Module:
    """Rebuild low-rank layers and load a portable SAFE-SVD checkpoint."""
    state_dict = load_state_dict_payload(path)
    rebuild_low_rank_topology(model, state_dict)
    model.load_state_dict(state_dict, strict=strict)
    return model


def model_parameter_report(model: nn.Module) -> dict[str, int]:
    """Return total and Linear-only parameter counts for every evaluator."""
    return {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "linear_parameters": sum(
            parameter.numel()
            for layer in model.modules()
            if isinstance(layer, nn.Linear)
            for parameter in layer.parameters()
        ),
    }


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path
