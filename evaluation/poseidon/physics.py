"""Physical-metric helpers used by :mod:`evaluate`.

This follows Poseidon's ``poseidon_analysis/physical_loss1.py`` structure, but
keeps only the calculation and aggregation path needed by SAFE-SVD.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset

from .PDE_calc import CompressibleLoss, EllipticLoss, IncompressibleLoss, WaveLoss


DATASET_METRICS = {
    "incompressible": (
        "base_loss",
        "continuity",
        "momentum",
        "vorticity_err",
        "h1_error",
        "sobolev_s012",
        "sobolev_time",
        "sobolev_012",
    ),
    "compressible": (
        "base_loss",
        "mass_cons",
        "momentum_cons",
        "energy_cons",
        "h1_error",
        "sobolev_s012",
        "sobolev_time",
        "sobolev_012",
    ),
    "wave": (
        "base_loss",
        "speed_cons",
        "sobolev_s012",
        "sobolev_time",
        "sobolev_012",
    ),
    "elliptic": (
        "base_loss",
        "h1_error",
        "laplacian_error",
        "sobolev_s012",
        "sobolev_012",
    ),
}


class PrevTimeDataset(Dataset):
    """Parallel dataset returning the preceding time step for PDE residuals."""

    def __init__(self, base_dataset: Dataset) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> Any:
        _, _, time_start, time_end = self.base_dataset._idx_map(index)
        return self.base_dataset[index - 1] if time_end > time_start else self.base_dataset[index]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_dataset, name)


def count_linear_params(model: nn.Module) -> int:
    """Count parameters in ``nn.Linear`` modules, matching the original report."""
    return sum(
        parameter.numel()
        for module in model.modules()
        if isinstance(module, nn.Linear)
        for parameter in module.parameters()
    )


def get_dataset_category(dataset_name: str) -> str:
    """Map a Poseidon dataset name to its PDE evaluator."""
    name = dataset_name.lower()
    if "incompressible" in name:
        return "incompressible"
    if "compressible" in name:
        return "compressible"
    if "wave" in name:
        return "wave"
    if any(key in name for key in ("elliptic", "helmholtz", "poisson")):
        return "elliptic"
    return "incompressible"


def get_dataset_metrics(dataset_name: str) -> tuple[str, ...]:
    """Metrics shown in the concise terminal summary."""
    return DATASET_METRICS[get_dataset_category(dataset_name)]


def get_loss_calculator(
    dataset_name: str,
    preds: Any,
    labels: Any,
    prev_preds: Any = None,
    prev_labels: Any = None,
    dataset_obj: Any = None,
) -> Any:
    """Create the same PDE calculator and constants used by Poseidon."""
    base_dataset = getattr(dataset_obj, "dataset", dataset_obj)
    constants = getattr(base_dataset, "constants", None)
    if constants is None and hasattr(base_dataset, "mean") and hasattr(base_dataset, "std"):
        constants = {"mean": base_dataset.mean, "std": base_dataset.std}
    transpose = getattr(dataset_obj, "transpose", False)
    category = get_dataset_category(dataset_name)

    if category == "incompressible":
        return IncompressibleLoss(preds, labels, constants, Re=2500.0, transpose=transpose)
    if category == "compressible":
        mean_pressure = getattr(dataset_obj, "mean_pressure", 0.0)
        if "RiemannKelvinHelmholtz" in dataset_name and mean_pressure == 0.0:
            mean_pressure = 1.33
        return CompressibleLoss(
            preds,
            labels,
            prev_preds,
            prev_labels,
            constants,
            mean_pressure=mean_pressure,
            transpose=transpose,
            dt=1 / 20.0,
        )
    if category == "wave":
        return WaveLoss(
            preds,
            labels,
            prev_preds,
            prev_labels,
            constants,
            transpose=transpose,
            dt=1 / 15.0,
        )
    return EllipticLoss(preds, labels, constants, transpose=transpose)


def flatten_rollout(values: Any) -> Any:
    """Flatten Poseidon's optional trajectory/time dimensions."""
    return values.reshape(-1, *values.shape[2:]) if values.ndim == 5 else values


def evaluate_and_result(
    trainer: Any,
    test_ds: Any,
    dataset_name: str,
    rollout: Callable[..., tuple[Any, Any, dict[str, Any]]],
    physics_batch_size: int = 10,
) -> dict[str, Any]:
    """Run rollout and aggregate the original physical-loss metrics.

    Compressible and wave datasets additionally evaluate ``PrevTimeDataset``;
    that previous rollout supplies the temporal derivative used in the PDE
    residuals.
    """
    category = get_dataset_category(dataset_name)
    prev_preds = prev_labels = None
    if category in {"compressible", "wave"}:
        prev_preds, prev_labels, _ = rollout(trainer, PrevTimeDataset(test_ds))
        prev_preds = flatten_rollout(prev_preds)
        prev_labels = flatten_rollout(prev_labels)

    preds, labels, rollout_metrics = rollout(trainer, test_ds)
    preds = flatten_rollout(preds)
    labels = flatten_rollout(labels)

    metric_values: dict[str, list[float]] = {}
    for start in range(0, len(preds), physics_batch_size):
        stop = start + physics_batch_size
        calculator = get_loss_calculator(
            dataset_name=dataset_name,
            preds=preds[start:stop],
            labels=labels[start:stop],
            prev_preds=None if prev_preds is None else prev_preds[start:stop],
            prev_labels=None if prev_labels is None else prev_labels[start:stop],
            dataset_obj=test_ds,
        )
        for key, value in calculator.compute().items():
            if value is not None:
                metric_values.setdefault(key, []).append(float(value))

    result = {key: float(np.mean(values)) for key, values in metric_values.items()}
    result["base_loss"] = _as_scalar(rollout_metrics.get("_loss", 0))
    result["rel_base_loss"] = "-"
    result["model_size"] = _as_scalar(rollout_metrics.get("model size", 0))
    result["linear_size"] = count_linear_params(trainer.model)
    return result


def _as_scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value
