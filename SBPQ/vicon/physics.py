# Adapted from SafeSVD/VICON; see VENDORED.md and LICENSE-*.txt in this directory.
"""Paper-compatible VICON physical metrics for the unified evaluator."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


SOBOLEV_ORDERS = {"NS2D": 2, "COMPRESSIBLE2D": 1, "EULER2D": 1}


def sobolev_metric(
    prediction: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    *,
    order: int,
    dx: float = 1 / 128,
    dy: float = 1 / 128,
) -> tuple[float, float]:
    """VICON's finite-difference L1 Sobolev error and its relative form."""
    pred = _numpy(prediction)
    label = _numpy(target)
    total = float(np.mean(np.abs(pred - label)))
    norm = float(np.mean(np.abs(label)))
    predictions = [pred]
    labels = [label]
    for current_order in range(1, order + 1):
        next_predictions: list[np.ndarray] = []
        next_labels: list[np.ndarray] = []
        error = 0.0
        reference = 0.0
        for current_prediction, current_label in zip(predictions, labels):
            pred_x = (current_prediction[..., :, 1:] - current_prediction[..., :, :-1]) / dx
            label_x = (current_label[..., :, 1:] - current_label[..., :, :-1]) / dx
            pred_y = (current_prediction[..., 1:, :] - current_prediction[..., :-1, :]) / dy
            label_y = (current_label[..., 1:, :] - current_label[..., :-1, :]) / dy
            error += float(np.mean(np.abs(pred_x - label_x)) + np.mean(np.abs(pred_y - label_y)))
            reference += float(np.mean(np.abs(label_x)) + np.mean(np.abs(label_y)))
            if current_order < order:
                next_predictions.extend((pred_x, pred_y))
                next_labels.extend((label_x, label_y))
        total += error
        norm += reference
        predictions, labels = next_predictions, next_labels
    return total, total / (norm + 1e-12)


def physics_metrics(dataset_type: str, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compute the VICON test metrics for one denormalised output batch."""
    order = SOBOLEV_ORDERS.get(dataset_type, 1)
    if dataset_type == "NS2D":
        return _ns2d_metrics(prediction, target, order=order)
    if dataset_type in {"COMPRESSIBLE2D", "EULER2D"}:
        sobolev, relative = sobolev_metric(prediction[:, :, [0, 1, 2, 3]], target[:, :, [0, 1, 2, 3]], order=order)
        return {"sobolev": sobolev, "rel_sobolev": relative}
    sobolev, relative = sobolev_metric(prediction, target, order=order)
    return {"sobolev": sobolev, "rel_sobolev": relative}


def average_metrics(values: list[Mapping[str, float]]) -> dict[str, float]:
    """Average a list of homogeneous metric dictionaries."""
    if not values:
        return {}
    return {key: float(np.mean([value[key] for value in values if key in value])) for key in values[0]}


def _ns2d_metrics(prediction: torch.Tensor, target: torch.Tensor, *, order: int) -> dict[str, float]:
    pred = _numpy(prediction)
    label = _numpy(target)
    velocity_x, velocity_y = pred[:, :, 1], pred[:, :, 2]
    label_x, label_y = label[:, :, 1], label[:, :, 2]
    dx = dy = 1 / 128

    d_velocity_x_dx = (velocity_x[..., :, 2:] - velocity_x[..., :, :-2]) / (2 * dx)
    d_velocity_y_dy = (velocity_y[..., 2:, :] - velocity_y[..., :-2, :]) / (2 * dy)
    divergence = d_velocity_x_dx[..., 1:-1, :] + d_velocity_y_dy[..., :, 1:-1]
    continuity = float(np.mean(np.abs(divergence)))

    d_label_x_dx = (label_x[..., :, 2:] - label_x[..., :, :-2]) / (2 * dx)
    d_label_y_dy = (label_y[..., 2:, :] - label_y[..., :-2, :]) / (2 * dy)
    velocity_scale = float(
        np.mean(np.abs(d_label_x_dx[..., 1:-1, :]))
        + np.mean(np.abs(d_label_y_dy[..., :, 1:-1]))
        + 1e-12
    )

    d_velocity_y_dx = (velocity_y[..., :, 2:] - velocity_y[..., :, :-2]) / (2 * dx)
    d_velocity_x_dy = (velocity_x[..., 2:, :] - velocity_x[..., :-2, :]) / (2 * dy)
    curl = d_velocity_y_dx[..., 1:-1, :] - d_velocity_x_dy[..., :, 1:-1]
    d_label_y_dx = (label_y[..., :, 2:] - label_y[..., :, :-2]) / (2 * dx)
    d_label_x_dy = (label_x[..., 2:, :] - label_x[..., :-2, :]) / (2 * dy)
    label_curl = d_label_y_dx[..., 1:-1, :] - d_label_x_dy[..., :, 1:-1]
    vorticity_error = float(np.mean(np.abs(curl - label_curl)))
    sobolev, relative_sobolev = sobolev_metric(pred[:, :, [1, 2, 5]], label[:, :, [1, 2, 5]], order=order)
    return {
        "sobolev": sobolev,
        "rel_sobolev": relative_sobolev,
        "continuity": continuity,
        "rel_continuity": continuity / velocity_scale,
        "vorticity_err": vorticity_error,
        "rel_vorticity_err": vorticity_error / (float(np.mean(np.abs(label_curl))) + 1e-12),
    }


def _numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    return value.detach().cpu().float().numpy() if isinstance(value, torch.Tensor) else value
