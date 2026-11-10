"""Calibration-only structure sensitivity; retains the legacy Sobolev scale."""
import torch
import torch.nn.functional as F
from quantization.common import forward_model
from quantization.progress import progress
from quantization.probabilistic import (
    _EPS, _first_tensor, _load, _reference_hooks, _requires_input_grad,
)


def _structure_loss(pred, clean, mode, constants, dataset_name):
    if mode == "mse":
        return F.mse_loss(pred, clean)
    if pred.ndim != 4:
        raise ValueError(f"{mode} sensitivity requires [B,C,H,W] predictions.")
    if mode == "divergence":
        # Matches the original SAPQ sensitivity convention: divergence in model-output units.
        channels = pred.shape[1]
        u, v = (1, 2) if channels >= 4 else (0, 1)
        if channels < 2 or min(pred.shape[-2:]) < 3:
            raise ValueError("Divergence sensitivity requires two velocity channels and H,W >= 3.")

        def divergence(field):
            dx, dy = 1.0 / (field.shape[-1] - 1), 1.0 / (field.shape[-2] - 1)
            return ((field[:, u, 1:-1, 2:] - field[:, u, 1:-1, :-2]) / (2 * dx)
                    + (field[:, v, 2:, 1:-1] - field[:, v, :-2, 1:-1]) / (2 * dy))

        return F.mse_loss(pred, clean) + F.mse_loss(divergence(pred), divergence(clean))
    if mode != "sobolev":
        raise ValueError(f"Unknown sensitivity mode: {mode}")
    # Preserve the existing discrete H1 convention: forward differences without
    # division by grid spacing. Constant means cancel in prediction differences.
    if constants is not None:
        std = torch.as_tensor(constants["std"], device=pred.device, dtype=pred.dtype)
        if std.numel() not in (1, pred.shape[1]):
            raise ValueError("Legacy Sobolev sensitivity std does not cover the output channels; "
                             "use mse sensitivity for this dataset or supply matching constants.")
        if std.numel() > 1:
            std = std.view(1, -1, 1, 1)
        pred, clean = pred * std, clean * std
    if ("incompressible" in dataset_name.lower() or dataset_name.lower().startswith("ns")):
        if pred.shape[1] == 4:
            pred, clean = pred[:, 1:], clean[:, 1:]
    if min(pred.shape[-2:]) < 2:
        raise ValueError("Sobolev sensitivity requires H,W >= 2.")
    return (F.mse_loss(pred, clean)
            + F.mse_loss(pred[..., 1:] - pred[..., :-1], clean[..., 1:] - clean[..., :-1])
            + F.mse_loss(pred[..., 1:, :] - pred[..., :-1, :],
                         clean[..., 1:, :] - clean[..., :-1, :]))


def _channel_sensitivity(grad):
    dims = (0, 2, 3) if grad.ndim == 4 else tuple(range(grad.ndim - 1))
    return grad.square().mean(dim=dims)


def _sensitivity(model, units, layers, reference_steps, paths, device, args, constants,
                 *, forward_fn=forward_model):
    """One backward pass per batch captures every block, keeping only channel sums."""
    totals, counts = {}, {}
    for path in progress(paths, desc=f"SAPQ {args.sensitivity} sensitivity"):
        record = _load(path)
        captured, handles = {}, []
        try:
            batch = _requires_input_grad(record["batch"], device)
            with _reference_hooks(layers, reference_steps):
                # Capture after the weight hook when a unit is itself Linear.
                for unit in units:
                    def capture(mod, inputs, output, name=unit.name):
                        if name in captured:
                            raise ValueError(f"Repeated reconstruction unit {name!r} is unsupported.")
                        captured[name] = _first_tensor(output)
                    handles.append(unit.module.register_forward_hook(capture))
                pred = forward_fn(model, batch, device)
            loss = _structure_loss(pred, record["clean"].to(device), args.sensitivity,
                                   constants, getattr(args, "dataset", ""))
            active = [(name, tensor) for name, tensor in captured.items() if tensor.requires_grad]
            if not active:
                raise ValueError("No differentiable reconstruction outputs were captured.")
            grads = torch.autograd.grad(loss, [value for _, value in active], allow_unused=True)
            for (name, _), grad in zip(active, grads):
                if grad is None:
                    continue
                value = _channel_sensitivity(grad).detach().cpu()
                totals[name] = totals.get(name, torch.zeros_like(value)) + value
                counts[name] = counts.get(name, 0) + 1
        finally:
            for handle in handles:
                handle.remove()
    raw, normalized = {}, {}
    for unit in units:
        if unit.name not in totals:
            raise ValueError(f"No sensitivity gradient reached reconstruction unit {unit.name!r}.")
        sens = totals[unit.name] / counts[unit.name]
        norm = (sens - sens.min()) / (sens.max() - sens.min() + _EPS)
        for name, layer in unit.layers.items():
            raw[name] = sens.clone() if sens.numel() == layer.out_features else sens.mean().repeat(layer.out_features)
            normalized[name] = norm.clone() if norm.numel() == layer.out_features else norm.mean().repeat(layer.out_features)
    return raw, normalized
