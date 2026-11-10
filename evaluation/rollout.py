"""Autoregressive inference with ScOT input and mask conventions."""
from __future__ import annotations

import torch


def _prediction(output):
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "output"):
        return output.output
    if isinstance(output, dict) and "output" in output:
        return output["output"]
    raise TypeError("Model must return a Tensor or a ScOT output with .output.")


def _mask_for_output(mask, prediction):
    mask = mask.to(device=prediction.device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.ndim == 2:
        mask = mask[:, :prediction.shape[1], None, None]
    elif mask.ndim == 4:
        mask = mask[:, :prediction.shape[1]]
    else:
        raise ValueError("Expected a channel or spatial ScOT pixel_mask.")
    return torch.broadcast_to(mask, prediction.shape)


@torch.no_grad()
def autoregressive_predict(model, batch, *, device, ar_steps=1):
    """Roll out predictions; preserve masked/input-only channels from inputs.

    ScOT normally copies masked channels from labels in forward(). Here these
    nonpredicted channels are copied from the current input instead, so future
    labels cannot leak into rollout. Extra input-only channels follow ScOT's
    Trainer convention and are carried forward unchanged.
    """
    if ar_steps < 1:
        raise ValueError("ar_steps must be positive.")
    x = batch["pixel_values"].to(device)
    time = batch.get("time")
    if time is not None:
        time = time.to(device=device, dtype=x.dtype) / ar_steps
    elif ar_steps != 1:
        raise ValueError("A time-conditioned batch is required for autoregression.")
    mask = batch.get("pixel_mask")
    for _ in range(ar_steps):
        kwargs = {"pixel_values": x, "return_dict": True}
        if time is not None:
            kwargs["time"] = time
        pred = _prediction(model(**kwargs))
        if pred.ndim != 4 or pred.shape[0] != x.shape[0] or pred.shape[-2:] != x.shape[-2:]:
            raise ValueError("Model output must match the input batch and spatial shape.")
        if pred.shape[1] > x.shape[1]:
            raise ValueError("ScOT output has more channels than its input.")
        if mask is not None:
            pred = torch.where(_mask_for_output(mask, pred), x[:, :pred.shape[1]], pred)
        x = torch.cat((pred, x[:, pred.shape[1]:]), dim=1)
    return pred
