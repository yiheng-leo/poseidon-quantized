"""Disk-backed layer IO and physical-regression gradient capture."""
from __future__ import annotations
from pathlib import Path
import torch
from quantization.common import forward_model
from quantization.progress import progress

def _cache_ptq_layer(model, name, layer, calibration, directory, device):
    """Cache all layer IO and regression-loss gradients, a batch at a time."""
    paths, captured, gelu_outputs = [], [], set()
    post_gelu = []
    handles = []

    def mark_gelu(module, inputs, output):
        if torch.is_tensor(output):
            gelu_outputs.add(id(output))

    def capture(module, inputs, output):
        output.retain_grad()
        captured.append((inputs[0].detach(), output))
        post_gelu.append(id(inputs[0]) in gelu_outputs)

    for module in model.modules():
        if "gelu" in module.__class__.__name__.lower():
            handles.append(module.register_forward_hook(mark_gelu))
    handles.append(layer.register_forward_hook(capture))
    original_requires_grad = {parameter: parameter.requires_grad for parameter in model.parameters()}
    model.requires_grad_(True)
    try:
        for index, batch in enumerate(progress(calibration, desc=f"PTQ4ViT IO/gradients: {name}", leave=False)):
            if not isinstance(batch, dict) or "labels" not in batch:
                raise ValueError("PTQ4ViT's regression Hessian proxy requires calibration labels")
            model.zero_grad(set_to_none=True)
            captured.clear()
            gelu_outputs.clear()
            prediction = forward_model(model, batch, device)
            labels = batch["labels"].to(prediction)
            if prediction.shape != labels.shape:
                raise ValueError("PTQ4ViT calibration labels and predictions must have identical shapes")
            # Sum over samples gives each observation the same weight even when
            # the final loader batch is smaller than the others.
            loss = (prediction - labels).square().flatten(1).mean(1).sum()
            loss.backward()
            for invocation, (value, output) in enumerate(captured):
                if output.grad is None:
                    raise ValueError(f"Layer {name} is disconnected from the calibration loss")
                record = {"x": value.cpu(), "y": output.detach().cpu(), "g": output.grad.detach().cpu()}
                if not all(torch.isfinite(tensor).all() for tensor in record.values()):
                    raise ValueError(f"Non-finite PTQ4ViT calibration values in {name}")
                path = Path(directory) / f"layer-{index:06d}-{invocation}.pt"
                torch.save(record, path)
                paths.append(path)
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)
        for parameter, enabled in original_requires_grad.items():
            parameter.requires_grad_(enabled)
    if not paths:
        raise ValueError(f"Calibration did not execute selected layer {name}")
    return paths, bool(post_gelu) and all(post_gelu)
