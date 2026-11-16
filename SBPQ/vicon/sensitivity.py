"""Mean squared block-output gradients over VICON calibration batches."""
import torch

from quantization.common import move_to_device
from quantization.progress import progress
from .model_utils import model_forward
from .sobolev import sensitivity_loss


def compute_sensitivity(model, batches, blocks, args, device):
    totals = {name: torch.zeros((), dtype=torch.float64) for name in blocks}
    activations, handles = {}, []
    for name, module in blocks.items():
        def capture(_module, _inputs, output, key=name):
            activations[key] = output
        handles.append(module.register_forward_hook(capture))
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    count = 0
    try:
        for source in progress(batches, desc="VICON Sobolev sensitivity", unit="batch"):
            batch = move_to_device(source, device)
            activations.clear()
            output = model_forward(model, batch, device)
            loss = sensitivity_loss(output, batch, args)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite VICON sensitivity loss")
            gradients = torch.autograd.grad(loss, tuple(activations[name] for name in blocks))
            for name, gradient in zip(blocks, gradients):
                totals[name] += gradient.detach().float().square().mean().cpu().double()
            count += 1
            activations.clear()
            del output, loss, gradients, gradient, batch
        if not count:
            raise ValueError("VICON sensitivity requires calibration batches")
        return {name: (total / count).float() for name, total in totals.items()}
    finally:
        activations.clear()
        for handle in handles:
            handle.remove()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None

