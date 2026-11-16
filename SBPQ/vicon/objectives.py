"""VICON masks, physical Sobolev sensitivity and SBPQ's MC reconstruction."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from SBPQ.noise import add_weight_quantization_noise
from quantization.common import move_to_device
from quantization.progress import progress
from .model_utils import model_forward


def query_fields(output, batch):
    return output[:, batch["min_ex"]:, :-1]


def valid_mask(batch):
    return ((batch["labels"][:, batch["min_ex"]:, -1:] == 0)
            * batch["channel_mask"][:, :, :-1]).float()


def masked_mean(error, mask):
    mask = mask.expand_as(error)
    return (error * mask).sum() / mask.sum().clamp_min(1)


def sensitivity_loss(output, batch, args):
    prediction = query_fields(output, batch)
    reference = query_fields(batch["labels"], batch)
    mask = valid_mask(batch)
    if not mask.any():
        raise ValueError("VICON sensitivity batch has no valid interior query cells")
    task = masked_mean((prediction - reference).square(), mask)
    # Restore physical units before comparing spatial derivatives. Match SBPQ:
    # all (dx,dy) of each total order, averaged within that order (including xy).
    error = (prediction - reference) * batch["std"][:, :, :-1]
    if args.sobolev_transpose:
        error, mask = error.transpose(-2, -1), mask.transpose(-2, -1)
    if args.sobolev_order >= min(error.shape[-2:]):
        raise ValueError("VICON Sobolev order must be smaller than the output spatial dimensions")
    weights = args.sobolev_order_weights
    if isinstance(weights, (int, float)):
        weights = [weights] * (args.sobolev_order + 1)
    sobolev = error.new_zeros(())
    for order in range(args.sobolev_order + 1):
        order_loss = error.new_zeros(())
        for x_order in range(order + 1):
            derivative, current_mask = error, mask
            for axis, count in ((-1, x_order), (-2, order - x_order)):
                for _ in range(count):
                    derivative = torch.diff(derivative, dim=axis)
                    current_mask = torch.minimum(current_mask.narrow(axis, 1, current_mask.shape[axis] - 1),
                                                 current_mask.narrow(axis, 0, current_mask.shape[axis] - 1))
            discrepancy = derivative.abs() if args.sobolev_norm == "l1" else derivative.square()
            order_loss = order_loss + masked_mean(discrepancy, current_mask)
        sobolev = sobolev + weights[order] * order_loss / (order + 1)
    return args.task_loss_weight * task + args.sobolev_loss_weight * sobolev


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


def noisy_forward(model, batch, step_sizes, device):
    handles = []
    try:
        for name, step in step_sizes.items():
            module = model.get_submodule(name)
            if not isinstance(module, nn.Linear) or step.numel() != module.out_features:
                raise ValueError(f"Invalid VICON noisy Linear/step size: {name}")
            def hook(layer, inputs, _output, value=step):
                weight = add_weight_quantization_noise(layer.weight, value, channel_axis=0)
                return F.linear(inputs[0], weight, layer.bias)
            handles.append(module.register_forward_hook(hook))
        return query_fields(model_forward(model, batch, device), batch)
    finally:
        for handle in handles:
            handle.remove()


class ViconNetworkLikelihood(nn.Module):
    """Same sum-squared log-mean-exp SBPQ energy, over valid VICON queries."""

    def __init__(self, model, frozen_batches, clean_network_outputs, num_mc_samples=10,
                 eta=1e-6, device="cuda"):
        super().__init__()
        self.model, self.batches, self.outputs = model, frozen_batches, clean_network_outputs
        if num_mc_samples < 1 or eta <= 0:
            raise ValueError("VICON MC samples and eta must be positive")
        self.num_mc_samples, self.eta, self.device = num_mc_samples, eta, device

    def forward(self, step_sizes, batch_index, memory_efficient=True):
        batch = move_to_device(self.batches[batch_index], self.device)
        clean = self.outputs[batch_index].to(self.device)
        mask = valid_mask(batch)
        names, steps = tuple(step_sizes), tuple(step_sizes.values())
        def score(*values):
            prediction = noisy_forward(self.model, batch, dict(zip(names, values)), self.device)
            energy = ((prediction - clean).square() * mask).flatten(1).sum(1)
            return -energy / (2 * self.eta)
        scores = [checkpoint(score, *steps, use_reentrant=False, preserve_rng_state=True)
                  if memory_efficient and torch.is_grad_enabled() else score(*steps)
                  for _ in range(self.num_mc_samples)]
        return -(torch.logsumexp(torch.stack(scores), dim=0) - math.log(self.num_mc_samples)).sum()
