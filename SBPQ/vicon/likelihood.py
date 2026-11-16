"""VICON network-wise SBPQ Monte Carlo reconstruction likelihood."""
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from SBPQ.noise import add_weight_quantization_noise
from quantization.common import move_to_device
from .model_utils import model_forward
from .sobolev import query_fields, valid_mask


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

