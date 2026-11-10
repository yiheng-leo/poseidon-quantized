"""SAPQ likelihood and priors, preserving the current numerical conventions.

Global/output, block/output, block/local and layer/local remain separate modes:
output reductions, MC estimators and outer prior multipliers are not unified.
"""
import torch
from quantization.probabilistic import _EPS, _backward_likelihood, compute_mdl_prior


def _prior(steps, ranges, sensitivity, args):
    mode = args.prior_mode
    if mode == "ppq":
        return compute_mdl_prior({name: (value, None) for name, value in steps.items()},
                                 ranges, gamma=args.gamma)
    if mode not in {"block_sens", "block_no_sens"}:
        raise ValueError(f"Unknown SAPQ prior mode: {mode}")
    prior = next(iter(steps.values())).new_zeros(())
    for name, step in steps.items():
        bits = torch.log2(ranges[name]["weight_ranges"].clamp_min(_EPS) / step.clamp_min(_EPS))
        sens = sensitivity[name].to(step.device) if mode == "block_sens" else torch.zeros_like(step)
        sigma = (args.sigma0 * (1.0 + args.alpha * sens)).clamp_min(_EPS)
        prior = prior + ((bits - args.target_bits).square() / (2.0 * sigma.square())).sum()
    return args.prior_scale * prior


def backward_likelihood(forward, target, args, device, *, fisher=None):
    return _backward_likelihood(
        forward, target, args, device,
        marginal=args.method in {"sapq", "sapq-block", "sapq-network"},
        fisher=fisher, output_sum=args.method == "sapq-network",
    )
