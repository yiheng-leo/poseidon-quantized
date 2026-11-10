"""PPQ objectives: mean Monte Carlo reconstruction and bit-budget priors."""
import torch
import torch.nn.functional as F
from quantization.probabilistic import _backward_likelihood, compute_mdl_prior


def prior_weighted_avg_bits_cap_poseidon(
    step_sizes_dict,
    ranges_dict,
    channel_weights,
    target_bits: float = 4.0,
    lam: float = 1.0,
    alpha: float = 10.0,
    eps: float = 1e-8,
):
    """
    Smooth penalty if parameter-weighted average bits exceeds target_bits.

    penalty = lam * softplus(alpha * (avg_bits - target_bits))^2
    """
    total_bits_weighted = None
    total_weight = None

    for name, (w_step, _a_step) in step_sizes_dict.items():
        rec = ranges_dict.get(name, None)
        if rec is None or "weight_ranges" not in rec or w_step is None:
            continue

        w_range = rec["weight_ranges"].to(w_step.device)
        bits = torch.log2((w_range + eps) / (w_step + eps))

        if channel_weights is not None and name in channel_weights:
            w = channel_weights[name].to(bits.device)
            if w.numel() == 1:
                w = w.expand_as(bits)
        else:
            w = torch.ones_like(bits)

        bits_w = (bits * w).sum()
        w_sum = w.sum()

        total_bits_weighted = bits_w if total_bits_weighted is None else (total_bits_weighted + bits_w)
        total_weight = w_sum if total_weight is None else (total_weight + w_sum)

    if total_weight is None or float(total_weight.detach().cpu()) == 0.0:
        device = next(
            p.device for pair in step_sizes_dict.values() for p in pair if p is not None
        )
        return torch.zeros((), device=device)

    bits_avg = total_bits_weighted / (total_weight + eps)
    excess = bits_avg - target_bits
    penalty = lam * F.softplus(alpha * excess) ** 2
    return penalty


def backward_likelihood(forward, target, args, device, *, layer_count=1):
    return _backward_likelihood(forward, target, args, device, marginal=False,
                                loss_scale=1.0 / layer_count)


def prior(steps, ranges, layers, args):
    if args.method == "ppq":
        # The active PPQ network objective uses the weighted cap, without MDL.
        return prior_weighted_avg_bits_cap_poseidon(
            {name: (step, None) for name, step in steps.items()}, ranges,
            {name: step.new_full(step.shape, layers[name].in_features)
             for name, step in steps.items()},
            target_bits=getattr(args, "avg_cap_bits", 8.0),
            lam=getattr(args, "avg_cap_lam", 400.0),
            alpha=getattr(args, "avg_cap_alpha", 10.0),
        )
    return compute_mdl_prior({name: (value, None) for name, value in steps.items()},
                             ranges, gamma=args.gamma)
