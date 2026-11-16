"""VICON query masks and physical-unit Sobolev sensitivity objective."""
import torch


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

