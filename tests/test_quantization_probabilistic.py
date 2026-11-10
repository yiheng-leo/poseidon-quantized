"""Small numerical and workflow checks; no checkpoints or external data needed."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from quantization.probabilistic import _backward_likelihood, _energy, _seed
from PPQ.quantize import quantize as quantize_ppq
from SAPQ.quantize import quantize as quantize_sapq
from SAPQ.loss import _prior
from SAPQ.sensitivity import _structure_loss


def quantize_probabilistic(model, loader, args, device, work_dir):
    quantize = quantize_ppq if args.method.startswith("ppq") else quantize_sapq
    return quantize(model, loader, args, device, work_dir)


class ScOTLayer(nn.Module):
    """Toy residual unit with the same tuple-output contract as ScOTLayer."""
    def __init__(self, width=4):
        super().__init__()
        self.first = nn.Linear(width, 2 * width)
        self.second = nn.Linear(2 * width, width)

    def forward(self, value, *, scale=1.0):
        return (value + scale * self.second(torch.tanh(self.first(value))),)


class ToyScOT(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([ScOTLayer(), ScOTLayer()])

    def forward(self, pixel_values, time=None, labels=None, pixel_mask=None):
        assert labels is None, "Calibration must not invoke supervised target loss"
        assert pixel_mask is None
        value = pixel_values
        for block in self.blocks:
            value = block(value, scale=0.7)[0]
        return SimpleNamespace(output=value.transpose(1, 2).reshape(-1, 4, 3, 3))


class OnePassLoader:
    def __init__(self):
        generator = torch.Generator().manual_seed(27)
        self.batches = [
            {"pixel_values": torch.randn(size, 9, 4, generator=generator),
             "labels": torch.full((size, 4, 3, 3), float("nan"))}
            for size in (2, 2, 1)
        ]
        self.iterations = 0
        self.consumed = 0
        self.dataset = SimpleNamespace(constants={"mean": [0.0] * 4, "std": [1.0, 2.0, 3.0, 4.0]})

    def __iter__(self):
        self.iterations += 1
        assert self.iterations == 1, "Adapter must freeze the supplied iterator once"
        for batch in self.batches:
            self.consumed += batch["pixel_values"].shape[0]
            yield batch


def arguments(method="sapq", **kwargs):
    values = dict(method=method, weight_bits=4, activation_bits=32, init_bits=5,
                  target_bits=3, epochs=2, mc_samples=2, learning_rate=1e-3,
                  eta=1e-3, prior_mode="block_sens", prior_scale=1e-4,
                  sigma0=0.5, alpha=1.0, gamma=0.005, sensitivity="mse",
                  sensitivity_normalization="raw", percentile=1e-4, bmax_bits=12,
                  reconstruction_iterations=10, seed=11,
                  avg_cap_bits=3.0, avg_cap_lam=1.0, avg_cap_alpha=2.0,
                  dataset="fluids.incompressible.PiecewiseConstants")
    values.update(kwargs)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("method", ["ppq", "ppq-layerwise", "sapq", "sapq-block", "sapq-network", "sapq-layerwise"])
def test_methods_consume_every_sample_learn_steps_and_save_without_hooks(tmp_path, method):
    torch.manual_seed(4)
    model = ToyScOT()
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    loader = OnePassLoader()
    args = arguments(method)
    result = quantize_probabilistic(model, loader, args, "cpu", tmp_path / "work")

    assert loader.iterations == 1 and loader.consumed == 5
    assert result.metadata["calibration_samples"] == 5
    assert result.metadata["calibration_batches"] == 3
    assert result.metadata["evaluated_test_data"] is False
    assert all(not module._forward_hooks for module in result.model.modules())
    assert not list((tmp_path / "work").iterdir())
    assert any(not torch.equal(step, result.state["initial_steps"][name])
               for name, step in result.state["steps"].items())

    modules = dict(result.model.named_modules())
    for name, step in result.state["steps"].items():
        scale = step.view(-1, 1)
        expected = torch.round(original[name + ".weight"] / scale) * scale
        torch.testing.assert_close(modules[name].weight, expected, rtol=0, atol=0)
        torch.testing.assert_close(modules[name].bias, original[name + ".bias"], rtol=0, atol=0)
    assert all(parameter.requires_grad for parameter in result.model.parameters())
    assert all(record["num_batches"] == 3 for record in result.state["history"])

    history_units = {record["unit"] for record in result.state["history"]}
    if method in {"ppq", "sapq"}:
        assert history_units == {"network"}
    elif method in {"sapq-block", "sapq-network"}:
        assert history_units == {"blocks.0", "blocks.1"}
    else:
        assert history_units == {"blocks.0.first", "blocks.0.second", "blocks.1.first", "blocks.1.second"}
    if method == "ppq":
        assert result.metadata["prior_mode"] == "weighted_average_bit_cap"
        assert result.metadata["gamma_used"] is False
    if method == "ppq-layerwise":
        assert result.metadata["layerwise_likelihood_divisor"] == len(result.state["steps"])
    if method.startswith("sapq"):
        assert result.state["sensitivity_raw"]
        for name, sensitivity in result.state["sensitivity"].items():
            torch.testing.assert_close(sensitivity, result.state["sensitivity_raw"][name])

    path = tmp_path / "quantized.pt"
    torch.save(result.model, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    probe = loader.batches[-1]["pixel_values"]
    with torch.no_grad():
        torch.testing.assert_close(result.model(probe).output, restored(probe).output, rtol=0, atol=0)


@pytest.mark.parametrize("method,fisher", [("sapq", False), ("sapq-block", True), ("sapq-network", False)])
def test_streamed_mc_has_same_likelihood_and_gradient_as_stacked_mc(method, fisher):
    args = arguments(method, mc_samples=3)
    device = torch.device("cpu")
    clean = torch.tensor([[0.1, -0.3], [0.2, 0.4]])
    gradients = torch.tensor([[0.6, 0.9], [0.7, 0.5]]) if fisher else None
    step = nn.Parameter(torch.tensor([0.12, 0.04]))

    def forward():
        return clean + (torch.rand_like(clean) - 0.5) * step

    torch.manual_seed(77)
    seeds = torch.randint(0, 2 ** 31 - 1, (args.mc_samples,)).tolist()
    scores = []
    for seed in seeds:
        with _seed(seed, device):
            scores.append(-_energy(forward(), clean, args.eta, gradients,
                                  output_sum=method == "sapq-network"))
    expected = -(torch.stack(scores).logsumexp(0) - torch.tensor(float(args.mc_samples)).log()).sum()
    expected.backward()
    expected_gradient = step.grad.detach().clone()
    step.grad = None

    torch.manual_seed(77)
    actual = _backward_likelihood(forward, clean, args, device, marginal=True, fisher=gradients,
                                  output_sum=method == "sapq-network")
    assert actual == pytest.approx(expected.item(), rel=1e-6, abs=1e-6)
    torch.testing.assert_close(step.grad, expected_gradient, rtol=1e-5, atol=1e-6)


def test_sapq_prior_keeps_training_sigma_convention():
    args = arguments(target_bits=4, prior_scale=0.3, alpha=2.0)
    step = nn.Parameter(torch.tensor([0.1, 0.2]))
    ranges = {"layer": {"weight_ranges": torch.tensor([2.0, 3.0])}}
    sensitivity = {"layer": torch.tensor([0.2, 0.6])}
    loss = _prior({"layer": step}, ranges, sensitivity, args)
    bits = torch.log2(ranges["layer"]["weight_ranges"] / step)
    sigma = args.sigma0 * (1 + args.alpha * sensitivity["layer"])
    torch.testing.assert_close(loss, args.prior_scale * ((bits - 4).square() / (2 * sigma.square())).sum())


@pytest.mark.parametrize("mode", ["sobolev", "divergence"])
def test_physical_sensitivity_computes_from_calibration_outputs(tmp_path, mode):
    result = quantize_probabilistic(ToyScOT(), OnePassLoader(),
        arguments(sensitivity=mode, epochs=1, mc_samples=1), "cpu", tmp_path)
    assert result.metadata["sensitivity"] == mode
    assert all(torch.isfinite(value).all() for value in result.state["sensitivity"].values())


def test_sobolev_uses_dataset_scale_and_preserves_discrete_differences():
    clean = torch.zeros(1, 2, 3, 3)
    pred = torch.arange(18, dtype=torch.float32).reshape_as(clean) / 18
    std = torch.tensor([2.0, 3.0]).view(1, 2, 1, 1)
    actual = _structure_loss(pred, clean, "sobolev", {"std": [2.0, 3.0]}, "wave.Gaussians")
    scaled = pred * std
    expected = scaled.square().mean() + (scaled[..., 1:] - scaled[..., :-1]).square().mean()
    expected += (scaled[..., 1:, :] - scaled[..., :-1, :]).square().mean()
    torch.testing.assert_close(actual, expected)


def test_invalid_activation_mode_does_not_consume_calibration(tmp_path):
    loader = OnePassLoader()
    with pytest.raises(ValueError, match="weight-only"):
        quantize_probabilistic(ToyScOT(), loader, arguments(activation_bits=8), "cpu", tmp_path)
    assert loader.iterations == 0


def test_layer_fallback_and_normalized_sensitivity(tmp_path):
    model = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 3))
    result = quantize_probabilistic(model, [torch.randn(2, 4)],
        arguments(sensitivity_normalization="normalized", epochs=1, mc_samples=1), "cpu", tmp_path)
    for name, sensitivity in result.state["sensitivity"].items():
        torch.testing.assert_close(sensitivity, result.state["sensitivity_normalized"][name])
    assert all(not module._forward_hooks for module in model.modules())


def test_seed_reproducibility(tmp_path):
    model = ToyScOT()
    first = quantize_probabilistic(deepcopy(model), OnePassLoader(),
        arguments("ppq", epochs=1), "cpu", tmp_path / "first")
    second = quantize_probabilistic(deepcopy(model), OnePassLoader(),
        arguments("ppq", epochs=1), "cpu", tmp_path / "second")
    for name, step in first.state["steps"].items():
        torch.testing.assert_close(step, second.state["steps"][name], rtol=0, atol=0)


def test_block_cache_keeps_legacy_fisher_damping(tmp_path):
    result = quantize_probabilistic(ToyScOT(), [torch.randn(1, 9, 4)],
        arguments("sapq-block", epochs=1, mc_samples=1, keep_work_dir=True), "cpu", tmp_path)
    records = [torch.load(path, weights_only=True) for path in tmp_path.rglob("unit_*.pt")]
    assert len(records) == 2
    assert all((record["fisher"] >= 1.0).all() for record in records)
    assert result.metadata["block_fisher_damping"] == 1.0


def test_streamed_ppq_layer_likelihood_retains_global_average_scaling():
    args = arguments("ppq-layerwise")
    step = nn.Parameter(torch.tensor([0.12, 0.04]))
    target = torch.zeros(2, 2)

    def forward():
        return (torch.rand_like(target) - 0.5) * step

    torch.manual_seed(93)
    unscaled = _backward_likelihood(forward, target, args, torch.device("cpu"), marginal=False)
    gradient = step.grad.clone()
    step.grad = None
    torch.manual_seed(93)
    scaled = _backward_likelihood(forward, target, args, torch.device("cpu"),
                                  marginal=False, loss_scale=1 / 4)
    assert scaled == pytest.approx(unscaled / 4)
    torch.testing.assert_close(step.grad, gradient / 4)
