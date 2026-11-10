"""Original SAPQ routing, calibration/cache policy and update equivalence."""
from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from SAPQ import origin
from quantization.runner import parse_args, validate_args, run_quantization


def arguments(tmp_path, *extra):
    _, args = parse_args([
        "--origin", "--DEV", "cpu", "--model_path", str(tmp_path / "model"),
        "--data_path", str(tmp_path / "data"), "--save_path", str(tmp_path / "artifact"),
        "--origin_root", str(tmp_path / "old-root"), *extra,
    ], default_method="sapq", family="sapq")
    return validate_args(args)


def test_origin_overrides_modern_config_and_explicit_experiment_knobs(tmp_path):
    args = arguments(tmp_path, "--config", "configs/sapq/default.json", "--variant", "global",
                     "--traj_num", "16", "--final_time", "10", "--epochs", "5", "--batch_size", "8")
    # Reference constants from main's unchanged archived PPQConfig.
    reference = Path(__file__).resolve().parents[1] / "archive/main/PPQ/config.py"
    tree = ast.parse(reference.read_text())
    config = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PPQConfig")
    values = {node.target.id: ast.literal_eval(node.value) for node in config.body
              if isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Constant)}
    mapping = {"batch_size": "calib_batchsize", "epochs": "num_epochs", "mc_samples": "num_mc_samples",
               "learning_rate": "base_lr", "percentile": "percentile_prob"}
    for key in ("init_bits", "target_bits", "eta", "prior_scale", "sigma0", "alpha",
                "bmax_bits", "min_epochs", "early_stop_bits", "calib_steps", *mapping):
        assert getattr(args, key) == values[mapping.get(key, key)]
    assert args.sensitivity_normalization == "normalized"
    assert (args.traj_num, args.final_time, args.method, args.variant) == (2048, 14, "sapq", "global")
    assert args.origin_overrides["traj_num"] == 16
    assert args.origin_overrides["epochs"] == 5
    assert args.model_path == str(tmp_path / "model")
    assert args.save_path == str(tmp_path / "artifact")
    assert args.data_path == str(tmp_path / "data")


def test_without_origin_previous_defaults_are_unchanged():
    _, args = parse_args(["--config", "configs/sapq/default.json", "--traj_num", "16", "--final_time", "10"],
                         default_method="sapq", family="sapq")
    assert not args.origin
    assert (args.traj_num, args.final_time, args.sensitivity_normalization) == (16, 10, "raw")
    assert not hasattr(args, "min_epochs")
    with pytest.raises(SystemExit):
        parse_args(["--origin"], default_method="ppq", family="ppq")


def test_origin_default_paths_and_config_switch(tmp_path, monkeypatch):
    monkeypatch.delenv("PPQ_MODEL_PATH", raising=False)
    monkeypatch.delenv("PPQ_DATA_PATH", raising=False)
    _, args = parse_args(["--origin"], default_method="sapq", family="sapq")
    assert args.model_path == "models/NS-PwC-L" and args.data_path == "dataset/NS-PwC"
    assert "origin_main_seed" in validate_args(args).save_path
    path = tmp_path / "origin.json"
    path.write_text(json.dumps({"origin": True, "data_path": "my-data", "model_path": "my-model",
                                "sensitivity_normalization": "raw"}))
    _, args = parse_args(["--config", str(path)], default_method="sapq", family="sapq")
    assert (args.data_path, args.model_path, args.sensitivity_normalization) == ("my-data", "my-model", "normalized")


class SampleDataset(Dataset):
    constants = {"mean": 0, "std": 1}

    def __len__(self):
        return 2048

    def __getitem__(self, index):
        return {"pixel_values": torch.full((3,), float(index)), "labels": torch.zeros(2)}


def test_origin_samples_1024_observations_without_time_overrides(tmp_path):
    args = arguments(tmp_path)
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return SampleDataset()

    torch.manual_seed(17)
    _, loader, metadata = origin.build_calibration_loader(args, factory)
    selected = torch.cat([batch["pixel_values"][:, 0] for batch in loader]).long()
    assert calls == [{"dataset": args.dataset, "which": "train", "num_trajectories": 2048,
                      "data_path": args.data_path}]
    assert len(loader) == 512 and len(selected) == len(selected.unique()) == 1024
    assert selected.tolist() != list(range(1024))
    assert metadata["selection"] == "shuffled_samples" and metadata["num_samples"] == 1024


def test_original_frozen_batches_are_reused_without_resampling_or_truncation(tmp_path):
    args = arguments(tmp_path)
    path = Path(args.origin_root) / "ppq_artifacts/frozen_calibration_batches.pt"
    path.parent.mkdir(parents=True)
    batches = [{"pixel_values": torch.full((2, 3), float(i))} for i in range(514)]
    torch.save(batches, path)
    _, loader, metadata = origin.build_calibration_loader(args, lambda **kw: SampleDataset())
    assert len(loader) == 514 and metadata["num_samples"] == 1028
    assert [b["pixel_values"][0, 0].item() for b in loader] == list(range(514))
    assert metadata["selection"] == "original_frozen_batches"
    assert metadata["frozen_batches_sha256"]
    args.origin_frozen_batches = str(tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError, match="origin_frozen_batches"):
        origin.build_calibration_loader(args, lambda **kw: SampleDataset())


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(3, 4)
        self.last = nn.Linear(4, 2)

    def forward(self, pixel_values, time=None, pixel_mask=None, labels=None):
        return SimpleNamespace(output=self.last(self.first(pixel_values).tanh()))


def load_original_functions():
    """Execute unchanged main functions, without their historical import paths."""
    directory = Path(__file__).resolve().parents[1] / "archive/main"
    namespace = {"torch": torch, "F": torch.nn.functional,
                 "get_model_output_tensor": lambda out: out.output}
    for file, names in {
        "SAPQ/sapq_loss.py": {"_get_weight_steps", "_infer_device_from_steps",
                             "compute_mc_negative_loglikelihood_network_global", "compute_sensitivity_aware_bit_prior"},
        "PPQ/noise.py": {"add_quantization_noise"},
        "PPQ/optimize.py": {"get_lr_for_epoch", "clamp_step_sizes_"},
        "PPQ/metrics.py": {"compute_avg_bits"},
    }.items():
        tree = ast.parse((directory / file).read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        assert len(functions) == len(names)
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                                  *functions], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(directory / file), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_mc_replay_preserves_gradients_noise_rng_and_one_live_graph(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable on this host.")
    device = torch.device(device_name)
    args = SimpleNamespace(mc_samples=5, eta=1e-3)
    step = nn.Parameter(torch.tensor([0.1, 0.2], dtype=torch.float64, device=device))
    target = torch.ones(2, 2, dtype=torch.float64, device=device)
    pending, draws = [], {False: [], True: []}

    def forward():
        # No previous MC output with a grad graph may remain referenced.
        if torch.is_grad_enabled():
            assert all(reference() is None for reference in pending)
        cpu_noise = torch.rand((), dtype=target.dtype)
        output = target + (torch.rand_like(target) - 0.5 + cpu_noise.to(device)) * step
        draws[torch.is_grad_enabled()].append(output.detach().clone())
        if torch.is_grad_enabled():
            pending.append(weakref.ref(output))
        return output

    torch.manual_seed(76)
    if device.type == "cuda":
        torch.cuda.manual_seed(76)
    before = origin._rng_state(device)
    scores = []
    for _ in range(args.mc_samples):
        # Use the same random operations as forward(), without its live-graph check.
        cpu_noise = torch.rand((), dtype=target.dtype)
        pred = target + (torch.rand_like(target) - 0.5 + cpu_noise.to(device)) * step
        scores.append(-(pred - target).square().mean(1) / (2 * args.eta))
    expected = -(torch.stack(scores).logsumexp(0)
                 - torch.log(torch.tensor(float(args.mc_samples), device=device))).sum()
    after = origin._rng_state(device)
    expected.backward()
    gradient = step.grad.clone()
    step.grad = None
    origin._restore_rng(before, device)
    actual = origin._backward_likelihood(forward, target, args, device)
    assert actual == pytest.approx(expected.item(), rel=1e-12, abs=1e-12)
    torch.testing.assert_close(step.grad, gradient, rtol=1e-12, atol=1e-12)
    assert len(pending) == args.mc_samples
    for original, replayed in zip(draws[False], draws[True]):
        torch.testing.assert_close(original, replayed, rtol=0, atol=0)
    for key, state in after.items():
        assert torch.equal(origin._rng_state(device)[key], state)


def test_origin_optimizer_matches_main_original_updates_and_stop_epoch(tmp_path):
    old = load_original_functions()
    torch.manual_seed(91)
    model = TinyModel().double().requires_grad_(False)
    layers = {"first": model.first, "last": model.last}
    ranges = {name: {"weight_ranges": torch.full((layer.out_features,), 0.8, dtype=torch.float64)}
              for name, layer in layers.items()}
    steps = {name: nn.Parameter(record["weight_ranges"].clone() / 512) for name, record in ranges.items()}
    original_steps = {name: (nn.Parameter(step.detach().clone()), None) for name, step in steps.items()}
    sensitivity = {name: torch.linspace(0, 1, len(step), dtype=torch.float64) for name, step in steps.items()}
    batches = [{"pixel_values": torch.randn(2, 3, dtype=torch.float64)} for _ in range(2)]
    targets = [model(**batch).output.detach() for batch in batches]
    paths = []
    for index, (batch, target) in enumerate(zip(batches, targets)):
        path = tmp_path / f"batch{index}.pt"
        torch.save({"batch": batch, "clean": target}, path)
        paths.append(path)
    args = SimpleNamespace(epochs=5, min_epochs=2, early_stop_bits=20, learning_rate=9.1e-4,
                           mc_samples=3, eta=1e-6, bmax_bits=20, prior_mode="block_sens",
                           target_bits=8, sigma0=0.5, alpha=1, prior_scale=1e-4)
    optimizer = torch.optim.Adam([pair[0] for pair in original_steps.values()], lr=args.learning_rate)
    old_history = []
    torch.manual_seed(35)
    for epoch in range(1, args.epochs + 1):
        lr = old["get_lr_for_epoch"](epoch, args.learning_rate, args.epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        for index in range(len(batches)):
            optimizer.zero_grad()
            like = old["compute_mc_negative_loglikelihood_network_global"](
                model, original_steps, batches, targets, index, num_mc_samples=args.mc_samples,
                eta=args.eta, device="cpu")
            prior = old["compute_sensitivity_aware_bit_prior"](
                original_steps, ranges, sensitivity, b_target=8, sigma0=0.5, alpha=1, prior_scale=1e-4)
            (like + prior).backward()
            optimizer.step()
            old["clamp_step_sizes_"](original_steps, ranges, 20, device="cpu", weight_only=True)
        bits = old["compute_avg_bits"](original_steps, ranges,
                 {name: torch.full_like(step, layers[name].in_features) for name, step in steps.items()})
        old_history.append(bits)
        if epoch >= args.min_epochs and bits < args.early_stop_bits:
            break
    torch.manual_seed(35)
    history = origin._optimize(model, layers, steps, ranges, sensitivity, paths, args, torch.device("cpu"))
    assert len(history) == len(old_history) == 2
    for name, step in steps.items():
        torch.testing.assert_close(step, original_steps[name][0], rtol=1e-12, atol=1e-12)
    assert [entry["avg_bits"] for entry in history] == pytest.approx(old_history, rel=1e-12)


def test_origin_exports_standalone_model_using_original_caches(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    torch.manual_seed(4)
    model = TinyModel()
    base = Path(args.origin_root)
    range_file = base / "precalculated_ranges/model/p1e-04/ranges.pt"
    step_file = base / "initial_step_sizes/model/p1e-04/9bit.pt"
    sens_file = base / "SAPQ/prior_sensitivity_sobo/model/prior_sensitivity.pt"
    layers = {"first": model.first, "last": model.last}
    ranges = {name: {"weight_ranges": torch.ones(layer.out_features),
                     "activation_ranges": torch.ones(layer.in_features)} for name, layer in layers.items()}
    steps = {name: (torch.full((layer.out_features,), 0.002), torch.ones(layer.in_features))
             for name, layer in layers.items()}
    for path, value in (
        (range_file, {"ranges_dict": ranges}), (step_file, {"step_sizes": steps}),
        (sens_file, {"layer_sensitivity_norm": {name: torch.ones(layer.out_features) for name, layer in layers.items()}}),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value, path)
    observed = {}

    def optimize(model, layers, learned, ranges, sensitivity, paths, args, device):
        assert (args.epochs, args.min_epochs, args.early_stop_bits) == (21, 10, 4)
        for name in learned:
            torch.testing.assert_close(learned[name], steps[name][0])
            assert torch.all(sensitivity[name] == 1)
        observed["steps"] = {name: value.detach().clone() for name, value in learned.items()}
        return [{"epoch": 10, "avg_bits": 8.9}]

    monkeypatch.setattr(origin, "_optimize", optimize)
    loader = [{"pixel_values": torch.randn(2, 3), "labels": torch.randn(2, 2)}]
    original = deepcopy(model.state_dict())
    destination, metadata = run_quantization(args, model=model, calibration_loader=loader,
                                            calibration_metadata={"selection": "injected", "num_samples": 2})
    assert metadata["quantization"]["origin"] is True
    assert metadata["quantization"]["origin_caches"]["sensitivity"]["sha256"]
    assert metadata["settings"]["sensitivity_normalization"] == "normalized"
    from quantization.artifacts import load_quantized_artifact
    restored, _ = load_quantized_artifact(destination)
    for name, step in observed["steps"].items():
        expected = (original[name + ".weight"] / step[:, None]).round() * step[:, None]
        torch.testing.assert_close(restored.get_submodule(name).weight, expected)
    assert all(not module._forward_hooks for module in restored.modules())
