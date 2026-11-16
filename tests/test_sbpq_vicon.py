"""VICON source parity, real-model gradients, materialization and restart checks."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from quantization.runner import parse_args, run_quantization, validate_args
from SBPQ.resume import fingerprint
from SBPQ.vicon.data_utils import build_calibration_loader, load_config
from SBPQ.vicon.model_utils import load_model, model_forward
from SBPQ.vicon.models import ICON_CROPPED
from SBPQ.vicon.likelihood import ViconNetworkLikelihood
from SBPQ.vicon.sobolev import query_fields, sensitivity_loss


@pytest.fixture
def vicon_case(tmp_path):
    pytest.importorskip("hydra")
    h5py = pytest.importorskip("h5py")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    data = tmp_path / "data"
    rng = np.random.default_rng(21)
    order = {}
    for category, folder in (("COMPRESSIBLE2D", "2D_Train_Rand"), ("EULER2D", "2D_Train_Turb"),
                             ("NS2D", "NavierStokes-2D-conditoned")):
        directory = data / folder
        if category != "NS2D":
            directory /= "train"
        directory.mkdir(parents=True)
        names = ["b_train.h5", "a_train.h5"] if category == "NS2D" else ["b.h5", "a.h5"]
        for name in names:
            with h5py.File(directory / name, "w") as handle:
                group = handle.create_group("train") if category == "NS2D" else handle
                fields = ("u", "vx", "vy") if category == "NS2D" else ("density", "pressure", "Vx", "Vy")
                for field in fields:
                    group[field] = rng.normal(size=(3, 6, 4, 4)).astype("float32")
        # Unreadable validation/test data catches accidental non-train access.
        (data / folder / "unused_valid.h5").write_text("not a training file")
        order[category] = names if category == "NS2D" else [f"train/{name}" for name in names]
    manifest = tmp_path / "order.json"
    manifest.write_text(json.dumps({"version": 1, "splits": {"train": order}}))
    # Test files contain the same fields in their own split; generating these
    # must not change the recorded training-data RNG sequence.
    for folder in ("2D_Train_Rand", "2D_Train_Turb", "NavierStokes-2D-conditoned"):
        if folder.startswith("2D_"):
            source = data / folder / "train/a.h5"
            target = data / folder / "test/0.h5"
        else:
            source = data / folder / "a_train.h5"
            target = data / folder / "a_test.h5"
        target.parent.mkdir(exist_ok=True)
        with h5py.File(source) as original, h5py.File(target, "w") as copied:
            source_group = original["train"] if "train" in original else original
            target_group = copied.create_group("test") if folder.startswith("Navier") else copied
            for field in source_group:
                target_group[field] = source_group[field][:]
    overrides = ["model.transformer.dim_token=8", "model.transformer.nhead=2",
                 "model.transformer.dim_feedforward=16", "model.transformer.num_layers=2",
                 "model.patch_num_in=2", "model.patch_num_out=2", "model.patch_resolution=2",
                 "model.demo_num=3", "datasets.demo_num=3", "datasets.delta_t_range=[1,2]", "loss.min_ex=1"]
    argv = ["--model", "vicon", "--DEV", "cpu",
            "--ckpt_path", str(tmp_path / "model.pth"), "--data_root", str(data),
            "--vicon_file_order", str(manifest), "--calibration_num", "2", "--batch_size", "2",
            "--total_steps", "4", "--save_every", "1", "--mc_samples", "2"]
    for value in overrides:
        argv += ["--config_override", value]
    def options(path=None, extra=()):
        additions = ["--save_path", str(path)] if path else []
        return validate_args(parse_args([*argv, *additions, *extra], default_method="sbpq", family="sbpq")[1])
    args = options(tmp_path / "initial")
    cfg = load_config(args)
    torch.manual_seed(17)
    native = ICON_CROPPED(cfg.model).eval()
    torch.save({"model_state_dict": {"module." + key: value for key, value in native.state_dict().items()}}, args.model_path)
    reference = json.loads((Path(__file__).parent / "fixtures/vicon_reference.json").read_text())
    try:
        yield options, native, reference
    finally:
        torch.set_num_threads(previous_threads)


def test_calibration_matches_safesvd_source_and_restarts_exactly(vicon_case):
    options, _, reference = vicon_case
    args = options()
    loader, metadata = build_calibration_loader(args)
    batches = list(loader)
    assert len(batches) == 3
    for batch in batches:
        expected = reference["categories"][batch["category"]]
        for key in ("mean", "std", "channel_mask", "time_in", "time_out", "delta_t"):
            torch.testing.assert_close(batch[key], torch.tensor(expected[key], dtype=batch[key].dtype))
        torch.testing.assert_close(batch["pixel_values"][0, 0, :, 1, 1], torch.tensor(expected["initial_probe"]))
        torch.testing.assert_close(batch["labels"][0, 0, :, 1, 1], torch.tensor(expected["label_probe"]))
        assert float(batch["pixel_values"].sum()) == pytest.approx(expected["initial_sum"], abs=1e-5)
        assert float(batch["labels"].sum()) == pytest.approx(expected["label_sum"], abs=1e-5)
    assert fingerprint(list(loader)) == fingerprint(batches)
    assert metadata["num_samples"] == 6
    assert metadata["category_samples"] == {"COMPRESSIBLE2D": 2, "EULER2D": 2, "NS2D": 2}
    changed, _ = build_calibration_loader(options(extra=("--calibration_seed", "43")))
    assert fingerprint(list(changed)) != fingerprint(batches)


def test_strict_checkpoint_attention_parity_and_query_masks(vicon_case):
    options, native, _ = vicon_case
    args = options()
    loader, _ = build_calibration_loader(args)
    batch = next(iter(loader))
    converted = load_model(args, "cpu")
    from SBPQ.vicon.blocks import build_block_mapping
    _, layers, _, _, _ = build_block_mapping(converted)
    calls = []
    hooks = [layer.register_forward_hook(lambda *_args, key=name: calls.append(key)) for name, layer in layers.items()]
    try:
        with torch.no_grad():
            torch.testing.assert_close(model_forward(native, batch, "cpu"), model_forward(converted, batch, "cpu"),
                                       rtol=2e-5, atol=2e-6)
    finally:
        for hook in hooks:
            hook.remove()
    assert set(calls) == set(layers)
    assert len(layers) == 14  # Two projections + six Linears per encoder layer.
    assert any("q_proj" in name for name in layers)
    output = model_forward(converted, batch, "cpu")
    original = sensitivity_loss(output, batch, args)
    altered = output.clone()
    altered[:, :batch["min_ex"]] += 1e5  # Demonstrations are outside the query objective.
    altered[:, :, -1] += 1e5  # Node-type metadata is not predicted physics.
    altered[:, :, 4] += 1e5  # Inactive CFD channel.
    torch.testing.assert_close(sensitivity_loss(altered, batch, args), original, rtol=0, atol=0)
    args.task_loss_weight, args.sobolev_order = 0, 0
    args.sobolev_order_weights = 1
    scaled = deepcopy(batch)
    scaled["std"] *= 3
    torch.testing.assert_close(sensitivity_loss(output, scaled, args), sensitivity_loss(output, batch, args) * 3)
    # A checkpoint with the wrong architecture is never partially loaded.
    payload = torch.load(args.model_path, weights_only=True)
    payload["model_state_dict"].pop("module.pre_proj.weight")
    torch.save(payload, args.model_path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_model(args, "cpu")


def test_checkpointed_mc_matches_stacked_loss_and_gradients(vicon_case):
    options, _, _ = vicon_case
    args = options()
    loader, _ = build_calibration_loader(args)
    batch = next(iter(loader))
    model = load_model(args, "cpu")
    from SBPQ.vicon.blocks import build_block_mapping
    _, layers, _, _, _ = build_block_mapping(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        clean = query_fields(model_forward(model, batch, "cpu"), batch)
    likelihood = ViconNetworkLikelihood(model, [batch], [clean], num_mc_samples=3, eta=1e-4, device="cpu")
    def evaluate(efficient):
        steps = {name: torch.full((layer.out_features,), 1e-3, requires_grad=True) for name, layer in layers.items()}
        torch.manual_seed(123)
        loss = likelihood(steps, 0, memory_efficient=efficient)
        return loss.detach(), torch.autograd.grad(loss, tuple(steps.values()))
    a, ga = evaluate(False)
    b, gb = evaluate(True)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    for original, checkpointed in zip(ga, gb):
        assert torch.isfinite(checkpointed).all()
        torch.testing.assert_close(original, checkpointed, rtol=0, atol=0)
    assert not any(module._forward_hooks for module in model.modules())


def test_public_vicon_quantization_export_and_interrupted_resume(vicon_case, tmp_path, monkeypatch):
    options, native, _ = vicon_case
    from SBPQ.trainer import SBPQTrainer
    from quantization.artifacts import load_quantized_artifact
    continuous, meta = run_quantization(options(tmp_path / "continuous"))
    original = SBPQTrainer.train_step
    def interrupt(self, *args, **kwargs):
        if len(self.history) == 2:
            raise KeyboardInterrupt
        return original(self, *args, **kwargs)
    monkeypatch.setattr(SBPQTrainer, "train_step", interrupt)
    partial_args = options(tmp_path / "partial")
    with pytest.raises(KeyboardInterrupt):
        run_quantization(partial_args)
    monkeypatch.setattr(SBPQTrainer, "train_step", original)
    resumed_args = validate_args(parse_args(["--resume", partial_args.save_path], default_method="sbpq", family="sbpq")[1])
    resumed, _ = run_quantization(resumed_args)
    a = torch.load(continuous / "sbpq_checkpoint.pt", weights_only=True)
    b = torch.load(resumed / "sbpq_checkpoint.pt", weights_only=True)
    torch.testing.assert_close(a["optimized_step_sizes"], b["optimized_step_sizes"], rtol=0, atol=0)
    torch.testing.assert_close(a["optimizer_state_dict"], b["optimizer_state_dict"], rtol=0, atol=0)
    assert a["history"] == b["history"]
    assert any(entry["gradient_norm"] > 0 for entry in a["history"])
    assert a["completed_steps"] == b["completed_steps"] == 4
    exported, restored_meta = load_quantized_artifact(resumed)
    assert restored_meta["artifact_type"] == "vicon_quantization"
    assert meta["calibration"]["num_samples"] == 6
    assert meta["quantization"]["sensitivity_batches"] == 3
    assert meta["evaluation_performed"] is False
    assert not any(module._forward_hooks for module in exported.modules())
    args = options()
    loader, _ = build_calibration_loader(args)
    batch = next(iter(loader))
    from SBPQ.vicon.checkpoints import load_compressed_state
    fp = load_compressed_state(load_model(args, "cpu"), resumed / "compressed_model.pth")
    with torch.no_grad():
        torch.testing.assert_close(model_forward(exported, batch, "cpu"), model_forward(fp, batch, "cpu"),
                                   rtol=2e-5, atol=2e-6)
    learned = torch.load(resumed / "quantization_state.pt", weights_only=True)
    clean = load_model(args, "cpu")
    for name, step in learned["steps"].items():
        expected = torch.round(clean.get_submodule(name).weight / step[:, None]) * step[:, None]
        torch.testing.assert_close(exported.get_submodule(name).weight, expected, rtol=0, atol=0)
    # Content changes cannot silently reuse cached targets or sensitivity.
    import h5py
    for path in (Path(args.data_path) / "2D_Train_Rand/train").glob("*.h5"):
        with h5py.File(path, "r+") as handle:
            handle["pressure"][:] += .5
    with pytest.raises(ValueError, match="Calibration data changed|preparation changed"):
        run_quantization(resumed_args)


def test_vicon_config_cli_overrides_and_validation():
    root = Path(__file__).resolve().parents[1]
    _, args = parse_args(["--config", str(root / "configs/sbpq/vicon.json"), "--calibration_num", "2048"],
                         default_method="sbpq", family="sbpq")
    assert (args.model_type, args.calibration_num, args.batch_size) == ("vicon", 2048, 64)
    assert (args.init_bits, args.mc_samples, args.beta_kappa) == (8, 10, 150)
    with pytest.raises(SystemExit):
        parse_args(["--safe_svd_root", "/unused"], default_method="sbpq", family="sbpq")


def test_local_network_matches_recorded_safesvd_prediction(vicon_case):
    options, native, reference = vicon_case
    loader, _ = build_calibration_loader(options())
    batch = next(iter(loader))
    with torch.no_grad():
        for name, parameter in native.named_parameters():
            values = torch.linspace(-0.1, 0.1, parameter.numel()).reshape_as(parameter)
            if "norm" in name and name.endswith("weight"):
                values += 1
            parameter.copy_(values)
        prediction = model_forward(native, batch, "cpu")
    torch.testing.assert_close(prediction.flatten()[:32], torch.tensor(reference["native_prediction"]["probe"]),
                               rtol=2e-5, atol=2e-6)
    assert float(prediction.sum()) == pytest.approx(reference["native_prediction"]["sum"], rel=2e-5, abs=2e-6)
    from SBPQ.vicon.evaluation import batch_factory, evaluate_model
    cfg = load_config(options())
    metrics = evaluate_model(native, batch_factory(cfg, batch_size=2, max_batches=1, protocol="paired"), cfg, "cpu")
    for category, values in reference["native_metrics"].items():
        for name, expected in values.items():
            assert metrics[category][name] == pytest.approx(expected, rel=2e-5, abs=2e-6)


def test_independent_local_evaluation_and_sampling_protocols(vicon_case, tmp_path):
    options, _, _ = vicon_case
    from SBPQ.vicon.evaluation import batch_factory, main
    args = options(tmp_path / "quantized", extra=("--total_steps", "1", "--mc_samples", "1"))
    directory, metadata = run_quantization(args)
    cfg = load_config(args)
    paired = batch_factory(cfg, batch_size=2, max_batches=1, protocol="paired")
    research = batch_factory(cfg, batch_size=2, max_batches=1, protocol="research")
    def selection(factory):
        return fingerprint([(category, payload[2], payload[3]) for category, payload in factory()])
    assert selection(paired) == selection(paired)
    assert selection(research) != selection(research)
    output = tmp_path / "evaluation.json"
    argv = ["--checkpoint", args.model_path, "--model_path", str(directory / "compressed_model.pth"),
            "--data_path", args.data_path, "--DEV", "cpu", "--batch_size", "2",
            "--max_batches", "1", "--protocol", "paired", "--output_path", str(output)]
    for override in args.config_override:
        argv += ["--config_override", override]
    result = main(argv)
    assert json.loads(output.read_text()) == result
    assert set(result) == {"avg_weight_bits", "metrics"}
    assert result["avg_weight_bits"] == metadata["quantization"]["avg_weight_bits"]
    assert set(result["metrics"]) == {"COMPRESSIBLE2D", "EULER2D", "NS2D"}
    for metrics in result["metrics"].values():
        assert "base_loss" in metrics and "sobolev" in metrics
        for values in metrics.values():
            assert set(values) == {"origin", "compressed", "change_percent"}
            assert np.isfinite(values["origin"]) and np.isfinite(values["compressed"])


def test_public_command_runs_without_any_external_vicon_or_safesvd(vicon_case, tmp_path):
    options, _, _ = vicon_case
    args = options()
    settings = {key: value for key, value in vars(args).items() if not key.startswith("_")}
    settings.update(save_path=str(tmp_path / "standalone"), total_steps=1, mc_samples=1)
    settings.pop("config", None)
    config = tmp_path / "standalone.json"
    config.write_text(json.dumps(settings))
    script = '''
import importlib.abc
import json
from pathlib import Path
import runpy
import sys

class RejectExternalSources(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'VICON', 'safe_svd'}:
            raise AssertionError('External source import: ' + fullname)

sys.meta_path.insert(0, RejectExternalSources())
config = sys.argv[1]
sys.argv = ['SBPQ.sbpq', '--config', config]
runpy.run_module('SBPQ.sbpq', run_name='__main__')
settings = json.loads(Path(config).read_text())
directory = next(Path(settings['save_path']).glob('*/metadata.json')).parent
from quantization.artifacts import load_quantized_artifact
model, metadata = load_quantized_artifact(directory)
assert all(not module.__class__.__module__.startswith(('VICON', 'safe_svd')) for module in model.modules())
assert metadata['quantization']['completed_steps'] == 1
print('VICON_STANDALONE_OK')
'''
    completed = subprocess.run([sys.executable, "-c", script, str(config)],
                               cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60,
                               env={**os.environ, "SAFE_SVD_ROOT": str(tmp_path / "does-not-exist")})
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "VICON_STANDALONE_OK" in completed.stdout


def test_old_artifact_class_names_resolve_locally():
    import io
    from quantization import compat
    from SBPQ.vicon.attention import ExplicitMultiheadAttention
    loader = compat.Unpickler(io.BytesIO())
    assert loader.find_class("VICON.src.models", "ICON_CROPPED") is ICON_CROPPED
    assert loader.find_class("safe_svd.helper.collector.vicon.vicon_attention", "ExplicitMultiheadAttention") is ExplicitMultiheadAttention
