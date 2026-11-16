"""VICON source parity, real-model gradients, materialization and restart checks."""
from copy import deepcopy
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quantization.runner import parse_args, run_quantization, validate_args
from SBPQ.resume import fingerprint
from SBPQ.vicon.data_utils import build_calibration_loader, load_config
from SBPQ.vicon.model_utils import configure_source, load_model, model_forward, source_root
from SBPQ.vicon.objectives import ViconNetworkLikelihood, query_fields, sensitivity_loss


@pytest.fixture
def vicon_case(tmp_path):
    pytest.importorskip("hydra")
    h5py = pytest.importorskip("h5py")
    try:
        root = source_root()
    except FileNotFoundError:
        pytest.skip("Set SAFE_SVD_ROOT to run integration checks against the SafeSVD VICON source")
    configure_source(root)
    from VICON.src.models import ICON_CROPPED
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
    overrides = ["model.transformer.dim_token=8", "model.transformer.nhead=2",
                 "model.transformer.dim_feedforward=16", "model.transformer.num_layers=2",
                 "model.patch_num_in=2", "model.patch_num_out=2", "model.patch_resolution=2",
                 "model.demo_num=3", "datasets.demo_num=3", "datasets.delta_t_range=[1,2]", "loss.min_ex=1"]
    argv = ["--model", "vicon", "--safe_svd_root", str(root), "--DEV", "cpu",
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
    try:
        yield options, native, root
    finally:
        torch.set_num_threads(previous_threads)


def original_preprocess(root, pairs, cfg):
    # Execute the actual upstream method in isolation; importing Trainer also
    # initializes plotting/logging dependencies unnecessary for quantization.
    module = ast.parse((root / "VICON/src/trainer.py").read_text())
    trainer = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Trainer")
    method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "_data_preprocess")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "VICON/src/trainer.py", "exec"), namespace)
    return namespace["_data_preprocess"](SimpleNamespace(loss_cfg=cfg.loss, model_cfg=cfg.model), pairs)


def test_calibration_matches_safesvd_source_and_restarts_exactly(vicon_case):
    options, _, root = vicon_case
    args = options()
    cfg = load_config(args)
    from VICON.src.dataset import all_datasets
    from safe_svd.helper.collector.vicon.vicon_calibration_dataset import build_calibration_loaders
    from safe_svd.helper.collector.vicon.vicon_file_order import apply_vicon_file_order
    datasets = all_datasets(cfg.datasets, 0, cfg.train_seed, "train")
    apply_vicon_file_order(datasets, args.vicon_file_order)
    original = build_calibration_loaders(datasets, calibration_num=2, batch_size=2, num_workers=0,
                                         random_calibration=False, calibration_seed=42)
    loader, metadata = build_calibration_loader(args)
    batches = list(loader)
    assert len(batches) == 3
    for (category, upstream), batch in zip(original.items(), batches):
        raw = next(iter(upstream))
        expected, mean, std = original_preprocess(root, raw[1], cfg)
        assert batch["category"] == category
        torch.testing.assert_close(batch["pixel_values"], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(batch["labels"], expected[1], rtol=0, atol=0)
        torch.testing.assert_close(batch["channel_mask"], expected[2], rtol=0, atol=0)
        torch.testing.assert_close(batch["mean"], mean, rtol=0, atol=0)
        torch.testing.assert_close(batch["std"], std, rtol=0, atol=0)
        torch.testing.assert_close(batch["time_in"], raw[2], rtol=0, atol=0)
    assert fingerprint(list(loader)) == fingerprint(batches)
    assert metadata["num_samples"] == 6
    assert metadata["category_samples"] == {"COMPRESSIBLE2D": 2, "EULER2D": 2, "NS2D": 2}
    changed, _ = build_calibration_loader(options(extra=("--calibration_seed", "43")))
    assert fingerprint(list(changed)) != fingerprint(batches)


def test_strict_checkpoint_attention_parity_and_query_masks(vicon_case):
    options, native, root = vicon_case
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
    from safe_svd.helper.collector.vicon.vicon_fisher_forward import _patched_forward
    _, layers, _, _, _ = build_block_mapping(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with _patched_forward(model):
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
    options, native, root = vicon_case
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
    # Exercise SafeSVD's actual checkpoint and attention evaluator modules under
    # an isolated package name (both repositories have an evaluation package).
    import importlib.util
    import importlib
    import sys
    spec = importlib.util.spec_from_file_location("_safe_vicon_evaluation", root / "evaluation/vicon/__init__.py",
                                                 submodule_search_locations=[str(root / "evaluation/vicon")])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    attention = importlib.import_module("_safe_vicon_evaluation.attention")
    checkpoints = importlib.import_module("_safe_vicon_evaluation.checkpoints")
    fp = deepcopy(native)
    attention.convert_attn_to_linear(fp)
    attention.disable_transformer_fast_path(fp)
    checkpoints.load_compressed_state(fp, resumed / "compressed_model.pth")
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
