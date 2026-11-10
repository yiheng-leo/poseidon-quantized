"""Small real-ScOT smoke coverage of every production quantization route.

The two synthetic calibration observations exercise integration only; they do
not stand in for the complete-trajectory dataset-selection tests or accuracy
experiments. Fresh subprocesses load only the exported quantization artifact.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from quantization import artifacts, calibration
from evaluation import evaluate
from quantization.runner import METHODS, build_parser, run_quantization


class TinyPhysicalDataset(Dataset):
    constants = {"mean": [0.8, 0.0, 0.0, 0.0], "std": [0.31, 0.391, 0.356, 0.185], "time": 20.0}
    resolution = 8
    transpose = False
    label_description = "[rho],[u,v],[p]"

    def __init__(self):
        generator = torch.Generator().manual_seed(12345)
        self.inputs = torch.randn(2, 4, 8, 8, generator=generator)
        self.inputs[:, 3] = 0.0

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        x = self.inputs[index]
        return {"pixel_values": x, "labels": x + 0.01,
                "time": torch.tensor(0.1 + 0.1 * index),
                "pixel_mask": torch.tensor([False, False, False, True])}


@pytest.fixture
def tiny_scot_case(tmp_path, monkeypatch):
    # Bound CPU work, and keep Hugging Face entirely offline and in test storage.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-cache"))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(tmp_path / "hf-cache"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    from scOT.model import ScOT, ScOTConfig
    torch.manual_seed(123)
    model = ScOT(ScOTConfig(
        image_size=8, patch_size=2, embed_dim=8, depths=[1, 1],
        num_heads=[2, 4], window_size=2, skip_connections=[0, 0],
        num_channels=4, num_out_channels=4, use_conditioning=True,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
        drop_path_rate=0.0,
    )).eval()
    dataset = TinyPhysicalDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    metadata = {
        "schema": "tiny-scot-integration-only", "dataset": "synthetic integration fixture",
        "num_trajectories": 2, "num_samples": 2, "trajectory_indices": [0, 1],
        "normalization": dataset.constants,
        "fingerprint": hashlib.sha256(dataset.inputs.numpy().tobytes()).hexdigest(),
    }
    yield model, dataset, loader, metadata
    torch.set_num_threads(previous_threads)


_FRESH_PROCESS = r'''
import json
import sys
from types import SimpleNamespace
import torch

torch.set_num_threads(1)
from quantization.artifacts import load_quantized_artifact
from quantization import calibration, runner
from evaluation import evaluate

def forbidden(*args, **kwargs):
    raise AssertionError("Evaluation attempted calibration or original-checkpoint loading")

calibration.build_calibration_loader = forbidden
runner._load_model = forbidden
model, metadata = load_quantized_artifact(sys.argv[1], "cpu")
batch = torch.load(sys.argv[2], map_location="cpu", weights_only=True)
prediction = evaluate.autoregressive_predict(model, batch, device="cpu", ar_steps=2)
changed = {**batch, "labels": torch.full_like(batch["labels"], 7777)}
other = evaluate.autoregressive_predict(model, changed, device="cpu", ar_steps=2)
torch.testing.assert_close(prediction, other, rtol=0, atol=0)
dataset = SimpleNamespace(constants={"mean":[0.8,0,0,0], "std":[0.31,0.391,0.356,0.185]},
                          label_description="[rho],[u,v],[p]", transpose=False)
metrics = evaluate.evaluate_model(model, [batch], dataset=dataset,
                                 dataset_name="fluids.incompressible.PiecewiseConstants",
                                 device="cpu", ar_steps=2)
torch.save({"prediction": prediction, "metrics": metrics,
            "calibration_fingerprint": metadata["calibration"]["fingerprint"]}, sys.argv[3])
'''


@pytest.mark.parametrize("method", METHODS)
def test_all_methods_real_scot_export_fresh_load_and_evaluate(method, tiny_scot_case, tmp_path, monkeypatch):
    model, dataset, loader, calibration_metadata = tiny_scot_case
    sample = next(iter(loader))
    missing_checkpoint = tmp_path / "original-checkpoint-is-not-present"
    activation_bits = 8 if method in {"dynamic", "static", "repq", "ptq4vit"} else 32
    args = build_parser().parse_args([
        "--method", method, "--DEV", "cpu", "--model_path", str(missing_checkpoint),
        "--save_path", str(tmp_path / method), "--dataset", "fluids.incompressible.PiecewiseConstants",
        "--data_path", str(tmp_path / "no-real-dataset"), "--traj_num", "2",
        "--batch_size", "1", "--final_time", "2", "--weight_bits", "4",
        "--activation_bits", str(activation_bits), "--init_bits", "5", "--target_bits", "4",
        "--epochs", "1", "--mc_samples", "2", "--total_steps", "2", "--reconstruction_iterations", "2",
        "--rec_batchsize", "1", "--learning_rate", "0.0001",
    ])
    # Keep the real Hessian interval-search code but bound its integration size.
    args.ptq_search_rounds = 1
    args.ptq_candidates = 3
    expected = {}
    original_save = artifacts.save_quantized_artifact
    def capture_then_save(result, path, metadata, **kwargs):
        expected["prediction"] = evaluate.autoregressive_predict(
            result.model.eval(), sample, device="cpu", ar_steps=2).detach().cpu()
        return original_save(result, path, metadata, **kwargs)
    monkeypatch.setattr(artifacts, "save_quantized_artifact", capture_then_save)
    def forbidden(*args, **kwargs):
        raise AssertionError("An injected integration loader must not reconstruct calibration data")
    monkeypatch.setattr(calibration, "build_calibration_loader", forbidden)

    destination, metadata = run_quantization(
        args, model=model, calibration_loader=loader, calibration_metadata=calibration_metadata)
    assert not missing_checkpoint.exists()
    for name in ("compressed_model.pt", "metadata.json", "quantization_state.pt"):
        assert (destination / name).is_file()
    saved_meta = json.loads((destination / "metadata.json").read_text())
    assert saved_meta["method"] == method
    assert saved_meta["evaluation_performed"] is False
    assert saved_meta["calibration"] == calibration_metadata
    assert saved_meta["model_sha256"] == metadata["model_sha256"]
    assert not list(destination.glob("work-*"))
    state = torch.load(destination / "quantization_state.pt", map_location="cpu", weights_only=True)
    assert isinstance(state, dict)
    if method != "fp":
        assert state

    loaded, _ = artifacts.load_quantized_artifact(destination, "cpu")
    local_prediction = evaluate.autoregressive_predict(loaded, sample, device="cpu", ar_steps=2)
    torch.testing.assert_close(local_prediction, expected["prediction"], rtol=1e-6, atol=1e-7)
    # The spatial/channel mask must work with real ScOT without passing labels.
    torch.testing.assert_close(local_prediction[:, 3], sample["pixel_values"][:, 3], rtol=0, atol=0)
    evaluation = evaluate.evaluate_model(loaded, loader, dataset=dataset,
                                         dataset_name="fluids.incompressible.PiecewiseConstants",
                                         device="cpu", ar_steps=2)
    assert evaluation["num_samples"] == 2
    assert all(torch.isfinite(torch.tensor(v)) for v in evaluation["metrics"].values())

    # Every exported quantization route must support the default SAFE-SVD path,
    # including ScOT's supervised loss/mask and the full physical metric set.
    from types import SimpleNamespace
    from evaluation.poseidon.evaluate import build_trainer, rollout as safe_rollout
    from evaluation.poseidon.physics import evaluate_and_result
    trainer = build_trainer(loaded, SimpleNamespace(DEV="cpu", batch_size=1))
    physical = evaluate_and_result(trainer, dataset,
                                   "fluids.incompressible.PiecewiseConstants", safe_rollout, 1)
    for key in ("base_loss", "momentum", "h1_error", "sobolev_s012", "vorticity_err"):
        assert torch.isfinite(torch.tensor(physical[key]))

    batch_path, output_path = tmp_path / "evaluation_input.pt", tmp_path / "fresh_output.pt"
    torch.save(sample, batch_path)
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS, str(destination), str(batch_path), str(output_path)],
        cwd=root, env=environment, capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    fresh = torch.load(output_path, map_location="cpu", weights_only=True)
    torch.testing.assert_close(fresh["prediction"], expected["prediction"], rtol=1e-6, atol=1e-7)
    assert fresh["calibration_fingerprint"] == calibration_metadata["fingerprint"]
    assert fresh["metrics"]["num_samples"] == 1


def test_sensitivity_reference_matches_legacy_poseidon_quantizer(tiny_scot_case):
    """The fixed reference must match legacy affine weight quantization itself."""
    from copy import deepcopy
    from BRECQ.poseidon import PoseidonQuantModel
    from quantization.common import forward_model
    from quantization.probabilistic import _reference_hooks, _reference_quantizers

    model, _, loader, _ = tiny_scot_case
    batch = next(iter(loader))
    legacy = PoseidonQuantModel(
        deepcopy(model),
        weight_quant_params={"n_bits": 4, "symmetric": False,
                             "channel_wise": False, "scale_method": "max"},
        act_quant_params={"n_bits": 8},
    ).eval()
    legacy.set_quant_state(True, False)
    reference_layers, reference_steps = _reference_quantizers(model, 4)
    with torch.no_grad():
        expected = forward_model(legacy, batch, "cpu")
        with _reference_hooks(reference_layers, reference_steps):
            actual = forward_model(model, batch, "cpu")
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert not any(module._forward_hooks for module in model.modules())


def test_origin_route_real_scot_without_historical_caches(tiny_scot_case, tmp_path):
    """Exercise regenerated sensitivity, original MC, export and inference."""
    from quantization.runner import parse_args
    from evaluation.poseidon.evaluate import build_trainer, rollout as safe_rollout
    from evaluation.poseidon.physics import evaluate_and_result

    model, dataset, loader, calibration_metadata = tiny_scot_case
    _, args = parse_args([
        "--origin", "--DEV", "cpu", "--model_path", str(tmp_path / "missing-original"),
        "--save_path", str(tmp_path / "origin-artifact"), "--origin_root", str(tmp_path / "no-caches"),
    ], default_method="sapq", family="sapq")
    destination, metadata = run_quantization(
        args, model=model, calibration_loader=loader, calibration_metadata=calibration_metadata)
    details = metadata["quantization"]
    assert details["origin"] is True
    assert details["sensitivity_normalization"] == "normalized"
    assert 10 <= details["epochs"] <= 21
    if details["stopped_early"]:
        assert details["avg_weight_bits"] < 4
    assert all(value is None for value in details["origin_caches"].values())
    state = torch.load(destination / "quantization_state.pt", map_location="cpu", weights_only=True)
    assert state["sensitivity_raw"] and state["sensitivity_normalized"]
    assert len(state["history"]) == details["epochs"]
    for name, sensitivity in state["sensitivity"].items():
        torch.testing.assert_close(sensitivity, state["sensitivity_normalized"][name])
    loaded, _ = artifacts.load_quantized_artifact(destination, "cpu")
    trainer = build_trainer(loaded, args)
    physical = evaluate_and_result(trainer, dataset, "fluids.incompressible.PiecewiseConstants", safe_rollout, 1)
    assert torch.isfinite(torch.tensor(physical["base_loss"]))
