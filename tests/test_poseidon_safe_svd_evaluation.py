"""Protocol regression tests, including direct comparisons to SAFE-SVD-v0."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evaluation.evaluate import build_parser, main
from evaluation.poseidon import evaluate, physics


def args(*extra):
    return build_parser().parse_args([
        "--model_path", "unused.pt", "--dataset", "fluids.incompressible.PiecewiseConstants",
        "--data_path", "unused", *extra,
    ])


@pytest.fixture
def reference(monkeypatch):
    directory = Path(os.environ.get("SAFE_SVD_REFERENCE_ROOT",
                                    str(Path(__file__).resolve().parents[2] / "SAFE-SVD-v0")))
    directory = directory / "evaluation" / "poseidon"
    if not (directory / "physics.py").is_file():
        pytest.skip("Optional SAFE-SVD-v0 reference checkout is unavailable.")
    # The reference checkout is read-only evidence, including tracked caches.
    monkeypatch.setattr(sys, "dont_write_bytecode", True)

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, directory / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    monkeypatch.setitem(sys.modules, "PDE_calc", load("_safe_reference_pde", "PDE_calc.py"))
    return load("_safe_reference_physics", "physics.py")


def test_real_scot_time_indexing_keeps_full_test_split():
    from scOT.problems.base import BaseTimeDataset

    class HeldOut(BaseTimeDataset):
        def __init__(self, **kwargs):
            kwargs.pop("dataset")
            super().__init__(**kwargs)
            self.N_max, self.N_val, self.N_test = 100, 5, 3
            self.label_description = "[rho],[u,v],[p]"
            self.post_init()

        def __len__(self):
            return self.length

        def __getitem__(self, index):
            return self._idx_map(index)

    dataset = evaluate.build_test_dataset(HeldOut, args())
    assert dataset.num_trajectories == 64
    assert dataset.start == 97
    assert dataset.time_indices == [(i, j) for i in range(21) for j in range(i, 21)]
    assert len(dataset) == 3 * 231  # Not 64 samples or 64 held-out trajectories.
    previous = physics.PrevTimeDataset(dataset)
    for index in range(len(dataset)):
        trajectory, _, start, stop = dataset[index]
        expected = dataset[index - 1] if stop > start else dataset[index]
        assert previous[index] == expected
        assert previous[index][0] == trajectory


def test_time_wrapped_static_dataset_receives_no_time_kwargs():
    parsed = args("--dataset", "elliptic.poisson.Gaussians.time")
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return object()

    evaluate.build_test_dataset(factory, parsed)
    assert captured == dict(dataset=parsed.dataset, which="test", num_trajectories=64,
                            data_path="unused")


@pytest.mark.parametrize("option,value", [
    ("--split", "val"), ("--ar_steps", "3"), ("--initial_time", "1"),
    ("--physics_batch_size", "0"), ("--num_workers", "1"), ("--eval_obj", "oncomp"),
])
def test_conflicting_protocol_arguments_fail(option, value):
    with pytest.raises(ValueError):
        evaluate.validate_args(args(option, value))


@pytest.mark.parametrize("name,channels,constants", [
    ("fluids.incompressible.PiecewiseConstants", 4,
     {"mean": [0.8, 0, 0, 0], "std": [0.31, 0.391, 0.356, 0.185]}),
    ("fluids.compressible.RiemannKelvinHelmholtz", 4,
     {"mean": [1, 0, 0, 0], "std": [0.1, 0.2, 0.2, 0.1]}),
    ("wave.Layer", 2, {"mean": 0.1, "std": 0.2, "mean_c": 1, "std_c": 0.5}),
    ("elliptic.poisson.Gaussians.time", 1, {"mean_solution": 0.01, "std_solution": 0.02}),
])
def test_every_physical_metric_matches_safe_svd(reference, name, channels, constants):
    rng = np.random.default_rng(123)
    labels = rng.normal(size=(5, channels, 8, 8)).astype(np.float32)
    preds = labels + rng.normal(0, 0.03, size=labels.shape).astype(np.float32)
    prev_labels = labels * 0.9
    prev_preds = preds * 0.8
    dataset = SimpleNamespace(constants=constants, transpose=False, mean_pressure=0.0)
    trainer = SimpleNamespace(model=torch.nn.Linear(2, 3))
    calls = []

    def fake_rollout(trainer, source):
        previous = isinstance(source, (physics.PrevTimeDataset, reference.PrevTimeDataset))
        calls.append(previous)
        return (prev_preds, prev_labels, {}) if previous else (preds, labels, {"_loss": 0.125})

    ours = physics.evaluate_and_result(trainer, dataset, name, fake_rollout, physics_batch_size=2)
    theirs = reference.evaluate_and_result(trainer, dataset, name, fake_rollout, physics_batch_size=2)
    assert ours.keys() == theirs.keys()
    for key in ours:
        if isinstance(ours[key], str):
            assert ours[key] == theirs[key]
        else:
            assert ours[key] == pytest.approx(theirs[key], rel=0, abs=0)
    has_previous = "wave" in name or "fluids.compressible" in name
    assert calls == ([True, False, True, False] if has_previous else [False, False])
    # SAFE-SVD averages physical chunks equally, including the last single sample.
    chunks = [physics.get_loss_calculator(name, preds[i:i+2], labels[i:i+2],
                                        prev_preds[i:i+2], prev_labels[i:i+2], dataset).compute()
              for i in range(0, 5, 2)]
    assert ours["sobolev_s0"] == np.mean([chunk["sobolev_s0"] for chunk in chunks])


@pytest.mark.parametrize("eval_obj", ["ongt", "oncomp"])
def test_saved_real_scot_baseline_and_artifact_end_to_end(tmp_path, monkeypatch, eval_obj):
    from scOT.model import ScOT, ScOTConfig
    from scOT.problems import base
    from scOT.problems.base import BaseTimeDataset

    class HeldOut(BaseTimeDataset):
        constants = {"mean": [0.8, 0, 0, 0], "std": [0.31, 0.391, 0.356, 0.185]}
        resolution = 8
        transpose = False
        label_description = "[rho],[u,v],[p]"

        def __init__(self, **kwargs):
            kwargs.pop("dataset")
            super().__init__(**kwargs)
            self.N_max, self.N_val, self.N_test = 100, 5, 2
            self.post_init()

        def __len__(self):
            return self.length

        def __getitem__(self, index):
            trajectory, delta, _, _ = self._idx_map(index)
            generator = torch.Generator().manual_seed(trajectory + 55)
            x = torch.randn(4, 8, 8, generator=generator)
            labels = x + delta / 20
            mask = torch.zeros_like(labels, dtype=torch.bool)
            mask[3] = True
            labels[3] = 7.0  # ScOT must copy the label's fixed channel, as in SAFE-SVD.
            return {"pixel_values": x, "labels": labels, "time": delta / 20,
                    "pixel_mask": mask}

    monkeypatch.setattr(base, "get_dataset", lambda **kw: HeldOut(**kw))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    model = ScOT(ScOTConfig(
        image_size=8, patch_size=2, embed_dim=8, depths=[1, 1], num_heads=[2, 4],
        window_size=2, skip_connections=[0, 0], num_channels=4, num_out_channels=4,
        use_conditioning=True, hidden_dropout_prob=0, attention_probs_dropout_prob=0,
        drop_path_rate=0, channel_slice_list_normalized_loss=[0, 1, 3, 4],
    )).eval()
    checkpoint = tmp_path / "origin"
    model.save_pretrained(checkpoint)
    artifact = tmp_path / "compressed_model.pt"
    torch.save(model, artifact)
    # Large artifact metadata must not leak into the evaluation report; use the
    # achieved mean bits rather than initial/target bits.
    (tmp_path / "metadata.json").write_text(json.dumps({
        "quantization": {"avg_weight_bits": 4.25, "target_bits": 4, "initial_bits": 8},
        "calibration": {"time_pairs": [[i, j] for i in range(21) for j in range(i, 21)]},
        "args": {"target_bits": 4},
    }))
    output = tmp_path / "metrics.json"
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = main([
            "--model", str(artifact), "--checkpoint", str(checkpoint),
            "--dataset", "fluids.incompressible.PiecewiseConstants",
            "--data-root", str(tmp_path), "--device", "cpu", "--final-time", "2",
            "--batch-size", "5", "--physics-batch-size", "5", "--eval-obj", eval_obj,
            "--output", str(output),
        ])
        assert set(result) == {"avg_weight_bits", "metrics"}
        assert result["avg_weight_bits"] == 4.25
        assert "L1" not in result["metrics"]
        assert not {"model_size", "linear_size", "rel_base_loss"} & result["metrics"].keys()
        for metric in result["metrics"].values():
            assert set(metric) == {"origin", "compressed", "change_percent"}
            assert metric["origin"] == metric["compressed"]
            assert metric["change_percent"] == (0.0 if metric["origin"] != 0 else None)
        assert result["metrics"]["base_loss"]["compressed"] >= 0
        if eval_obj == "oncomp":
            assert result["metrics"]["base_loss"]["compressed"] == 0
        assert json.loads(output.read_text()) == result
        assert len(output.read_text().splitlines()) < 150
        # The same artifact is also evaluable without the original checkpoint.
        if eval_obj == "ongt":
            standalone = evaluate.run(args("--model_path", str(artifact), "--final_time", "2",
                                           "--batch_size", "5", "--physics_batch_size", "5"))
            assert standalone["avg_weight_bits"] == 4.25
            for key, metric in standalone["metrics"].items():
                assert metric == {"compressed": result["metrics"][key]["compressed"]}
    finally:
        torch.set_num_threads(previous_threads)
