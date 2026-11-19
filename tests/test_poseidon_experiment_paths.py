"""Experiment-table names, explicit overrides, and relocated artifact contracts."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from quantization.paths import default_artifact_path, default_evaluation_path, poseidon_run_name
from quantization.runner import parse_args, validate_args


def settings(**changes):
    values = dict(method="sbpq", dataset="wave.Gaussians", traj_num=16, final_time=10,
                  target_bits=8.0, init_bits=9, weight_bits=4, random_calibration="n",
                  calibration_seed=42, seed=7)
    return {**values, **changes}


@pytest.mark.parametrize("method,bits", [("sbpq", 8), ("ppq", 8), ("ppq-layerwise", 8),
                                        ("brecq", 4), ("uniform", 4)])
@pytest.mark.parametrize("dataset,label", [
    ("fluids.incompressible.PiecewiseConstants", "NS-PwC"),
    ("fluids.compressible.RichtmyerMeshkov", "CE-RM"), ("wave.Gaussians", "Wave-Gauss")])
def test_configured_bits_and_dataset_labels(method, bits, dataset, label):
    result = default_artifact_path(settings(method=method, dataset=dataset))
    assert result == Path(f"quantization_artifacts/{label}/{method}-traj16-ft10-bit{bits}-default")


def test_seed_label_tracks_calibration_sampling_and_not_optimizer_rng():
    fixed = settings(calibration_seed=123, seed=9)
    assert poseidon_run_name(fixed) == "sbpq-traj16-ft10-bit8-default"
    assert poseidon_run_name({**fixed, "seed": 999, "calibration_seed": 456}) == poseidon_run_name(fixed)
    sampled = {**fixed, "random_calibration": "y"}
    assert poseidon_run_name(sampled) == "sbpq-traj16-ft10-bit8-seed123"
    assert poseidon_run_name({**sampled, "seed": 999}) == poseidon_run_name(sampled)
    assert poseidon_run_name({**sampled, "calibration_seed": 456}).endswith("-seed456")


def test_default_and_explicit_cli_paths_are_idempotent(tmp_path):
    options = ["--dataset", "wave.Gaussians", "--traj_num", "16", "--final_time", "10",
               "--random_calibration", "y", "--calibration_seed", "3", "--seed", "99"]
    _, args = parse_args(options, default_method="ppq", family="ppq")
    validate_args(args)
    assert args.save_path == "quantization_artifacts/Wave-Gauss/ppq-traj16-ft10-bit8-seed3"
    assert validate_args(args).save_path == args.save_path
    custom = tmp_path / "my-result"
    _, explicit = parse_args(options + ["--save_path", str(custom), "--output_root", str(tmp_path / "parent")],
                             default_method="ppq", family="ppq")
    assert validate_args(explicit).save_path == str(custom)
    assert validate_args(explicit).save_path == str(custom)


def test_legacy_artifact_metadata_drives_evaluation_name(tmp_path):
    directory = tmp_path / "legacy-long-name_cfg-0123456789"
    directory.mkdir()
    model = directory / "compressed_model.pt"
    model.touch()
    record = {"settings": settings(random_calibration="y", calibration_seed=2),
              "quantization": {"avg_weight_bits": 3.55}}
    (directory / "metadata.json").write_text(json.dumps(record))
    expected = Path("quantization_evaluation/Wave-Gauss/sbpq-traj16-ft10-bit8-seed2.json")
    assert default_evaluation_path(directory, "wave.Gaussians") == expected
    assert default_evaluation_path(model, "wave.Gaussians") == expected


@pytest.mark.parametrize("explicit", [False, True])
def test_evaluation_writes_automatic_or_explicit_result_path(tmp_path, monkeypatch, explicit):
    from evaluation import evaluate
    from evaluation.poseidon import evaluate as physical
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "custom-model-location"
    directory.mkdir()
    (directory / "metadata.json").write_text(json.dumps({"settings": settings()}))
    result = {"avg_weight_bits": 7.93, "metrics": {}}
    monkeypatch.setattr(physical, "run", lambda args: deepcopy(result))
    options = ["--model_path", str(directory), "--dataset", "wave.Gaussians",
               "--data_path", str(tmp_path), "--final_time", "15"]
    expected = Path("quantization_evaluation/Wave-Gauss/sbpq-traj16-ft10-bit8-default.json")
    if explicit:
        expected = tmp_path / "chosen" / "result.json"
        options += ["--output_path", str(expected)]
    assert evaluate.main(options) == result
    assert json.loads(expected.read_text()) == result


def test_renamed_resume_keeps_saved_configuration(tmp_path):
    from quantization.experiments import effective_settings, register_experiment
    old = tmp_path / "legacy-long-run"
    _, args = parse_args(["--save_path", str(old), "--DEV", "cpu"],
                         default_method="ppq", family="ppq")
    validate_args(args)
    saved = register_experiment(args)
    (old / "ppq_checkpoint.pt").touch()
    new = tmp_path / "ppq-traj64-ft20-bit8-default"
    old.rename(new)
    _, resumed = parse_args(["--resume", str(new)], default_method="ppq", family="ppq")
    validate_args(resumed)
    assert Path(resumed.save_path) == new
    assert effective_settings(resumed) == saved["settings"]
    assert register_experiment(resumed) == saved


def test_original_checkpoints_and_other_model_families_keep_output_behavior(tmp_path):
    assert default_evaluation_path(None, "wave.Gaussians") is None
    assert default_evaluation_path(tmp_path, "wave.Gaussians") is None
    (tmp_path / "metadata.json").write_text(json.dumps({"settings": settings(model_type="vicon")}))
    assert default_evaluation_path(tmp_path, "wave.Gaussians") is None
