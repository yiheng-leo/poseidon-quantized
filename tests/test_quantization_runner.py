"""Public method entry, file configuration, and optional-calibration contracts."""
import json

import pytest
import torch

from quantization.runner import build_parser, parse_args, run_quantization, validate_args


def test_config_defaults_cli_override_and_variant_resolution(tmp_path):
    path = tmp_path / "sapq.json"
    path.write_text(json.dumps({"method": "sapq-block", "traj_num": 5, "target_bits": 4,
                                "sensitivity_normalization": "normalized"}))
    _, args = parse_args(["--config", str(path), "--traj_num", "7", "--variant", "network"],
                         default_method="sapq", family="sapq")
    args = validate_args(args)
    assert args.method == "sapq-network"
    assert (args.traj_num, args.target_bits, args.sensitivity_normalization) == (7, 4, "normalized")
    assert args.activation_bits == 32


def test_configuration_errors_and_cross_method_dispatch_fail_early(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text('{"traj_numm": 64}')
    with pytest.raises(SystemExit):
        parse_args(["--config", str(path)], default_method="ppq", family="ppq")
    with pytest.raises(SystemExit):
        parse_args(["--method", "sapq"], default_method="ppq", family="ppq")
    with pytest.raises(SystemExit):
        parse_args(["--variant", "block"], default_method="ppq", family="ppq")
    with pytest.raises(SystemExit):
        parse_args(["--method", "ppq", "--variant", "global"], default_method="sapq", family="sapq")
    with pytest.raises(SystemExit):
        parse_args(["--method", "sapq-network", "--variant", "block"], default_method="sapq", family="sapq")


def test_explicit_method_overrides_file_variant(tmp_path):
    path = tmp_path / "sapq.json"
    path.write_text('{"variant": "block"}')
    _, args = parse_args(["--config", str(path), "--method", "sapq-network"],
                         default_method="sapq", family="sapq")
    assert (args.method, args.variant) == ("sapq-network", "network")


@pytest.mark.parametrize("value", [64.9, True, None])
def test_config_rejects_inexact_trajectory_counts(tmp_path, value):
    path = tmp_path / "invalid-count.json"
    path.write_text(json.dumps({"traj_num": value}))
    with pytest.raises(SystemExit):
        parse_args(["--config", str(path)], default_method="sapq", family="sapq")


def test_repq_default_and_explicit_activation_policy():
    _, args = parse_args([], default_method="repq", family="repq")
    assert validate_args(args).activation_bits == 8
    _, args = parse_args(["--activation_bits", "32"], default_method="repq", family="repq")
    with pytest.raises(ValueError, match="RepQ"):
        validate_args(args)


@pytest.mark.parametrize("method", ["fp", "uniform", "dynamic", "static"])
def test_methods_without_offline_calibration_do_not_open_data(method, tmp_path, monkeypatch):
    import quantization.calibration
    from quantization.artifacts import load_quantized_artifact

    def forbidden(*args, **kwargs):
        raise AssertionError("This configuration must not load a calibration dataset")

    monkeypatch.setattr(quantization.calibration, "build_calibration_loader", forbidden)
    args = build_parser(method).parse_args([
        "--DEV", "cpu", "--activation_bits", "32", "--data_path", str(tmp_path / "absent-data"),
        "--model_path", str(tmp_path / "absent-original"), "--save_path", str(tmp_path / method),
    ])
    model = torch.nn.Sequential(torch.nn.Linear(3, 2)).eval()
    destination, metadata = run_quantization(args, model=model)
    assert metadata["calibration"]["selection"] == "not_required"
    restored, _ = load_quantized_artifact(destination)
    assert restored(torch.ones(1, 3)).shape == (1, 2)
