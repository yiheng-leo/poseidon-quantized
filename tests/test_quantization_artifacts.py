import json
import subprocess
import sys

import pytest
import torch
from torch import nn

from quantization.artifacts import load_quantized_artifact, save_quantized_artifact
from quantization.common import QuantizationResult, forward_model, select_quantizable_layers
from quantization.quantize import build_parser, validate_args


def test_standalone_artifact_roundtrip_and_fresh_process(tmp_path):
    model = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2)).eval()
    x = torch.randn(4, 3)
    expected = model(x).detach()
    save_quantized_artifact(QuantizationResult(model, {"effective_bits": 4.2}, {"step": torch.ones(2)}),
                            tmp_path, {"calibration": {"fingerprint": "example"}})
    restored, metadata = load_quantized_artifact(tmp_path)
    torch.testing.assert_close(restored(x), expected)
    assert metadata["calibration"]["fingerprint"] == "example"
    assert metadata["quantization"]["effective_bits"] == 4.2
    torch.save({"x": x, "expected": expected}, tmp_path / "reference.pt")
    subprocess.run([sys.executable, "-c", """
import sys, torch
from pathlib import Path
from quantization.artifacts import load_quantized_artifact
p = Path(sys.argv[1])
model, _ = load_quantized_artifact(p)
reference = torch.load(p / 'reference.pt', weights_only=True)
torch.testing.assert_close(model(reference['x']), reference['expected'])
""", str(tmp_path)], check=True)


def test_artifact_detects_incomplete_or_mismatched_output(tmp_path):
    result = QuantizationResult(nn.Linear(2, 2))
    save_quantized_artifact(result, tmp_path, {})
    with pytest.raises(FileExistsError):
        save_quantized_artifact(result, tmp_path, {})
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["model_sha256"] = "not-the-model"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="checksum"):
        load_quantized_artifact(tmp_path)


def test_layer_policy_matches_existing_norm_time_exclusions():
    model = nn.ModuleDict({"query": nn.Linear(3, 2), "norm": nn.Linear(1, 2),
                           "time_projection": nn.Linear(1, 2), "patch_merge": nn.Linear(3, 2)})
    assert set(select_quantizable_layers(model)) == {"query", "patch_merge"}


def test_forward_preserves_known_mask_without_supervised_targets():
    class StrictModel(nn.Module):
        def forward(self, pixel_values, time=None):
            return pixel_values * 2
    x = torch.ones(2, 4, 3, 3)
    batch = {"pixel_values": x, "labels": torch.full_like(x, 99),
             "pixel_mask": torch.tensor([[False, False, False, True]] * 2)}
    y = forward_model(StrictModel(), batch, "cpu")
    torch.testing.assert_close(y[:, :3], x[:, :3] * 2)
    torch.testing.assert_close(y[:, 3], x[:, 3])


def test_cli_retains_mysvd_calibration_defaults_and_rejects_unsupported_activation():
    parser = build_parser()
    args = validate_args(parser.parse_args(["--method", "sapq", "--DEV", "cpu"]))
    assert (args.traj_num, args.random_calibration, args.calibration_seed) == (64, "n", 42)
    assert (args.final_time, args.time_back) == (20, 0)
    assert (args.init_bits, args.target_bits, args.sensitivity_normalization) == (9, 8, "raw")
    assert "init9_target8" in args.save_path
    args.activation_bits = 8
    with pytest.raises(ValueError, match="weight-only"):
        validate_args(args)
