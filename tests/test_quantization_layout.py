"""Compatibility checks for public evaluation commands and pre-migration models."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


def _environment(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["HF_HOME"] = str(tmp_path / "hf-cache")
    environment["TRANSFORMERS_CACHE"] = str(tmp_path / "hf-cache")
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    return root, environment


def test_legacy_evaluation_api_forwards_to_the_independent_implementation():
    from evaluation import evaluate, metrics, rollout
    from quantization import evaluate as legacy

    assert legacy.main is evaluate.main
    assert legacy.build_evaluation_loader is evaluate.build_evaluation_loader
    assert legacy.evaluate_model is evaluate.evaluate_model
    assert legacy.autoregressive_predict is rollout.autoregressive_predict
    assert legacy.denormalize_output is metrics.denormalize_output
    assert legacy.per_sample_metrics is metrics.per_sample_metrics


@pytest.mark.parametrize("module", ["evaluation.evaluate", "quantization.evaluate"])
def test_evaluation_module_commands_remain_available(module, tmp_path):
    root, environment = _environment(tmp_path)
    result = subprocess.run([sys.executable, "-m", module, "--help"],
                            cwd=root, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    for option in ("--model_path", "--dataset", "--split", "--ar_steps", "--num_trajectories"):
        assert option in result.stdout


@pytest.mark.parametrize("environment_key, default_directory, kind", [
    ("QUANTIZATION_LEGACY_ARTIFACT", "/private/tmp/ritchie-legacy-artifact", "linear"),
    ("QUANTIZATION_LEGACY_BRECQ_ARTIFACT", "/private/tmp/ritchie-legacy-brecq", "brecq"),
])
def test_pre_migration_full_model_loads_in_a_fresh_process(
        tmp_path, environment_key, default_directory, kind):
    # Immutable artifacts created before migration contain the original
    # QuantizedLinear or BRECQ QuantModule/AdaRound class paths, not new aliases.
    directory = Path(os.environ.get(environment_key, default_directory))
    if not (directory / "compressed_model.pt").is_file():
        pytest.skip(f"Set {environment_key} to the pre-migration fixture.")
    assert (directory / "reference.pt").is_file()
    script = r'''
import importlib.abc
import sys
import torch

class RejectHistoricalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"archive", "reference"}:
            raise AssertionError("Production model loading must not import historical source")
        return None

sys.meta_path.insert(0, RejectHistoricalImports())
torch.set_num_threads(1)
from pathlib import Path
from quantization.artifacts import load_quantized_artifact

directory = Path(sys.argv[1])
model, _ = load_quantized_artifact(directory, "cpu")
reference = torch.load(directory / "reference.pt", map_location="cpu", weights_only=True)
if sys.argv[2] == "linear":
    from quantization.baselines import QuantizedLinear
    assert any(isinstance(module, QuantizedLinear) for module in model.modules())
else:
    from BRECQ.quantizers import QuantModule, AdaRoundQuantizer
    assert any(isinstance(module, QuantModule) for module in model.modules())
    assert any(isinstance(module, AdaRoundQuantizer) for module in model.modules())
with torch.no_grad():
    prediction = model(reference["x"])
torch.testing.assert_close(prediction, reference["expected"], rtol=0, atol=0)
'''
    root, environment = _environment(tmp_path)
    result = subprocess.run([sys.executable, "-c", script, str(directory), kind],
                            cwd=root, env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
