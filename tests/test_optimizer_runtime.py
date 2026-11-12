"""Prevent delayed optimizer import failures after costly calibration/scales."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from quantization import calibration, runner, runtime


@pytest.mark.parametrize('name', ['Adam', 'AdamW'])
def test_real_optimizer_probe_on_cpu_preserves_torch_rng(name):
    before = torch.get_rng_state().clone()
    details = runtime.check_optimizer_runtime(torch.device('cpu'), name)
    assert details['torch_version'] == str(torch.__version__)
    assert details['optimizer'] == name
    assert details['torch_python'] == torch.__file__
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize('method', sorted(runner.PROBABILISTIC_METHODS | {'brecq'}))
def test_broken_optimizer_fails_before_any_model_data_or_artifact_work(method, tmp_path, monkeypatch):
    original = ImportError("cannot import name 'get_current_stream' from 'torch._C._dynamo.utils'")
    expected_optimizer = 'AdamW' if method == 'sbpq' else 'Adam'
    seen = []
    def broken(*args, **kwargs):
        seen.append(expected_optimizer)
        raise original
    def forbidden(*args, **kwargs):
        raise AssertionError('Optimizer failure must precede expensive model/calibration work')
    monkeypatch.setattr(torch.optim, expected_optimizer, broken)
    monkeypatch.setattr(calibration, 'build_calibration_loader', forbidden)
    monkeypatch.setattr(runner, '_load_model', forbidden)
    destination = tmp_path / 'artifact'
    args = runner.parse_args(['--method', method, '--DEV', 'cpu', '--save_path', str(destination)])[1]
    with pytest.raises(RuntimeError, match='before calibration/model loading') as caught:
        runner.run_quantization(args)
    assert seen == [expected_optimizer]
    assert caught.value.__cause__ is original
    assert 'get_current_stream' in str(caught.value)
    assert str(torch.__version__) in str(caught.value)
    assert str(torch.__file__) in str(caught.value)
    assert not destination.exists()


def test_runtime_cli_reports_versions_and_exercises_adam():
    result = subprocess.run([sys.executable, '-m', 'quantization.runtime', '--DEV', 'cpu'],
                            cwd=Path(__file__).resolve().parents[1], text=True,
                            capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.rstrip().endswith('Adam OK (cpu)')
    details = json.loads(result.stdout[:result.stdout.rfind('}')+1])
    assert details['torch_native']
    assert details['device'] == 'cpu'
