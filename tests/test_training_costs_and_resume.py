"""Exact restart, immutable names and real resource accounting for production routes."""
from copy import deepcopy
import json
from pathlib import Path
import time

import pytest
import torch
from torch.utils.data import DataLoader

from quantization import runner
from quantization.costs import CostTracker, track, phase
from quantization.checkpoints import fingerprint
from test_quantization_baselines import ToyPhysicalModel
from test_quantization_scot_pipeline import tiny_scot_case


def toy_case():
    torch.manual_seed(147)
    model = ToyPhysicalModel().eval()
    generator = torch.Generator().manual_seed(123)
    batches = [{'pixel_values': torch.randn(1, 3, generator=generator),
                'labels': torch.randn(1, 2, generator=generator)} for _ in range(3)]
    return model, batches, {'dataset': 'toy', 'num_samples': 3, 'num_trajectories': 3,
                             'selection': 'injected', 'fingerprint': fingerprint(batches)}


def arguments(method, root, extra=()):
    return runner.parse_args(['--DEV', 'cpu', '--model_path', '/missing/FP', '--data_path', '/missing/data',
                              '--save_path', str(root), '--seed', '7', '--batch_size', '1',
                              '--epochs', '2', '--mc_samples', '1', '--save_every', '1',
                              '--reconstruction_iterations', '3', '--rec_batchsize', '2',
                              '--profile_interval', '0.01', '--total_steps', '3', *extra],
                             default_method=method, family='ppq' if method.startswith('ppq') else method)[1]


def test_parameter_names_are_stable_complete_and_bounded(tmp_path):
    from quantization.experiments import effective_settings, config_digest
    base = runner.validate_args(arguments('sbpq', tmp_path))
    assert Path(base.save_path).parent == tmp_path
    assert 'sbpq_i8_t8_k150' in Path(base.save_path).name
    assert 'cfg-' in Path(base.save_path).name
    assert len(Path(base.save_path).name.encode()) < 255
    assert runner.validate_args(base).save_path == base.save_path  # Idempotent validation.
    for option, value in [('--beta_kappa', '130'), ('--eta', '0.000002'), ('--batch_size', '2'),
                           ('--sobolev_loss_weight', '2'), ('--percentile', '0.0002'),
                           ('--calibration_seed', '8'), ('--total_steps', '4')]:
        changed = runner.validate_args(arguments('sbpq', tmp_path, (option, value)))
        assert changed.save_path != base.save_path
    same = runner.validate_args(arguments('sbpq', tmp_path, ('--save_every', '2')))
    assert same.save_path == base.save_path  # Operational checkpoint cadence is not a training hyperparameter.
    assert base._experiment_id == config_digest(effective_settings(base))


@pytest.mark.parametrize('method', ['ppq', 'ppq-layerwise', 'brecq'])
def test_interrupted_run_matches_continuous_weights_history_and_rng(method, tmp_path, monkeypatch):
    model, loader, metadata = toy_case()
    continuous_args = arguments(method, tmp_path / 'continuous')
    continuous, _ = runner.run_quantization(continuous_args, model=deepcopy(model),
                                           calibration_loader=loader, calibration_metadata=metadata)
    if method == 'brecq':
        import BRECQ.resume as implementation
        stop_step = 4  # First completed block, then interrupt inside the second.
    else:
        import PPQ.quantize as implementation
        stop_step = 1  # Incomplete epoch, partial aggregate losses must survive.
    original_save = implementation.save_checkpoint
    interrupted_args = arguments(method, tmp_path / 'interrupted')
    def interrupt(payload, path):
        original_save(payload, path)
        if Path(path).name == method.split('-')[0] + '_checkpoint.pt' and payload.get('completed_steps') == stop_step:
            raise KeyboardInterrupt('simulated interruption after an atomic checkpoint')
    monkeypatch.setattr(implementation, 'save_checkpoint', interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.run_quantization(interrupted_args, model=deepcopy(model),
                                calibration_loader=loader, calibration_metadata=metadata)
    interrupted = Path(interrupted_args.save_path)
    assert (interrupted / 'work').is_dir()
    if method == 'brecq':
        def forbidden(*args, **kwargs):
            raise AssertionError('Resume must reuse every initialized weight scale')
        monkeypatch.setattr(implementation, '_initialize_brecq_weight_quantizer', forbidden)
    monkeypatch.setattr(implementation, 'save_checkpoint', original_save)
    resumed_args = runner.parse_args(['--resume', str(interrupted)], default_method=method,
                                    family='ppq' if method.startswith('ppq') else method)[1]
    resumed, resumed_meta = runner.run_quantization(resumed_args, model=deepcopy(model),
                                                   calibration_loader=loader, calibration_metadata=metadata)
    assert resumed == interrupted
    a = torch.load(continuous / 'quantization_state.pt', weights_only=True)
    b = torch.load(resumed / 'quantization_state.pt', weights_only=True)
    assert fingerprint(a) == fingerprint(b)
    filename = method.split('-')[0] + '_checkpoint.pt'
    state_a = torch.load(continuous / filename, weights_only=True)
    state_b = torch.load(resumed / filename, weights_only=True)
    assert fingerprint(state_a['rng_state']) == fingerprint(state_b['rng_state'])
    assert state_a['completed_steps'] == state_b['completed_steps']
    assert not (resumed / 'work').exists()
    costs = json.loads((resumed / 'costs.json').read_text())
    assert len(costs['sessions']) == 2
    assert costs['sessions'][0]['status'] == 'interrupted'
    assert costs['sessions'][1]['status'] == 'completed'
    assert costs['cumulative']['optimization_seconds'] > 0
    assert costs['cumulative']['cpu_peak_rss_bytes'] > 0
    assert costs['cumulative']['gpu_peak_allocated_bytes'] is None
    assert costs['cumulative']['phases']['checkpoint_saving']['wall_seconds'] > 0
    assert resumed_meta['costs_summary'] == costs['cumulative']
    # Model/data/batch settings are restored from the manifest; explicit changes reject early.
    changed = runner.parse_args(['--resume', str(resumed), '--batch_size', '4'], default_method=method,
                                family='ppq' if method.startswith('ppq') else method)[1]
    with pytest.raises(ValueError, match='hyperparameters changed'):
        runner.run_quantization(changed, model=deepcopy(model), calibration_loader=loader,
                                calibration_metadata=metadata)


def test_brecq_recovers_partially_initialized_scales(tmp_path, monkeypatch):
    import BRECQ.resume as implementation
    model, loader, metadata = toy_case()
    args = arguments('brecq', tmp_path / 'partial')
    original_save = implementation.save_checkpoint
    def interrupt(payload, path):
        original_save(payload, path)
        if payload.get('phase') == 'initialization' and payload.get('initialized_layers') == 1:
            raise KeyboardInterrupt
    monkeypatch.setattr(implementation, 'save_checkpoint', interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.run_quantization(args, model=deepcopy(model), calibration_loader=loader, calibration_metadata=metadata)
    monkeypatch.setattr(implementation, 'save_checkpoint', original_save)
    original_initialize = implementation._initialize_brecq_weight_quantizer
    calls = []
    def capture(module):
        calls.append(module)
        original_initialize(module)
    monkeypatch.setattr(implementation, '_initialize_brecq_weight_quantizer', capture)
    resumed = runner.parse_args(['--resume', args.save_path], default_method='brecq', family='brecq')[1]
    runner.run_quantization(resumed, model=deepcopy(model), calibration_loader=loader, calibration_metadata=metadata)
    assert len(calls) == 1  # Toy model has two layers; the first one was checkpointed.


def test_sbpq_resume_retains_exact_training_and_records_cost(tiny_scot_case, tmp_path, monkeypatch):
    from SBPQ.trainer import SBPQTrainer
    model, _, loader, metadata = tiny_scot_case
    continuous, _ = runner.run_quantization(arguments('sbpq', tmp_path / 'continuous'), model=deepcopy(model),
                                            calibration_loader=loader, calibration_metadata=metadata)
    args = arguments('sbpq', tmp_path / 'interrupted')
    original = SBPQTrainer.train_step
    calls = []
    def interrupt(self, *a, **kw):
        if calls:
            raise KeyboardInterrupt
        calls.append(True)
        return original(self, *a, **kw)
    monkeypatch.setattr(SBPQTrainer, 'train_step', interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.run_quantization(args, model=deepcopy(model), calibration_loader=loader, calibration_metadata=metadata)
    monkeypatch.setattr(SBPQTrainer, 'train_step', original)
    resumed_args = runner.parse_args(['--resume', args.save_path], default_method='sbpq', family='sbpq')[1]
    resumed, _ = runner.run_quantization(resumed_args, model=deepcopy(model), calibration_loader=loader,
                                        calibration_metadata=metadata)
    a = torch.load(continuous / 'quantization_state.pt', weights_only=True)
    b = torch.load(resumed / 'quantization_state.pt', weights_only=True)
    assert fingerprint(a) == fingerprint(b)
    cost = json.loads((resumed / 'costs.json').read_text())
    assert cost['cumulative']['optimization_seconds'] > 0
    assert len(cost['sessions']) == 2


def test_stage_costs_are_exclusive_and_survive_restart(tmp_path):
    path = tmp_path / 'costs.json'
    tracker = CostTracker(path, 'cpu', interval=.005)
    with track(tracker), phase('optimization'):
        time.sleep(.02)
        with phase('checkpoint_saving'):
            time.sleep(.04)
        time.sleep(.02)
    tracker.close('completed')
    report = json.loads(path.read_text())
    phases = report['cumulative']['phases']
    assert phases['optimization']['wall_seconds'] >= .035
    assert phases['checkpoint_saving']['wall_seconds'] >= .035
    assert sum(value['wall_seconds'] for value in phases.values()) <= report['cumulative']['wall_seconds'] + .001
    assert report['cumulative']['gpu_peak_reserved_bytes'] is None
    before = report['cumulative']['wall_seconds']
    resumed = CostTracker(path, 'cpu', interval=.005, resume=True)
    resumed.close('completed')
    report = json.loads(path.read_text())
    assert len(report['sessions']) == 2
    assert report['cumulative']['wall_seconds'] >= before


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_peak_measures_allocator_memory(tmp_path):
    tracker = CostTracker(tmp_path / 'costs.json', 'cuda', interval=.01)
    with track(tracker), phase('optimization'):
        tensor = torch.ones(2 * 1024 * 1024, device='cuda')
    tracker.close('completed')
    summary = tracker.snapshot()['cumulative']['phases']['optimization']
    assert summary['gpu_peak_allocated_bytes'] >= tensor.numel() * tensor.element_size()
    assert summary['gpu_peak_reserved_bytes'] >= summary['gpu_peak_allocated_bytes']


def test_legacy_sbpq_path_and_unmeasured_historical_cost(tiny_scot_case, tmp_path):
    import hashlib
    import subprocess
    try:
        old_adapter = subprocess.check_output(['git', 'show', '292df51:SBPQ/quantize.py'], stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        pytest.skip('Historical adapter reference absent in shallow checkout')
    model, _, loader, metadata = tiny_scot_case
    directory, _ = runner.run_quantization(arguments('sbpq', tmp_path / 'legacy'), model=deepcopy(model),
                                           calibration_loader=loader, calibration_metadata=metadata)
    original = torch.load(directory / 'quantization_state.pt', weights_only=True)
    prepared = torch.load(directory / 'sbpq_preparation.pt', weights_only=True)
    prepared['identity']['implementation']['SBPQ/quantize.py'] = hashlib.sha256(old_adapter).hexdigest()
    torch.save(prepared, directory / 'sbpq_preparation.pt')
    for name in ['sbpq_checkpoint.pt', 'sbpq_trainer_state.pt']:
        checkpoint = torch.load(directory / name, weights_only=True)
        checkpoint['metadata']['preparation_identity'] = fingerprint(prepared['identity'])
        torch.save(checkpoint, directory / name)
    (directory / 'experiment.json').unlink()
    (directory / 'costs.json').unlink()
    # This models the old metadata/checkpoint/preparation contract, before the
    # named-run manifest and costs were introduced.
    resumed = runner.parse_args(['--resume', str(directory)], default_method='sbpq', family='sbpq')[1]
    result, _ = runner.run_quantization(resumed, model=deepcopy(model), calibration_loader=loader,
                                        calibration_metadata=metadata)
    assert result == directory
    assert fingerprint(original) == fingerprint(torch.load(result / 'quantization_state.pt', weights_only=True))
    costs = json.loads((result / 'costs.json').read_text())
    assert costs['unmeasured_history_before_first_session'] is True
    assert costs['legacy_reported_elapsed_quantization_seconds'] > 0
    assert costs['cumulative']['counters'].get('optimization_updates', 0) == 0


def test_nested_gpu_peaks_are_attributed_to_the_right_phase(tmp_path, monkeypatch):
    from types import SimpleNamespace
    values = {'allocated': 100, 'reserved': 200, 'current': 100}
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda *a: 'test CUDA allocator')
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda *a: SimpleNamespace(total_memory=1024**3))
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda *a: values['allocated'])
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda *a: values['reserved'])
    def reset(*args):
        values['allocated'], values['reserved'] = values['current'], values['current'] * 2
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', reset)
    tracker = CostTracker(tmp_path / 'costs.json', 'cuda', interval=10)
    with track(tracker), phase('optimization'):
        values.update(allocated=400, reserved=800, current=400)
        with phase('checkpoint_saving'):
            values.update(allocated=900, reserved=1800, current=400)
        values.update(allocated=600, reserved=1200, current=400)
    tracker.close('completed')
    summary = tracker.snapshot()['cumulative']
    assert summary['gpu_peak_allocated_bytes'] == 900
    assert summary['phases']['optimization']['gpu_peak_allocated_bytes'] == 600
    assert summary['phases']['checkpoint_saving']['gpu_peak_allocated_bytes'] == 900
