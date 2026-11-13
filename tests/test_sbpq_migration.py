"""SBPQ public contract, disk calibration, materialization, and restart integration."""
from copy import deepcopy
from pathlib import Path
import json

import pytest
import torch
from torch.utils.data import DataLoader

from quantization.artifacts import load_quantized_artifact
from quantization.runner import parse_args, validate_args, run_quantization
from SBPQ.resume import fingerprint
from SBPQ.storage import DiskSequence
from test_quantization_scot_pipeline import tiny_scot_case


def options(path, steps, extra=()):
    return validate_args(parse_args([
        '--DEV', 'cpu', '--model_path', '/missing/unused/checkpoint', '--save_path', str(path),
        '--total_steps', str(steps), '--save_every', '1', '--mc_samples', '2', *extra,
    ], default_method='sbpq', family='sbpq')[1])


def test_defaults_config_and_generic_entry_agree():
    root = Path(__file__).resolve().parents[1]
    for argv, kwargs in [([], {'default_method': 'sbpq', 'family': 'sbpq'}),
                         (['--method', 'sbpq'], {}),
                         (['--config', str(root / 'configs/sbpq/default.json')], {})]:
        args = validate_args(parse_args(argv, **kwargs)[1])
        assert (args.init_bits, args.target_bits, args.beta_kappa, args.mc_samples) == (8, 8, 150, 10)
        assert (args.learning_rate, args.total_steps, args.prior_scale) == (3.75e-5, 800, 1)
        assert (args.sobolev_order, args.sobolev_order_weights, args.seed) == (2, 1, 42)
        assert (args.traj_num, args.final_time, args.time_back, args.batch_size) == (64, 20, 0, 2)
    args = validate_args(parse_args(['--mc_samples', '3', '--beta_kappa', '130'],
                                    default_method='sbpq', family='sbpq')[1])
    assert (args.mc_samples, args.beta_kappa) == (3, 130)


@pytest.mark.parametrize('extra', [('--sobolev_order_weights', '1,1'), ('--minimum_bits', '9'),
                                   ('--maximum_bits', '7'), ('--beta_kappa', '0'),
                                   ('--total_steps', '0'), ('--activation_bits', '8')])
def test_invalid_sbpq_settings_rejected_before_loading(extra, tmp_path):
    with pytest.raises(ValueError):
        options(tmp_path, 1, extra)


def test_disk_sequences_have_same_identity_as_lists_without_loading_all_items(tmp_path):
    values = [{'pixel_values': torch.ones(2, 3)}, {'pixel_values': torch.zeros(1, 3)}]
    paths = []
    for index, value in enumerate(values):
        path = tmp_path / f'{index}.pt'
        torch.save(value, path)
        paths.append(path)
    sequence = DiskSequence(paths)
    assert fingerprint(sequence) == fingerprint(values)
    assert len(sequence[:1]) == 1
    torch.testing.assert_close(sequence[-1], values[-1])
    with pytest.raises(IndexError):
        sequence[len(sequence)]


def test_adapter_sensitivity_loss_matches_author_runner(tiny_scot_case):
    from types import SimpleNamespace
    from SBPQ.quantize import sensitivity_loss
    from SBPQ.poseidon.run_sbpq_network_global import build_sensitivity_loss_function
    from SBPQ.poseidon.config import SBPQConfig
    model, dataset, loader, metadata = tiny_scot_case
    args = SimpleNamespace(dataset='fluids.incompressible.PiecewiseConstants', sobolev_order=2,
                           sobolev_order_weights=1., sobolev_norm='l1', sobolev_transpose=False,
                           task_loss_weight=1., sobolev_loss_weight=1.)
    batch = next(iter(loader))
    prediction = (batch['labels'] + torch.randn_like(batch['labels']) * .01).requires_grad_()
    cfg = SBPQConfig(sobolev_order=2, sobolev_order_weights=1.)
    ours = sensitivity_loss(args, dataset)(prediction, batch['labels'])
    original = build_sensitivity_loss_function(cfg, dataset)(prediction, batch['labels'])
    torch.testing.assert_close(ours, original, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(ours, prediction), torch.autograd.grad(original, prediction),
                               rtol=0, atol=0)


def test_export_and_resume_match_continuous_with_full_calibration(tiny_scot_case, tmp_path, monkeypatch):
    from quantization import calibration
    from SBPQ import quantize as adapter
    from SBPQ.poseidon.evaluation import _make_weight_quantization_hook
    model, dataset, loader, metadata = tiny_scot_case
    template = deepcopy(model)
    original_sensitivity = adapter.compute_poseidon_block_sensitivity
    calls = []
    def capture(*args, **kwargs):
        calls.append(True)
        return original_sensitivity(*args, **kwargs)
    monkeypatch.setattr(adapter, 'compute_poseidon_block_sensitivity', capture)
    def forbidden(*args, **kwargs):
        raise AssertionError('Compression attempted to construct another dataset')
    monkeypatch.setattr(calibration, 'build_calibration_loader', forbidden)
    from SBPQ.trainer import SBPQTrainer
    continuous_args = options(tmp_path / 'continuous', 4)
    continuous, _ = run_quantization(continuous_args, model=deepcopy(template),
                                    calibration_loader=loader, calibration_metadata=metadata)
    interrupted_args = options(tmp_path / 'interrupted', 4)
    original_step = SBPQTrainer.train_step
    attempts = []
    def interrupt_after_checkpoint(self, *a, **kw):
        if len(attempts) == 2:
            raise KeyboardInterrupt
        attempts.append(True)
        return original_step(self, *a, **kw)
    monkeypatch.setattr(SBPQTrainer, 'train_step', interrupt_after_checkpoint)
    interrupted_args.save_every = 1
    with pytest.raises(KeyboardInterrupt):
        run_quantization(interrupted_args, model=deepcopy(template), calibration_loader=loader,
                         calibration_metadata=metadata)
    monkeypatch.setattr(SBPQTrainer, 'train_step', original_step)
    interrupted = Path(interrupted_args.save_path)
    run_quantization(options(interrupted, 4, ('--resume', str(interrupted))), model=deepcopy(template),
                     calibration_loader=loader, calibration_metadata=metadata)
    assert len(calls) == 2  # Restart reuses only content-verified sensitivity.
    a = torch.load(continuous / 'sbpq_checkpoint.pt', weights_only=True)
    b = torch.load(interrupted / 'sbpq_checkpoint.pt', weights_only=True)
    torch.testing.assert_close(a['optimized_step_sizes'], b['optimized_step_sizes'], rtol=0, atol=0)
    torch.testing.assert_close(a['optimizer_state_dict'], b['optimizer_state_dict'], rtol=0, atol=0)
    assert a['history'] == b['history']
    assert b['completed_steps'] == 4
    assert [entry['batch_index'] for entry in b['history']] == [0, 1, 0, 1]
    exported, meta = load_quantized_artifact(interrupted)
    assert meta['calibration'] == metadata
    assert meta['evaluation_performed'] is False
    assert meta['quantization']['calibration_samples'] == len(dataset)
    assert meta['quantization']['calibration_batches'] == len(loader)
    assert meta['quantization']['sensitivity_batches'] == len(loader)
    assert not any(module._forward_hooks for module in exported.modules())
    assert not list(interrupted.glob('work-*'))
    state = torch.load(interrupted / 'quantization_state.pt', weights_only=True)
    hooks = [template.get_submodule(name).register_forward_hook(_make_weight_quantization_hook(step))
             for name, step in state['steps'].items()]
    batch = next(iter(loader))
    try:
        with torch.no_grad():
            torch.testing.assert_close(exported(**batch).output, template(**batch).output,
                                       rtol=1e-6, atol=1e-7)
    finally:
        for hook in hooks:
            hook.remove()
    # Saved states do not refer to the deleted temporary data paths.
    assert 'work-' not in repr(a['training_signature'])
    # Same run directory and new calibration bytes must not silently reuse old priors.
    altered = deepcopy(dataset)
    altered.inputs += .25
    with pytest.raises(ValueError, match='Calibration data changed|preparation does not match'):
        run_quantization(options(interrupted, 4, ('--resume', str(interrupted))),
                         model=deepcopy(model), calibration_loader=DataLoader(altered, batch_size=1),
                         calibration_metadata={**metadata, 'fingerprint': 'changed'})


def test_public_runner_builds_only_complete_training_trajectories(tiny_scot_case, tmp_path, monkeypatch):
    from quantization import calibration
    from torch.utils.data import Dataset
    model, base, _, _ = tiny_scot_case
    class CompleteTrajectories(Dataset):
        constants = base.constants
        resolution = base.resolution
        transpose = base.transpose
        label_description = base.label_description
        num_trajectories = 2
        time_indices = [(0, 0), (0, 1), (1, 1)]
        def __len__(self):
            return 6
        def __getitem__(self, index):
            return base[index % len(base)]
    calls = []
    def factory(**kwargs):
        calls.append(kwargs)
        assert kwargs['which'] == 'train'
        assert kwargs['num_trajectories'] == 2
        assert kwargs['time_step_size'] == 1
        assert kwargs['fix_input_to_time_step'] is None
        assert kwargs['max_num_time_steps'] == 1
        return CompleteTrajectories()
    monkeypatch.setattr(calibration, 'get_dataset', factory)
    args = options(tmp_path / 'artifact', 2, ('--traj_num', '2', '--final_time', '1', '--batch_size', '4'))
    _, metadata = run_quantization(args, model=model)
    assert len(calls) == 1
    assert metadata['calibration']['trajectory_indices'] == [0, 1]
    assert metadata['calibration']['num_samples'] == 6
    assert metadata['calibration']['samples_per_trajectory'] == 3
    assert metadata['calibration']['loader']['drop_last'] is False
    assert metadata['quantization']['calibration_samples'] == 6
    assert metadata['quantization']['sensitivity_batches'] == 2  # Includes final B=2.
    assert metadata['quantization']['optimization_unique_batches'] == 2
