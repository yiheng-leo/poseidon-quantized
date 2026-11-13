"""PPQ network/layer workflows with exact per-update restart and cost accounting."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import torch
from torch import nn

from quantization.cache import PersistentCalibration
from quantization.checkpoints import (atomic_json, cpu_tree, fingerprint, load_resume,
    rng_state, restore_rng, save_checkpoint, training_identity, validate_checkpoint, forward_cache_identity)
from quantization.common import QuantizationResult, forward_model, move_to_device, select_quantizable_layers
from quantization.costs import phase, counter
from quantization.probabilistic import (_Unit, _average_bits, _clamp, _first_tensor,
    _inputs, _load, _seed, _validate, _weight_hooks, _weight_ranges, get_lr_for_epoch, _cache_layer)
from quantization.progress import progress, progress_write
from .loss import backward_likelihood, prior as compute_prior


def _layer_paths(model, unit, paths, directory, device, identity):
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'manifest.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        candidates = [directory / name for name in saved['files']]
        if saved['identity'] != identity:
            raise ValueError('PPQ layer-cache identity changed')
        if all(path.exists() and fingerprint(_load(path)) == digest
               for path, digest in zip(candidates, saved['digests'])):
            return candidates
    result = _cache_layer(model, unit, paths, directory, device)
    atomic_json({'identity': identity, 'files': [path.name for path in result],
                 'digests': [fingerprint(_load(path)) for path in result]}, manifest)
    return result


def quantize(model, calibration_loader, args, device, work_dir):
    if args.method not in {'ppq', 'ppq-layerwise'}:
        raise ValueError(f'Unknown PPQ method: {args.method}')
    _validate(args)
    device = torch.device(device)
    model.to(device).eval()
    layers = select_quantizable_layers(model)
    if not layers:
        raise ValueError('No eligible Linear layers were found for probabilistic quantization.')
    flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    managed = bool(getattr(args, 'save_path', None))
    destination = Path(getattr(args, 'save_path', work_dir))
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / 'ppq_checkpoint.pt'
    try:
        with _seed(args.seed, device):
            with phase('identity_verification'):
                model_digest = fingerprint(model.state_dict())
            with phase('fp_output_cache'):
                calibration = PersistentCalibration(
                    calibration_loader, work_dir / 'ppq-calibration',
                    forward_cache_identity(model_digest, args.method, device), transform=_inputs,
                    build_record=lambda batch: {'clean': forward_model(model, batch, device).detach().cpu()},
                )
            identity = training_identity(model, calibration.digest, args, layers=layers, device=device, model_digest=model_digest)
            identity['clean_outputs'] = fingerprint([fingerprint(_load(path)['clean']) for path in calibration.paths])
            saved = load_resume(args)
            validate_checkpoint(saved, identity)
            with phase('range_estimation'):
                ranges = (move_to_device(saved['ranges'], device) if saved else
                          _weight_ranges(layers, args.percentile))
                steps = {name: nn.Parameter((saved['steps'][name].to(device).clone() if saved else
                                            record['weight_ranges'].clone() / (2.0 ** args.init_bits)))
                         for name, record in ranges.items()}
                _clamp(steps, ranges, args.bmax_bits)
                initial_steps = (saved['initial_steps'] if saved else
                                 {name: value.detach().cpu().clone() for name, value in steps.items()})
            cursor = saved['cursor'] if saved else {'unit': 0, 'epoch': 1, 'batch': 0, 'steps': 0,
                                                      'totals': {'likelihood': 0.0, 'prior': 0.0, 'total': 0.0}}
            history = list(saved['history']) if saved else []
            units = ([None] if args.method == 'ppq' else
                     [_Unit(name, layer, {name: layer}) for name, layer in layers.items()])
            optimizer = None

            def persist():
                if not managed:
                    return
                with phase('checkpoint_saving'):
                    save_checkpoint({'schema_version': 1, 'method': args.method, 'identity': identity,
                                     'phase': 'complete' if cursor['unit'] == len(units) else 'optimization',
                                     'cursor': dict(cursor), 'completed_steps': cursor['steps'],
                                     'steps': steps, 'ranges': ranges, 'initial_steps': initial_steps,
                                     'optimizer_state': optimizer.state_dict() if optimizer else None,
                                     'history': history, 'rng_state': rng_state(device)}, checkpoint)
                    atomic_json(history, destination / 'history.json')
            if not saved:
                persist()
            first_resume_unit = True
            while cursor['unit'] < len(units):
                unit_index = cursor['unit']
                unit = units[unit_index]
                local_layers = unit.layers if unit else layers
                local_steps = {name: steps[name] for name in local_layers}
                name = unit.name if unit else 'network'
                with phase('reconstruction_cache'):
                    paths = (_layer_paths(model, unit, calibration.paths, work_dir / f'ppq-unit-{unit_index}',
                                          device, {'training': fingerprint(identity), 'unit': name})
                             if unit else calibration.paths)
                with phase('optimizer_setup'):
                    optimizer = torch.optim.Adam(list(local_steps.values()), lr=args.learning_rate)
                    if saved and first_resume_unit and saved.get('optimizer_state'):
                        optimizer.load_state_dict(saved['optimizer_state'])
                if saved and first_resume_unit:
                    restore_rng(saved['rng_state'], device)
                first_resume_unit = False
                count = args.epochs * len(paths)
                remaining = count - ((cursor['epoch'] - 1) * len(paths) + cursor['batch'])
                batches = progress(range(remaining), desc=f'{args.method} optimize: {name}', unit='update')
                with phase('optimization'):
                    for _ in batches:
                        epoch, batch_index = cursor['epoch'], cursor['batch']
                        lr = get_lr_for_epoch(epoch, args.learning_rate, args.epochs)
                        for group in optimizer.param_groups:
                            group['lr'] = lr
                        record = _load(paths[batch_index])
                        target = record['target' if unit else 'clean'].to(device)
                        if unit is None:
                            batch = move_to_device(record['batch'], device)
                            def forward():
                                with _weight_hooks(local_layers, local_steps, noisy=True):
                                    return forward_model(model, batch, device)
                        else:
                            inputs = move_to_device(record['args'], device)
                            kwargs = move_to_device(record['kwargs'], device)
                            def forward():
                                with _weight_hooks(local_layers, local_steps, noisy=True):
                                    return _first_tensor(unit.module(*inputs, **kwargs))
                        optimizer.zero_grad(set_to_none=True)
                        likelihood = backward_likelihood(forward, target, args, device,
                                                         layer_count=len(layers) if unit else 1)
                        prior = compute_prior(local_steps, ranges, local_layers, args)
                        prior.backward()
                        if not math.isfinite(likelihood) or not torch.isfinite(prior):
                            raise FloatingPointError('Non-finite probabilistic quantization loss.')
                        if any(step.grad is None or not torch.isfinite(step.grad).all() for step in local_steps.values()):
                            raise FloatingPointError('Missing or non-finite step-size gradients.')
                        optimizer.step()
                        _clamp(local_steps, ranges, args.bmax_bits)
                        totals = cursor['totals']
                        totals['likelihood'] += likelihood
                        totals['prior'] += float(prior.detach())
                        totals['total'] += likelihood + float(prior.detach())
                        cursor['batch'] += 1
                        cursor['steps'] += 1
                        counter('optimization_updates')
                        counter('optimization_samples', calibration.batch_sizes[batch_index])
                        if cursor['batch'] == len(paths):
                            history.append({'unit': name, 'epoch': epoch, 'learning_rate': lr,
                                            'num_batches': len(paths), 'avg_bits': _average_bits(steps, ranges, layers),
                                            **{key: value / len(paths) for key, value in totals.items()}})
                            cursor.update(epoch=epoch + 1, batch=0,
                                          totals={'likelihood': 0.0, 'prior': 0.0, 'total': 0.0})
                            progress_write(f'[{args.method}] {name} epoch {epoch}/{args.epochs} '
                                           f'loss={history[-1]["total"]:.6g} bits={history[-1]["avg_bits"]:.4f}')
                        batches.set_postfix(likelihood=likelihood, prior=float(prior.detach()), lr=lr, refresh=False)
                        if cursor['steps'] % getattr(args, 'save_every', 100) == 0:
                            persist()
                cursor.update(unit=unit_index + 1, epoch=1, batch=0,
                              totals={'likelihood': 0.0, 'prior': 0.0, 'total': 0.0})
                optimizer = None
                persist()
            if saved and saved.get('phase') == 'complete':
                # No optimizer work is needed for a completed checkpoint.
                restore_rng(saved['rng_state'], device)
            avg_bits = _average_bits(steps, ranges, layers)
            with phase('materialization'), torch.no_grad():
                for name, layer in progress(layers.items(), desc='PPQ materialize weights', unit='layer'):
                    scale = steps[name].to(layer.weight.dtype).view(-1, 1)
                    layer.weight.copy_(torch.round(layer.weight / scale) * scale)
            state = {'steps': cpu_tree(steps), 'initial_steps': initial_steps, 'ranges': cpu_tree(ranges),
                     'sensitivity': {}, 'sensitivity_raw': {}, 'sensitivity_normalized': {}, 'history': history}
            metadata = {'method': args.method, 'representation': 'materialized_float_fake_quantized_weights',
                        'weight_only': True, 'activation_bits': 32, 'num_layers': len(layers), 'layer_names': list(layers),
                        'calibration_samples': calibration.samples, 'calibration_batches': len(calibration),
                        'calibration_storage': 'persistent_verified_disk', 'initial_bits': args.init_bits,
                        'target_bits': args.target_bits, 'avg_weight_bits': avg_bits, 'epochs': args.epochs,
                        'completed_steps': cursor['steps'], 'mc_samples': args.mc_samples, 'likelihood': 'mc_mean_mse',
                        'prior_mode': 'weighted_average_bit_cap' if args.method == 'ppq' else 'ppq',
                        'gamma_used': args.method == 'ppq-layerwise',
                        'avg_cap_bits': getattr(args, 'avg_cap_bits', 8.0) if args.method == 'ppq' else None,
                        'sensitivity': None, 'sensitivity_normalization': getattr(args, 'sensitivity_normalization', 'raw'),
                        'sensitivity_reference_bits': args.weight_bits,
                        'sensitivity_reference_quantizer': 'per_tensor_affine_linear_and_conv2d',
                        'sobolev_denormalized': False, 'sobolev_scaling_protocol': 'legacy_dataset_std_broadcast_all_channels',
                        'sobolev_differences': 'forward_difference_without_grid_spacing',
                        'variance_convention': 'sigma0^2 * (1 + alpha * sensitivity)^2',
                        'block_prior_multiplier': 1.0, 'block_fisher_damping': None,
                        'layerwise_likelihood_divisor': len(layers) if args.method == 'ppq-layerwise' else 1,
                        'evaluated_test_data': False, 'resume_supported': True}
            if not managed and not getattr(args, 'keep_work_dir', False):
                shutil.rmtree(work_dir / 'ppq-calibration')
                for directory in work_dir.glob('ppq-unit-*'):
                    shutil.rmtree(directory)
            return QuantizationResult(model, metadata, state)
    finally:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(flags[name])
