"""Network-wise SBPQ with complete-trajectory calibration and portable output.

The Beta construction, output sum-squared MC likelihood and optimizer are the
same as the author's runner. This adapter changes the calibration protocol,
cache identity and artifact lifecycle; it never constructs validation/test data.
"""
from __future__ import annotations

from collections import OrderedDict
import json
import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from quantization.cache import CalibrationCache
from quantization.common import QuantizationResult, select_quantizable_layers
from quantization.progress import progress, progress_write
from SBPQ.poseidon.beta_parameter_builder import (
    build_poseidon_beta_parameters, save_poseidon_beta_parameters,
)
from SBPQ.poseidon.blocks import build_poseidon_block_mapping, compute_block_parameter_counts
from SBPQ.poseidon.evaluation import build_channel_parameter_weights
from SBPQ.poseidon.poseidon_utils import poseidon_forward
from SBPQ.poseidon.ranges import compute_weight_ranges_poseidon
from SBPQ.poseidon.sensitivity import compute_poseidon_block_sensitivity
from SBPQ.poseidon.sobolev import compute_poseidon_sobolev_loss, get_order_weights
from SBPQ.resume import atomic_save, fingerprint, resolve_checkpoint
from SBPQ.storage import DiskSequence
from SBPQ.trainer import SBPQTrainer, freeze_model_parameters


def validate_settings(args):
    if args.total_steps < 1 or args.save_every < 1:
        raise ValueError("SBPQ total_steps/save_every must be positive")
    if not 0 < args.minimum_bits < args.maximum_bits <= 30:
        raise ValueError("SBPQ requires 0 < minimum_bits < maximum_bits <= 30")
    if not args.minimum_bits <= args.init_bits <= args.maximum_bits:
        raise ValueError("SBPQ init_bits must be inside minimum_bits..maximum_bits")
    if not args.minimum_bits < args.target_bits < args.maximum_bits:
        raise ValueError("SBPQ target_bits must lie strictly inside the bit bounds")
    if args.beta_kappa <= 0 or args.delta_bits < 0 or args.likelihood_scale < 0:
        raise ValueError("SBPQ requires positive beta_kappa and nonnegative delta_bits/likelihood_scale")
    if not 0 < args.beta_epsilon < 0.5 or args.beta_relative_epsilon <= 0:
        raise ValueError("Invalid SBPQ Beta epsilon")
    if args.task_loss_weight < 0 or args.sobolev_loss_weight < 0:
        raise ValueError("SBPQ sensitivity loss weights must be nonnegative")
    if args.task_loss_weight + args.sobolev_loss_weight == 0 or not 0 <= args.sobolev_order <= 8:
        raise ValueError("SBPQ needs a nonzero sensitivity loss and nonnegative Sobolev order")
    weights = args.sobolev_order_weights
    if isinstance(weights, str):
        values = tuple(float(value.strip()) for value in weights.split(','))
        weights = values[0] if len(values) == 1 else values
    if isinstance(weights, (list, tuple)):
        weights = tuple(float(value) for value in weights)
        if not all(math.isfinite(value) for value in weights):
            raise ValueError("Sobolev weights must be finite")
    elif isinstance(weights, bool) or not math.isfinite(float(weights)):
        raise ValueError("Sobolev weights must be finite numbers")
    else:
        weights = float(weights)
    resolved = get_order_weights(args.sobolev_order, weights)
    if any(value < 0 for value in resolved):
        raise ValueError('Sobolev weights must be nonnegative')
    args.sobolev_order_weights = weights


def sensitivity_loss(args, dataset):
    """Exactly the author's supervised task + physical Sobolev sensitivity loss."""
    def loss(prediction, reference):
        sobolev = compute_poseidon_sobolev_loss(
            prediction=prediction, reference=reference, dataset=dataset,
            dataset_name=args.dataset, max_order=args.sobolev_order,
            order_weights=args.sobolev_order_weights, norm=args.sobolev_norm,
            transpose_spatial_axes=args.sobolev_transpose,
        )
        return args.task_loss_weight * F.mse_loss(prediction, reference) + args.sobolev_loss_weight * sobolev
    return loss


def prepare(model, batches, dataset, args, device, directory):
    """Reuse preparation only for the same weights, batch bytes, layers and loss."""
    blocks, block_to_layers, layer_to_block, _ = build_poseidon_block_mapping(model)
    layers = OrderedDict((name, layer) for name, layer in select_quantizable_layers(model).items()
                         if name in layer_to_block)
    if not layers:
        raise ValueError("SBPQ requires eligible Linear layers inside Poseidon structural blocks")
    active_blocks = [name for name in blocks if any(layer in layers for layer in block_to_layers[name])]
    mapping = {name: layer_to_block[name] for name in layers}
    progress_write(f"[INFO] Selected candidate Linear layers: {len(layers)} in {len(active_blocks)} blocks.")
    constants = getattr(dataset, 'constants', {})
    root = Path(__file__).resolve().parents[1]
    implementations = ['SBPQ/quantize.py', 'SBPQ/poseidon/sobolev.py',
                       'SBPQ/poseidon/sensitivity.py', 'SBPQ/poseidon/ranges.py',
                       'SBPQ/poseidon/blocks.py', 'scOT/model.py']
    identity = {
        'version': 1, 'implementation': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                          for name in implementations}, 'model': fingerprint(model.state_dict()), 'calibration': fingerprint(batches),
        'layer_order': list(layers), 'layer_to_block': mapping, 'constants': fingerprint(constants),
        'transpose': bool(getattr(dataset, 'transpose', False)), 'dataset': args.dataset,
        'percentile': args.percentile, 'sobolev_order': args.sobolev_order,
        'sobolev_order_weights': args.sobolev_order_weights, 'sobolev_norm': args.sobolev_norm,
        'sobolev_transpose': args.sobolev_transpose, 'task_loss_weight': args.task_loss_weight,
        'sobolev_loss_weight': args.sobolev_loss_weight,
    }
    path = directory / 'sbpq_preparation.pt'
    saved = torch.load(path, map_location='cpu', weights_only=True) if path.exists() else None
    if saved is not None and saved['identity'] != identity:
        if args.resume or not args.overwrite:
            raise ValueError('SBPQ preparation does not match this model/calibration/loss; use a new save_path')
        saved = None
    if saved is None:
        ranges = compute_weight_ranges_poseidon(model, list(layers), device,
                                                percentile_prob=args.percentile, range_method='percentile')
        # Sensitivity needs gradients through the unmodified FP network. Training
        # immediately clears these buffers and freezes all original weights.
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        try:
            raw = compute_poseidon_block_sensitivity(
                model, batches, device, sensitivity_loss(args, dataset),
                max_batches=None, block_names=active_blocks,
            )
        finally:
            freeze_model_parameters(model)
        saved = {'identity': identity, 'ranges': {name: {'weight_ranges': value.detach().cpu()}
                                                 for name, value in ranges.items()},
                 'raw_sensitivity': {name: value.detach().cpu() for name, value in raw.items()}}
        atomic_save(saved, path)
    else:
        progress_write('[INFO] Reusing SBPQ preparation verified against model and calibration contents.')
    counts = compute_block_parameter_counts(model, block_to_layers, layer_names=list(layers))
    counts = {name: counts[name] for name in active_blocks}
    beta = build_poseidon_beta_parameters(
        saved['raw_sensitivity'], args.minimum_bits, args.maximum_bits,
        args.target_bits, args.delta_bits, args.beta_kappa, block_parameter_counts=counts,
        mean_epsilon=args.beta_epsilon, relative_epsilon=args.beta_relative_epsilon,
    )
    beta_path = directory / 'sbpq_beta_parameters.pt'
    # Beta is rebuilt deterministically from the verified sensitivity; legacy
    # basename-only caches are deliberately not consulted.
    save_poseidon_beta_parameters(beta, beta_path, metadata={'preparation_identity': fingerprint(identity)})
    return layers, mapping, saved, beta_path


def quantize(model, calibration_loader, args, device, work_dir):
    validate_settings(args)
    device = torch.device(device)
    model.to(device).eval()
    torch.set_float32_matmul_precision('high')
    destination = Path(args.save_path)
    checkpoint = destination / 'sbpq_checkpoint.pt'
    if checkpoint.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f'SBPQ checkpoint already exists: {checkpoint}; use --resume or a new save_path')
    work_dir = Path(work_dir)
    snapshot = CalibrationCache(calibration_loader, work_dir)
    batches = DiskSequence(snapshot.paths)
    layers, mapping, prepared, beta_path = prepare(
        model, batches, calibration_loader.dataset, args, device, destination,
    )
    freeze_model_parameters(model)
    outputs = []
    with torch.no_grad():
        for index, batch in enumerate(progress(batches, desc='Cache SBPQ FP outputs')):
            path = work_dir / f'clean-{index:06d}.pt'
            torch.save(poseidon_forward(model, batch, device).detach().cpu(), path)
            outputs.append(path)
    trainer = SBPQTrainer(
        model, batches, DiskSequence(outputs), prepared['ranges'], mapping, beta_path,
        initial_bits=args.init_bits, minimum_bits=args.minimum_bits, maximum_bits=args.maximum_bits,
        learning_rate=args.learning_rate, num_mc_samples=args.mc_samples, eta=args.eta,
        prior_scale=args.prior_scale, likelihood_scale=args.likelihood_scale,
        beta_boundary_epsilon=args.beta_epsilon, weight_decay=0.0, gradient_clip_norm=None,
        channel_weights=build_channel_parameter_weights(model, list(layers)), device=device,
    )
    resume_metadata = {'protocol': 'ritchie-complete-trajectory-v1',
                       'preparation_identity': fingerprint(prepared['identity'])}
    if args.resume:
        trainer.load(resolve_checkpoint(args.resume), expected_metadata=resume_metadata)
    if args.total_steps < len(batches):
        progress_write(f'[INFO] {len(batches)} calibration batches are available; {args.total_steps} updates '
                       'use a prefix of them. Sensitivity uses the full selection. Increase --total_steps '
                       'for one or more complete optimization passes.')
    trainer.optimize(args.total_steps, checkpoint_path=checkpoint, save_every=args.save_every,
                     checkpoint_metadata={**resume_metadata, 'target_total_steps': args.total_steps})
    trainer.save(destination / 'sbpq_trainer_state.pt', metadata=resume_metadata)
    (destination / 'history.json').write_text(json.dumps(trainer.history, indent=2) + '\n')
    steps = {name: value.detach().cpu().clone() for name, value in trainer.get_step_sizes().items()}
    average = trainer.calculate_average_bitwidth(parameter_weighted=True)
    # Round once at export, matching the original evaluation hook; no clipping
    # or integer bit rounding is introduced. Inference needs no hooks/MC loops.
    with torch.no_grad():
        for name, layer in progress(layers.items(), desc='SBPQ materialize weights', unit='layer'):
            step = steps[name].to(device=layer.weight.device, dtype=layer.weight.dtype).reshape(-1, 1).clamp_min(1e-12)
            layer.weight.copy_(torch.round(layer.weight / step) * step)
    details = {
        'method': 'sbpq', 'representation': 'materialized_float_fake_quantized_weights',
        'weight_only': True, 'activation_bits': 32, 'packed_integer_weights': False,
        'num_layers': len(layers), 'layer_names': list(layers), 'scope': 'selected Linear inside Poseidon blocks',
        'calibration_samples': snapshot.samples, 'calibration_batches': len(batches),
        'calibration_storage': 'temporary_disk', 'calibration_truncated': False,
        'sensitivity_batches': len(batches), 'completed_steps': len(trainer.history),
        'optimization_unique_batches': min(len(trainer.history), len(batches)),
        'optimization_batch_schedule': 'cyclic_source_order',
        'avg_weight_bits': average, 'initial_bits': args.init_bits, 'target_bits': args.target_bits,
        'minimum_bits': args.minimum_bits, 'maximum_bits': args.maximum_bits,
        'beta_kappa': args.beta_kappa, 'delta_bits': args.delta_bits,
        'likelihood': 'mc_log_mean_exp_output_sum_squared', 'prior': 'parameter_weighted_centered_block_beta',
        'weight_range': 'empirical_percentile', 'percentile': args.percentile,
        'sobolev_order': args.sobolev_order, 'evaluated_test_data': False,
    }
    state = {'steps': steps, 'ranges': prepared['ranges'], 'sensitivity_raw': prepared['raw_sensitivity'],
             'history': list(trainer.history), 'training_signature': trainer.training_signature}
    return QuantizationResult(model, details, state)
