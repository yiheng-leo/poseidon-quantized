"""BRECQ scale-prefix, completed-block shards and current Adam checkpoint."""
from __future__ import annotations

import copy
from pathlib import Path
import shutil

import torch
from torch import nn

from quantization.cache import PersistentCalibration, record_calibration
from quantization.checkpoints import (cpu_tree, file_sha256, fingerprint, load_resume,
    restore_rng, rng_state, save_checkpoint, training_identity, validate_checkpoint)
from quantization.costs import phase, counter
from quantization.common import forward_model
from quantization.layers import _replace
from quantization.progress import progress
from .quantizers import QuantModule, UniformAffineQuantizer, AdaRoundQuantizer, _initialize_brecq_weight_quantizer
from .reconstruction import _QuantStateModel, _reconstruct_brecq_block


def quantizer_states(modules):
    result = {}
    for name, module in modules.items():
        quantizer = module.weight_quantizer
        if quantizer.delta is None:
            continue
        record = {'delta': cpu_tree(quantizer.delta), 'zero_point': cpu_tree(torch.as_tensor(quantizer.zero_point)),
                  'n_bits': quantizer.n_bits, 'sym': quantizer.sym,
                  'kind': 'adaround' if isinstance(quantizer, AdaRoundQuantizer) else 'uniform'}
        if isinstance(quantizer, AdaRoundQuantizer):
            record.update(alpha=cpu_tree(quantizer.alpha), soft_targets=quantizer.soft_targets)
        result[name] = record
    return result


def restore_quantizers(modules, states, *, trainable=False):
    for name, record in states.items():
        if name not in modules:
            raise ValueError(f'Unknown BRECQ quantizer in checkpoint: {name}')
        module = modules[name]
        device = module.weight.device
        uniform = UniformAffineQuantizer(n_bits=record['n_bits'], symmetric=record['sym'],
                                         channel_wise=True, scale_method='mse')
        uniform.delta = record['delta'].to(device)
        uniform.zero_point = record['zero_point'].to(device)
        uniform.inited = True
        if record['kind'] == 'adaround':
            quantizer = AdaRoundQuantizer(uniform, module.org_weight, 'learned_hard_sigmoid')
            with torch.no_grad():
                quantizer.alpha.copy_(record['alpha'].to(device))
            quantizer.alpha.requires_grad_(trainable)
            quantizer.soft_targets = record['soft_targets']
        else:
            quantizer = uniform
        module.weight_quantizer = quantizer


def run(model, layers, calibration_loader, args, weight_bits, activation_bits, device, work_dir, metadata):
    work_dir = Path(work_dir)
    destination = Path(getattr(args, 'save_path', work_dir))
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / 'brecq_checkpoint.pt'
    with phase('identity_verification'):
        model_digest = fingerprint(model.state_dict())
    with phase('calibration_cache'):
        calibration = PersistentCalibration(calibration_loader, work_dir / 'brecq-calibration',
                                             {'model': model_digest, 'method': 'brecq'})
    record_calibration(metadata, calibration)
    identity = training_identity(model, calibration.digest, args, device=device, model_digest=model_digest)
    saved = load_resume(args)
    validate_checkpoint(saved, identity)
    with phase('model_wrapping'):
        original = copy.deepcopy(model)
        weight_parameters = {'n_bits': weight_bits, 'channel_wise': True, 'scale_method': 'mse'}
        activation_parameters = {'n_bits': 8, 'channel_wise': False, 'scale_method': 'max'}
        poseidon = model.__class__.__module__ == 'scOT.model'
        if poseidon:
            from .poseidon import PoseidonQuantModel, QuantScOTLayer, QuantConvNeXtBlock, QuantResNetBlock
            wrapped = PoseidonQuantModel(model, weight_parameters, activation_parameters).to(device).eval()
            qnn = _QuantStateModel(wrapped.model).to(device).eval()
            block_types = (QuantScOTLayer, QuantConvNeXtBlock, QuantResNetBlock)
            blocks = [(name, module) for name, module in qnn.model.named_modules()
                      if isinstance(module, block_types)]
        else:
            for name, layer in layers.items():
                model = _replace(model, name, QuantModule(layer, weight_parameters, activation_parameters))
            qnn = _QuantStateModel(model).to(device).eval()
            blocks = []
        modules = {name: module for name, module in qnn.model.named_modules() if isinstance(module, QuantModule)}
        if not modules:
            raise ValueError('BRECQ found no quantizable operations')
        covered = {id(module) for _, block in blocks for module in block.modules() if isinstance(module, QuantModule)}
        blocks.extend((name, module) for name, module in modules.items() if id(module) not in covered)
        plan = [name for name, _ in blocks]
        if saved and (saved['block_plan'] != plan or saved['layer_order'] != list(modules)):
            raise ValueError('BRECQ block/layer order changed')
    state = saved or {'schema_version': 1, 'method': 'brecq', 'identity': identity,
                      'phase': 'initialization', 'block_plan': plan, 'layer_order': list(modules),
                      'scales': {}, 'initialized_layers': 0, 'completed_blocks': [],
                      'unit_index': 0, 'iteration': 0, 'completed_steps': 0,
                      'current_quantizers': {}, 'optimizer_state': None, 'loss_count': 0}
    state['identity'] = identity
    def persist():
        state['rng_state'] = rng_state(torch.device(device))
        save_checkpoint(state, checkpoint_path)
    if not saved:
        persist()
    with phase('scale_initialization'):
        restore_quantizers(modules, state['scales'])
        if saved and state['phase'] == 'initialization':
            restore_rng(state['rng_state'], torch.device(device))
        for index, (name, module) in enumerate(progress(modules.items(), desc='BRECQ initialize weight scales', unit='layer')):
            if index < state['initialized_layers']:
                continue
            with torch.no_grad():
                _initialize_brecq_weight_quantizer(module)
            state['scales'].update(quantizer_states({name: module}))
            state['initialized_layers'] = index + 1
            counter('weight_scales_initialized')
            # Save every completed layer: an hour-long MSE search is resumable.
            persist()
    if state['phase'] == 'initialization':
        state['phase'] = 'reconstruction'
        persist()
    with phase('completed_block_restore'):
        for record in state['completed_blocks']:
            path = destination / record['file']
            if not path.is_file() or file_sha256(path) != record['sha256']:
                raise ValueError(f'BRECQ completed-block checkpoint missing/corrupt: {path}')
            shard = torch.load(path, map_location='cpu', weights_only=True)
            restore_quantizers(modules, shard['quantizers'])
    qnn.requires_grad_(False)
    if saved and state['phase'] == 'complete':
        restore_rng(state['rng_state'], torch.device(device))
    cached_batches = 0
    iterations = int(args.reconstruction_iterations)
    batch_size = int(args.rec_batchsize)
    shards = destination / 'checkpoints' / 'brecq-completed'
    shards.mkdir(parents=True, exist_ok=True)
    for unit_index in progress(range(state['unit_index'], len(blocks)), desc='BRECQ reconstruct blocks', unit='block'):
        name, block = blocks[unit_index]
        owned_ids = {id(module) for module in block.modules() if isinstance(module, QuantModule)}
        local_modules = {key: module for key, module in modules.items() if id(module) in owned_ids}
        io_directory = work_dir / f'brecq-block-{unit_index:04d}'
        io_directory.mkdir(parents=True, exist_ok=True)
        current = state
        def checkpoint_update(iteration, optimizer, loss_count):
            state.update(phase='reconstruction', unit_index=unit_index, iteration=iteration,
                         completed_steps=unit_index * iterations + iteration)
            if iteration % getattr(args, 'save_every', 100) == 0 or iteration == iterations:
                with phase('checkpoint_saving'):
                    state['current_quantizers'] = quantizer_states(local_modules)
                    state['optimizer_state'] = optimizer.state_dict()
                    state['loss_count'] = loss_count
                    persist()
        qnn.eval()
        cached_batches = _reconstruct_brecq_block(
            qnn, block, calibration, iterations, batch_size, io_directory, device,
            resume_state=current, checkpoint_update=checkpoint_update,
            restore_current=lambda values: restore_quantizers(local_modules, values, trainable=True),
            cache_identity={'training': fingerprint(identity), 'unit': name,
                            'prefix': [item['sha256'] for item in state['completed_blocks']]},
            learning_rate=getattr(args, 'reconstruction_lr', 1e-3),
        )
        with phase('checkpoint_saving'):
            path = shards / f'block-{unit_index:04d}.pt'
            save_checkpoint({'name': name, 'quantizers': quantizer_states(local_modules)}, path)
            state['completed_blocks'].append({'name': name, 'file': str(path.relative_to(destination)),
                                              'sha256': file_sha256(path)})
            for module in local_modules.values():
                module.weight_quantizer.alpha.grad = None
                module.weight_quantizer.alpha.requires_grad_(False)
            state.update(unit_index=unit_index + 1, iteration=0, current_quantizers={},
                         optimizer_state=None, loss_count=0, completed_steps=(unit_index + 1) * iterations)
            persist()
        shutil.rmtree(io_directory)
    state['phase'] = 'complete'
    persist()
    exported_state = {}
    original_modules = dict(original.named_modules())
    with phase('materialization'), torch.no_grad():
        for name, module in progress(modules.items(), desc='BRECQ materialize weights', unit='layer'):
            quantizer = module.weight_quantizer
            if hasattr(quantizer, 'soft_targets'):
                quantizer.soft_targets = False
            original_modules[name].weight.copy_(quantizer(module.weight).detach())
            exported_state[name] = {'scale': cpu_tree(quantizer.delta), 'zero_point': cpu_tree(quantizer.zero_point),
                                    'rounding_alpha': cpu_tree(getattr(quantizer, 'alpha', None))}
        qnn.set_quant_state(True, False)
        example = next(iter(calibration))
        expected = forward_model(qnn, example, device)
        actual = forward_model(original.eval(), example, device)
        difference = (expected - actual).abs().amax()
        if not torch.allclose(expected, actual, atol=1e-6, rtol=1e-5):
            raise RuntimeError(f'BRECQ materialization changed predictions (max difference {float(difference):.3g})')
    details = {'reconstruction_blocks': plan, 'reconstruction_iterations': iterations,
               'reconstruction_batch_size': batch_size, 'reconstruction_cached_batches': cached_batches,
               'reconstruction_learning_rate': getattr(args, 'reconstruction_lr', 1e-3),
               'completed_steps': state['completed_steps'], 'resume_supported': True,
               'calibration_cache': 'persistent content-verified disk caches and one current block',
               'reconstruction_sampling': 'uniform sample index mapped to its cached batch (batch-size-proportional sampling)',
               'materialization_max_abs_difference': float(difference), 'quantized_layers': list(modules),
               'scope': 'Poseidon Conv2d and Linear, including conditional norm projections' if poseidon else 'selected Linear'}
    return original.eval(), exported_state, details
