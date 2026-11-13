"""BRECQ block reconstruction with complete calibration data and bounded disk caches."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import torch
from torch import nn
from .quantizers import QuantModule, BaseQuantBlock, AdaRoundQuantizer, lp_loss
from quantization.common import forward_model
from quantization.progress import progress, progress_write

class StopForwardException(Exception):
    """
    Used to stop the full forward once the target block has been reached.
    """
    pass


def move_to_device(obj: Any, device: torch.device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    else:
        return obj


def detach_to_cpu(obj: Any):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    elif isinstance(obj, dict):
        return {k: detach_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_to_cpu(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(detach_to_cpu(v) for v in obj)
    else:
        return obj


class DataSaverHook:
    """
    Forward hook that stores block input args and block output.
    For Poseidon ScOTLayer, the hook input is a tuple like:
        (hidden_states, input_dimensions, time, head_mask, output_attentions, always_partition)
    """
    def __init__(self, store_input: bool = False, store_output: bool = False, stop_forward: bool = False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch

        if self.store_output:
            if isinstance(output_batch, tuple):
                self.output_store = output_batch[0]
            else:
                self.output_store = output_batch

        if self.stop_forward:
            raise StopForwardException


class PoseidonGetLayerInpOut:
    """
    Poseidon-specific version of GetLayerInpOut.

    Main differences from original BRECQ:
    - model input is a batch dict, not a single tensor
    - block input is a tuple of multiple arguments, not a single tensor
    """
    def __init__(
        self,
        model,
        layer: Union[QuantModule, BaseQuantBlock],
        device: torch.device,
        asym: bool = False,
        act_quant: bool = False,
    ):
        self.model = model
        self.layer = layer
        self.asym = asym
        self.device = device
        self.act_quant = act_quant
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=True)

    def __call__(self, model_input: Dict[str, Any]):
        self.model.eval()
        self.model.set_quant_state(False, False)

        batch_on_device = move_to_device(model_input, self.device)
        handle = self.layer.register_forward_hook(self.data_saver)

        with torch.no_grad():
            try:
                _ = self.model(**batch_on_device)
            except StopForwardException:
                pass

            if self.asym:
                # recompute block input using quantized prefix, but keep FP target output
                self.data_saver.store_output = False
                self.model.set_quant_state(weight_quant=True, act_quant=self.act_quant)
                try:
                    _ = self.model(**batch_on_device)
                except StopForwardException:
                    pass
                self.data_saver.store_output = True

        handle.remove()

        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train()

        block_args = detach_to_cpu(self.data_saver.input_store)
        block_out = detach_to_cpu(self.data_saver.output_store)
        return block_args, block_out


def move_block_args_to_device(block_args: Tuple[Any, ...], device: torch.device):
    """
    Move cached block input tuple back to device before reconstruction forward.
    """
    return tuple(move_to_device(x, device) for x in block_args)


def get_reconstruction_output(block_output):
    """
    Normalize block forward output to the main tensor output.
    """
    if isinstance(block_output, tuple):
        return block_output[0]
    return block_output

class PoseidonLossFunction:
    def __init__(
        self,
        block,
        round_loss: str = "relaxation",
        weight: float = 1.0,
        rec_loss: str = "mse",
        max_count: int = 2000,
        b_range: tuple = (10, 2),
        decay_start: float = 0.0,
        warmup: float = 0.0,
        p: float = 2.0,
    ):
        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p

        self.temp_decay = LinearTempDecay(
            max_count,
            rel_start_decay=warmup + (1 - warmup) * decay_start,
            start_b=b_range[0],
            end_b=b_range[1],
        )
        self.count = 0

    def __call__(self, pred, tgt, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy.
        """
        self.count += 1

        if self.rec_loss == "mse":
            rec_loss = lp_loss(pred, tgt, p=self.p)

        elif self.rec_loss == "fisher_diag":
            if grad is None:
                raise ValueError("grad must not be None when rec_loss='fisher_diag'")
            # Matches original BRECQ logic, but generalized to arbitrary tensor dims.
            reduce_dims = tuple(range(1, pred.dim()))
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(reduce_dims).mean()

        elif self.rec_loss == "fisher_full":
            if grad is None:
                raise ValueError("grad must not be None when rec_loss='fisher_full'")
            # Matches original BRECQ logic, but generalized to arbitrary tensor dims.
            a = (pred - tgt).abs()
            grad = grad.abs()
            reduce_dims = tuple(range(1, pred.dim()))
            view_shape = (-1,) + (1,) * (pred.dim() - 1)
            batch_dotprod = torch.sum(a * grad, dim=reduce_dims).view(*view_shape)
            rec_loss = (batch_dotprod * a * grad).mean() / 100

        else:
            raise ValueError(f"Not supported reconstruction loss function: {self.rec_loss}")

        b = self.temp_decay(self.count)

        if self.count < self.loss_start or self.round_loss == "none":
            b = 0
            round_loss = 0

        elif self.round_loss == "relaxation":
            round_loss = 0
            for _, module in self.block.named_modules():
                if isinstance(module, QuantModule):
                    round_vals = module.weight_quantizer.get_soft_targets()
                    round_loss += self.weight * (
                        1 - ((round_vals - 0.5).abs() * 2).pow(b)
                    ).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss

        if self.count % 500 == 0:
            progress_write(
                "Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}".format(
                    float(total_loss),
                    float(rec_loss),
                    float(round_loss),
                    b,
                    self.count,
                )
            )

        return total_loss


class LinearTempDecay:
    def __init__(
        self,
        t_max: int,
        rel_start_decay: float = 0.2,
        start_b: int = 10,
        end_b: int = 2,
    ):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Temperature scheduler for rounding relaxation.
        """
        if t < self.start_decay:
            return self.start_b
        rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
        return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))

class _QuantStateModel(nn.Module):
    """Transient BRECQ adapter, removed before exporting the model."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def set_quant_state(self, weight_quant=False, act_quant=False):
        for module in self.model.modules():
            if hasattr(module, "set_quant_state"):
                module.set_quant_state(weight_quant, act_quant)

    def train(self, mode=True):
        # The legacy IO-capture helper calls train() after calibration. PTQ
        # reconstructs inference behavior; keep dropout/BN in eval throughout.
        return super().train(False)

    def forward(self, **batch):
        return forward_model(self.model, batch, next(self.parameters()).device)


def _brecq_calibration_batches(calibration, batch_size):
    """Rebatch every calibration sample to the requested reconstruction size."""
    from torch.utils.data import default_collate
    if batch_size < 1:
        raise ValueError("BRECQ rec_batchsize must be positive")
    pending = []
    for batch in calibration:
        if not isinstance(batch, dict):
            raise ValueError("BRECQ calibration requires dictionary batches with pixel_values")
        size = batch["pixel_values"].shape[0]
        # The model adapter only needs inputs/conditioning. Reconstruction
        # targets are captured FP block outputs, never supervised test labels.
        for index in range(size):
            pending.append({key: value[index] for key, value in batch.items()
                            if key in ("pixel_values", "time") and torch.is_tensor(value)})
            if len(pending) == batch_size:
                yield default_collate(pending)
                pending = []
    if pending:
        yield default_collate(pending)


def _reconstruct_brecq_block(qnn, block, calibration, iterations, batch_size, directory, device,
                             *, resume_state=None, checkpoint_update=None, restore_current=None,
                             cache_identity=None, learning_rate=1e-3):
    """Original AdaRound updates with verified IO reuse and exact iteration resume."""
    import json
    from quantization.checkpoints import atomic_json, atomic_save, fingerprint, restore_rng
    from quantization.costs import phase, counter
    directory = Path(directory)
    with phase('optimizer_setup'):
        qnn.set_quant_state(False, False)
        block.set_quant_state(True, False)
        for module in block.modules():
            if isinstance(module, QuantModule):
                module.weight_quantizer = AdaRoundQuantizer(
                    uaq=module.weight_quantizer, round_mode='learned_hard_sigmoid',
                    weight_tensor=module.org_weight.detach())
                module.weight_quantizer.soft_targets = True
        if resume_state and resume_state.get('current_quantizers'):
            restore_current(resume_state['current_quantizers'])
        parameters = [module.weight_quantizer.alpha for module in block.modules()
                      if isinstance(module, QuantModule)]
        if not parameters:
            raise ValueError('BRECQ reconstruction block contains no weight quantizers')
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        if resume_state and resume_state.get('optimizer_state'):
            optimizer.load_state_dict(resume_state['optimizer_state'])
        loss_function = PoseidonLossFunction(block, round_loss='relaxation', weight=0.01,
                                            max_count=iterations, rec_loss='mse',
                                            b_range=(20, 2), warmup=0.2, p=2.)
        start = int(resume_state.get('iteration', 0)) if resume_state else 0
        loss_function.count = int(resume_state.get('loss_count', start)) if resume_state else 0
        if not 0 <= start <= iterations or loss_function.count != start:
            raise ValueError('BRECQ saved iteration/loss-schedule count is invalid')
    get_io = PoseidonGetLayerInpOut(qnn, block, torch.device(device), asym=True, act_quant=False)
    total_samples, count = 0, 0
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest and manifest['identity'] != cache_identity:
        raise ValueError('BRECQ block IO cache identity changed')
    entries = manifest['entries'] if manifest else []
    with phase('reconstruction_cache'):
        cache_total = ((calibration.samples + batch_size - 1) // batch_size
                       if hasattr(calibration, 'samples') else None)
        batches = progress(_brecq_calibration_batches(calibration, batch_size),
                           total=cache_total, desc='BRECQ cache block IO', leave=False)
        for count, batch in enumerate(batches, start=1):
            total_samples += int(batch['pixel_values'].shape[0])
            path = directory / f'block-{count-1:06d}.pt'
            input_digest = fingerprint(batch)
            if count <= len(entries) and path.exists():
                record = torch.load(path, map_location='cpu', weights_only=True)
                if entries[count-1]['input'] != input_digest or entries[count-1]['payload'] != fingerprint(record):
                    raise ValueError(f'BRECQ block IO cache changed/corrupt: {path}')
                counter('block_io_records_reused')
                del record
            else:
                inputs, target = get_io(batch)
                if inputs is None or target is None:
                    raise ValueError('Calibration did not execute a BRECQ reconstruction block')
                record = {'inputs': inputs, 'target': target}
                digest = fingerprint(record)
                if count <= len(entries):
                    if entries[count-1] != {'input': input_digest, 'payload': digest}:
                        raise ValueError('Rebuilt BRECQ IO target differs from checkpoint cache')
                else:
                    entries.append({'input': input_digest, 'payload': digest})
                atomic_save(record, path)
                counter('cache_bytes_written', path.stat().st_size)
                counter('block_io_records_written')
                if count % 32 == 0:
                    atomic_json({'identity': cache_identity, 'entries': entries, 'complete': False}, manifest_path)
                get_io.data_saver.input_store = None
                get_io.data_saver.output_store = None
        if not count or count < len(entries):
            raise ValueError('BRECQ calibration is empty or its batch count changed')
        atomic_json({'identity': cache_identity, 'entries': entries, 'complete': True}, manifest_path)
    if resume_state and 'rng_state' in resume_state:
        restore_rng(resume_state['rng_state'], torch.device(device))
    reconstruction = progress(range(start, iterations), desc='BRECQ AdaRound reconstruction',
                              unit='iteration', leave=False)
    with phase('optimization'):
        for iteration in reconstruction:
            index = int(torch.randint(total_samples, (1,))) // batch_size
            record = torch.load(directory / f'block-{index:06d}.pt', map_location='cpu', weights_only=False)
            inputs = move_block_args_to_device(record['inputs'], device)
            target = record['target'].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = get_reconstruction_output(block(*inputs))
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            counter('optimization_updates')
            physical_batch = min(batch_size, total_samples - index * batch_size)
            counter('optimization_samples', physical_batch)
            if checkpoint_update:
                checkpoint_update(iteration + 1, optimizer, loss_function.count)
            if iteration == 0 or (iteration + 1) % 100 == 0 or iteration + 1 == iterations:
                reconstruction.set_postfix(loss=float(loss.detach()), refresh=False)
    for module in block.modules():
        if isinstance(module, QuantModule):
            module.weight_quantizer.soft_targets = False
    return count
