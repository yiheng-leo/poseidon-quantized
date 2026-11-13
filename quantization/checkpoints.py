"""Atomic, tensor-only checkpoints shared by the trainable quantizers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from SBPQ.resume import atomic_save, fingerprint, rng_state, restore_rng

RESUMABLE = {'sbpq', 'brecq', 'ppq', 'ppq-layerwise'}


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def atomic_json(value, path):
    import os
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.json-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint_file(directory, method):
    return Path(directory) / (method.split('-')[0] + '_checkpoint.pt')


def resolve_resume(path, method=None):
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return path.parent, path
    if not path.is_dir():
        raise FileNotFoundError(f'Resume path not found: {path}')
    candidates = ([checkpoint_file(path, method)] if method else
                  [path / f'{name}_checkpoint.pt' for name in ('sbpq', 'brecq', 'ppq')])
    candidates += [path / 'sbpq_trainer_state.pt']
    found = [candidate for candidate in candidates if candidate.is_file()]
    if len(found) > 1 and method is None and len({p.stem.split('_')[0] for p in found}) > 1:
        raise ValueError('Resume directory contains checkpoints from different methods')
    if found:
        return path, found[0]
    if (path / 'experiment.json').is_file():
        return path, None  # Interrupted while preparing; verified caches can continue.
    raise FileNotFoundError(f'No optimization checkpoint or experiment manifest in {path}')


def load_resume(args):
    if not getattr(args, 'resume', None):
        return None
    _, path = resolve_resume(args.resume, args.method)
    return torch.load(path, map_location='cpu', weights_only=True) if path else None


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def training_identity(model, calibration_digest, args, *, layers=None, device="cpu", model_digest=None):
    from .experiments import effective_settings
    root = Path(__file__).resolve().parents[1]
    files = ['scOT/model.py', 'quantization/common.py']
    files += (['BRECQ/quantizers.py', 'BRECQ/reconstruction.py', 'BRECQ/poseidon.py', 'BRECQ/resume.py']
              if args.method == 'brecq' else ['PPQ/quantize.py', 'PPQ/loss.py', 'quantization/probabilistic.py'])
    implementation = {name: file_sha256(root / name) for name in files}
    return {'implementation': implementation, 'method': args.method, 'settings': effective_settings(args),
            'model': model_digest or fingerprint(model.state_dict()), 'calibration': calibration_digest,
            'layers': list(layers or []), 'device_type': torch.device(getattr(args, 'DEV', device)).type,
            'torch_version': str(torch.__version__),
            'matmul_precision': torch.get_float32_matmul_precision()}


def validate_checkpoint(saved, identity):
    if saved is None:
        return
    differences = [key for key, value in identity.items() if saved['identity'].get(key) != value]
    if differences:
        raise ValueError(f'Resume inputs/settings differ: {differences}')


def save_checkpoint(value, path):
    from .costs import phase, counter, flush
    with phase('checkpoint_saving'):
        atomic_save(cpu_tree(value), path)
        counter('checkpoint_bytes_written', Path(path).stat().st_size)
        flush()


def forward_cache_identity(model_digest, method, device):
    """Bind cached FP targets to the code/library/precision that produced them."""
    root = Path(__file__).resolve().parents[1]
    files = ['scOT/model.py', 'SBPQ/poseidon/poseidon_utils.py' if method == 'sbpq' else 'quantization/common.py']
    return {'schema': 1, 'model': model_digest, 'method': method,
            'implementation': {name: file_sha256(root / name) for name in files},
            'torch_version': str(torch.__version__), 'device_type': torch.device(device).type,
            'matmul_precision': torch.get_float32_matmul_precision()}
