"""Stable experiment names and immutable effective configurations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .checkpoints import RESUMABLE, atomic_json, resolve_resume

# Operational switches may change on resume; the numerical experiment may not.
_OPERATIONAL = {'config', 'save_path', 'output_root', 'resume', 'overwrite', 'keep_work_dir',
                'save_every', 'profile_interval'}


def effective_settings(args):
    values = {key: value for key, value in vars(args).items()
              if key not in _OPERATIONAL and not key.startswith('_')}
    values.pop('variant', None)  # method already resolves the variant uniquely.
    for key in ('model_path', 'data_path'):
        if values.get(key):
            values[key] = str(Path(values[key]).expanduser().resolve())
    return json.loads(json.dumps(values, sort_keys=True, allow_nan=False))


def config_digest(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def token(value):
    if isinstance(value, float):
        value = format(value, '.8g')
    if isinstance(value, (list, tuple)):
        value = '-'.join(token(item) for item in value)
    return re.sub(r'[^a-zA-Z0-9.-]+', '-', str(value)).replace('.', 'p').replace('-', 'm')


def run_name(args):
    common = (f'traj{args.traj_num}_{"rnd" if args.random_calibration == "y" else "fix"}'
              f'_cs{args.calibration_seed}_ft{args.final_time}_tb{args.time_back}'
              f'_bs{args.batch_size}_s{args.seed}')
    if args.method == 'sbpq':
        parameters = (f'i{args.init_bits}_t{token(args.target_bits)}_k{token(args.beta_kappa)}'
                      f'_d{token(args.delta_bits)}_mc{args.mc_samples}_eta{token(args.eta)}'
                      f'_lr{token(args.learning_rate)}_ps{token(args.prior_scale)}'
                      f'_so{args.sobolev_order}_sw{token(args.sobolev_order_weights)}'
                      f'_steps{args.total_steps}')
    elif args.method == 'brecq':
        parameters = (f'w{args.weight_bits}a{args.activation_bits}_recbs{args.rec_batchsize}'
                      f'_iter{args.reconstruction_iterations}_lr{token(args.reconstruction_lr)}')
    else:
        parameters = (f'i{args.init_bits}_t{token(args.target_bits)}_ep{args.epochs}'
                      f'_mc{args.mc_samples}_eta{token(args.eta)}_lr{token(args.learning_rate)}')
    readable = f'{args.method}_{parameters}_{common}'
    return readable[:200] + '_cfg-' + config_digest(effective_settings(args))[:20]


def resolve_output(args):
    if args.method not in RESUMABLE:
        return
    if args.resume:
        directory, _ = resolve_resume(args.resume, args.method)
        args.save_path = str(directory)
    else:
        if not hasattr(args, '_requested_output_root'):
            args._requested_output_root = (args.output_root or args.save_path or
                                          str(Path('quantization_artifacts') / args.model_name / args.dataset))
        args.save_path = str(Path(args._requested_output_root) / run_name(args))
    args._experiment_settings = effective_settings(args)
    args._experiment_id = config_digest(args._experiment_settings)


def register_experiment(args):
    directory = Path(args.save_path)
    path = directory / 'experiment.json'
    settings = effective_settings(args)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['settings'] != settings:
            changed = sorted(key for key in saved['settings'].keys() | settings.keys()
                             if saved['settings'].get(key) != settings.get(key))
            raise ValueError(f'Experiment hyperparameters changed: {changed}; use a new run directory')
        if not args.resume and not args.overwrite:
            raise FileExistsError(f'Experiment exists: {directory}; use --resume or --overwrite')
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone
    saved = {'schema_version': 2, 'method': args.method, 'run_name': directory.name,
             'config_sha256': config_digest(settings), 'settings': settings,
             'created_at_utc': datetime.now(timezone.utc).isoformat(),
             'legacy_resume': bool(args.resume)}
    atomic_json(saved, path)
    return saved


def resume_defaults(path, method=None):
    directory, _ = resolve_resume(path, method)
    manifest = directory / 'experiment.json'
    if manifest.is_file():
        return json.loads(manifest.read_text())['settings']
    legacy = directory / 'metadata.json'
    if legacy.is_file():
        return json.loads(legacy.read_text()).get('settings', {})
    return {}
