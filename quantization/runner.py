"""Load calibration data, quantize, and save; evaluation is a separate command."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import shlex
import subprocess
import tempfile
import time


METHODS = (
    "fp", "uniform", "dynamic", "static", "sbpq", "ppq", "ppq-layerwise",
    "sapq", "sapq-block", "sapq-layerwise", "sapq-network", "brecq", "repq", "ptq4vit",
)
PROBABILISTIC_METHODS = {"sbpq", "ppq", "ppq-layerwise", "sapq", "sapq-block", "sapq-layerwise", "sapq-network"}
METHOD_MODULES = {
    "sbpq": "SBPQ.quantize",
    "uniform": "UniformQ.quantize", "dynamic": "DynamicQ.quantize", "static": "StaticQ.quantize",
    "ppq": "PPQ.quantize", "ppq-layerwise": "PPQ.quantize",
    "sapq": "SAPQ.quantize", "sapq-block": "SAPQ.quantize",
    "sapq-layerwise": "SAPQ.quantize", "sapq-network": "SAPQ.quantize",
    "brecq": "BRECQ.quantize", "repq": "RepQ.quantize", "ptq4vit": "PTQ4ViT.quantize",
}
VARIANTS = {
    "sbpq": {"global": "sbpq"},
    "ppq": {"global": "ppq", "layerwise": "ppq-layerwise"},
    "sapq": {"global": "sapq", "network": "sapq-network", "block": "sapq-block", "layerwise": "sapq-layerwise"},
}


def build_parser(default_method=None, defaults=None, *, family=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default=default_method,
                        required=default_method is None)
    parser.add_argument("--config", help="JSON configuration; explicit command-line options take precedence.")
    if family in VARIANTS:
        parser.add_argument("--variant", choices=tuple(VARIANTS[family]), default=None)
    if family == "sapq":
        parser.add_argument("--origin", action="store_true",
                            help="Route to main's original global SAPQ experiment defaults.")
        parser.add_argument("--origin_root", help="Original repository/cache root; defaults to this checkout.")
        for option in ("frozen_batches", "sensitivity", "ranges", "initial_steps", "layers"):
            parser.add_argument(f"--origin_{option}", help=f"Explicit original {option} file.")
    parser.add_argument("--model_name", default="Poseidon-L")
    parser.add_argument("--dataset", default=os.environ.get(
        "PPQ_DATASET_NAME", "fluids.incompressible.PiecewiseConstants"))
    parser.add_argument("--traj_num", type=int, default=64)
    parser.add_argument("--random_calibration", choices=("y", "n"), default="n")
    parser.add_argument("--calibration_seed", type=int, default=42)
    parser.add_argument("--final_time", type=int, default=20)
    parser.add_argument("--time_back", type=int, default=0)
    parser.add_argument("--use_phyloss", default="no_phyloss")
    parser.add_argument("--DEV", default="cuda")
    parser.add_argument("--model_path", default=os.environ.get("PPQ_MODEL_PATH"))
    parser.add_argument("--data_path", default=os.environ.get(
        "PPQ_DATA_PATH", "poseidon/camlab-ethz/down_streams"))
    parser.add_argument("--save_path", help="Output parent for SBPQ/PPQ/BRECQ; parameter-named child is automatic.")
    parser.add_argument("--output_root", help="Alias for the output parent of parameter-named experiments.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Memory batch size; every selected trajectory/time pair is consumed.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="Quantization RNG seed (independent of trajectory selection).")
    parser.add_argument("--weight_bits", type=int, default=8)
    parser.add_argument("--activation_bits", type=int, default=None,
                        help="32 keeps activations floating point; method support is checked explicitly.")
    parser.add_argument("--init_bits", type=int, default=9)
    parser.add_argument("--target_bits", type=float, default=8)
    parser.add_argument("--epochs", type=int, default=21)
    parser.add_argument("--mc_samples", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=9.1e-4)
    parser.add_argument("--eta", type=float, default=1e-6)
    parser.add_argument("--prior_mode", choices=("ppq", "block_no_sens", "block_sens"),
                        default=os.environ.get("SAPQ_PRIOR_MODE", "block_sens"))
    parser.add_argument("--prior_scale", type=float, default=1e-4)
    parser.add_argument("--sigma0", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.005)
    parser.add_argument("--sensitivity", choices=("sobolev", "mse", "divergence"), default="sobolev")
    parser.add_argument("--sensitivity_normalization", choices=("raw", "normalized"), default="raw")
    parser.add_argument("--percentile", type=float, default=1e-4)
    parser.add_argument("--bmax_bits", type=int, default=20)
    parser.add_argument("--avg_cap_bits", type=float, default=8.0)
    parser.add_argument("--avg_cap_lam", type=float, default=400.0)
    parser.add_argument("--avg_cap_alpha", type=float, default=10.0)
    parser.add_argument("--reconstruction_iterations", type=int, default=20000)
    parser.add_argument("--rec_batchsize", type=int, default=32)
    parser.add_argument("--reconstruction_lr", type=float, default=1e-3)
    parser.add_argument("--ptq_candidates", type=int, default=100)
    parser.add_argument("--ptq_search_rounds", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep_work_dir", action="store_true",
                        help="Keep this run's temporary calibration/reconstruction caches.")
    parser.add_argument("--total_steps", "--total-steps", type=int, default=800,
                        help="SBPQ cumulative optimizer steps, including restored progress.")
    parser.add_argument("--resume", help="SBPQ/PPQ/BRECQ checkpoint file or experiment directory.")
    parser.add_argument("--save_every", "--save-every", type=int, default=100)
    parser.add_argument("--profile_interval", type=float, default=0.1, help="CPU process-tree sampling interval in seconds.")
    parser.add_argument("--beta_kappa", type=float, default=150.0)
    parser.add_argument("--delta_bits", type=float, default=2.0)
    parser.add_argument("--minimum_bits", type=float, default=1.0)
    parser.add_argument("--maximum_bits", type=float, default=16.0)
    parser.add_argument("--beta_epsilon", type=float, default=1e-6)
    parser.add_argument("--beta_relative_epsilon", type=float, default=1e-12)
    parser.add_argument("--likelihood_scale", type=float, default=1.0)
    parser.add_argument("--sobolev_order", type=int, default=2)
    parser.add_argument("--sobolev_order_weights", default="1",
                        help="SBPQ scalar broadcast or comma-separated weights for orders 0..k.")
    parser.add_argument("--sobolev_norm", choices=("l1", "mse"), default="l1")
    parser.add_argument("--task_loss_weight", type=float, default=1.0)
    parser.add_argument("--sobolev_loss_weight", type=float, default=1.0)
    parser.add_argument("--sobolev_transpose", action="store_true")
    if defaults:
        parser.set_defaults(**defaults)
    return parser


def validate_args(args):
    if getattr(args, "origin", False):
        from SAPQ.origin import apply_defaults
        apply_defaults(args)
    if args.resume and args.method not in {"sbpq", "brecq", "ppq", "ppq-layerwise"}:
        raise ValueError("--resume supports SBPQ, PPQ and BRECQ")
    if args.save_every < 1 or args.profile_interval <= 0 or args.reconstruction_lr <= 0:
        raise ValueError("save_every, profile_interval and reconstruction_lr must be positive")
    if args.method == "sbpq":
        from SBPQ.quantize import validate_settings
        validate_settings(args)
    if args.activation_bits is None:
        args.activation_bits = 8 if args.method == "repq" else 32
    if any(isinstance(value, float) and not math.isfinite(value) for value in vars(args).values()):
        raise ValueError("Numeric options must be finite")
    for name in ("traj_num", "batch_size", "epochs", "mc_samples", "rec_batchsize", "ptq_candidates", "ptq_search_rounds"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be positive")
    if args.num_workers < 0 or args.reconstruction_iterations < 1:
        raise ValueError("num_workers must be nonnegative and reconstruction_iterations positive")
    if not 2 <= args.weight_bits <= 16 or args.activation_bits not in (*range(2, 17), 32):
        raise ValueError("weight_bits must be 2..16; activation_bits must be 2..16 or 32")
    if not 0 < args.percentile < 0.5 or args.eta <= 0 or args.learning_rate <= 0:
        raise ValueError("Require 0 < percentile < 0.5 and positive eta/learning_rate")
    if not 0 < args.target_bits <= args.bmax_bits or not 1 <= args.init_bits <= args.bmax_bits:
        raise ValueError("target_bits/init_bits must be positive and no greater than bmax_bits")
    if args.bmax_bits > 30:
        raise ValueError("bmax_bits must not exceed 30")
    if args.sigma0 <= 0 or args.prior_scale < 0 or args.alpha < 0 or args.gamma < 0:
        raise ValueError("sigma0 must be positive; prior_scale, alpha and gamma must be nonnegative")
    if args.method in PROBABILISTIC_METHODS and args.activation_bits != 32:
        raise ValueError("PPQ/SAPQ/SBPQ currently implement weight-only quantization; use --activation_bits 32")
    if args.method == "uniform" and args.activation_bits != 32:
        raise ValueError("UniformQ supports weight-only quantization; use activation_bits=32")
    if args.method == "repq" and args.activation_bits == 32:
        raise ValueError("RepQ requires activation_bits < 32")
    if args.method == "brecq" and (args.activation_bits != 32 or args.weight_bits > 8):
        raise ValueError("BRECQ supports weight_bits=2..8 and activation_bits=32")
    if args.model_path is None and args.method == "sbpq":
        args.model_path = str(Path("models/poseidon") / args.dataset)
    if args.model_path is None:
        args.model_path = str(Path("checkpoints") / args.model_name / args.use_phyloss
                              / args.dataset / "origin" / "none")
    from .experiments import resolve_output
    resolve_output(args)
    if args.save_path is None:
        tag = f"traj{args.traj_num}"
        if args.random_calibration == "y":
            tag += f"_random_seed{args.calibration_seed}"
        tag += f"_ftime{args.final_time}_timeback{args.time_back}"
        budget = (f"init{args.init_bits}_target{args.target_bits:g}"
                  if args.method in PROBABILISTIC_METHODS
                  else f"w{args.weight_bits}a{args.activation_bits}")
        if args.method == "sbpq":
            budget += f"_k{args.beta_kappa:g}_d{args.delta_bits:g}_steps{args.total_steps}"
        if args.method.startswith("sapq"):
            budget += f"_{args.prior_mode}_{args.sensitivity}_{args.sensitivity_normalization}"
        args.save_path = str(Path("quantization_artifacts") / args.model_name / args.dataset
                             / args.method / f"{tag}_{budget}_seed{args.seed}")
        if getattr(args, "origin", False):
            args.save_path = str(Path("quantization_artifacts") / args.model_name / args.dataset
                                 / "sapq" / f"origin_main_seed{args.seed}")
    return args


def _load_model(path, device):
    import torch
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint is unavailable: {path}. Set --model_path explicitly.")
    if path.is_file():
        from . import compat
        model = torch.load(path, map_location="cpu", weights_only=False, pickle_module=compat)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("A model file must contain an nn.Module; use a pretrained directory for state dicts.")
    else:
        from scOT.model import ScOT
        model = ScOT.from_pretrained(str(path), local_files_only=True)
    return model.to(device).eval()


def _revision():
    root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _implementation_hashes():
    """Identify the actual source even when running uncommitted branch changes."""
    root = Path(__file__).resolve().parents[1]
    files = set()
    for name in ("quantization", "evaluation", "UniformQ", "DynamicQ", "StaticQ", "PPQ", "SAPQ",
                 "BRECQ", "RepQ", "PTQ4ViT", "SBPQ", "scOT", "physical"):
        files.update((root / name).rglob("*.py"))
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def run_quantization(args, *, model=None, calibration_loader=None, calibration_metadata=None):
    """Shared lifecycle, cost accounting and immutable experiment configuration."""
    import numpy as np
    import torch
    from .artifacts import save_quantized_artifact
    from .calibration import build_calibration_loader
    from .common import resolve_device
    from .progress import progress_write
    from .checkpoints import RESUMABLE, atomic_json
    from .costs import CostTracker, track, phase, counter
    from .experiments import register_experiment

    args = validate_args(args)
    device = resolve_device(args.DEV)
    destination = Path(args.save_path)
    progress_write(f'Experiment directory: {destination}')
    if (destination / "compressed_model.pt").exists() and not args.overwrite and not args.resume:
        raise FileExistsError(f"Artifact already exists: {destination}; choose another parent or --overwrite")
    if args.method in PROBABILISTIC_METHODS | {"brecq"}:
        from .runtime import check_optimizer_runtime
        progress_write("Checking PyTorch optimizer runtime...")
        check_optimizer_runtime(device, "AdamW" if args.method == "sbpq" else "Adam")
    if args.method in RESUMABLE:
        register_experiment(args)
        if args.overwrite and not args.resume:
            for name in ('work', 'checkpoints'):
                shutil.rmtree(destination / name, ignore_errors=True)
            for name in ('metadata.json', 'compressed_model.pt', 'quantization_state.pt',
                         'history.json', 'sbpq_checkpoint.pt', 'brecq_checkpoint.pt', 'ppq_checkpoint.pt',
                         'sbpq_trainer_state.pt', 'sbpq_preparation.pt', 'sbpq_beta_parameters.pt'):
                (destination / name).unlink(missing_ok=True)
    destination.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    work_dir = (destination / 'work' if args.method in RESUMABLE else
                Path(tempfile.mkdtemp(prefix='work-', dir=destination)))
    work_dir.mkdir(exist_ok=True)
    tracker = CostTracker(destination / 'costs.json', device, interval=args.profile_interval,
                          resume=bool(args.resume))
    tracker.session['source_revision'] = _revision()
    if args.resume and tracker.report.get('unmeasured_history_before_first_session') and (destination / 'metadata.json').exists():
        previous = json.loads((destination / 'metadata.json').read_text())
        tracker.report['legacy_reported_elapsed_quantization_seconds'] = previous.get('elapsed_quantization_seconds')
    tracker.session['runtime_settings'] = {'save_every': args.save_every, 'num_workers': args.num_workers,
                                           'profile_interval': args.profile_interval, 'resume': bool(args.resume)}
    started = time.perf_counter()
    succeeded = False
    final_status = 'interrupted'
    metadata = None
    atomic_json({'status': 'preparing', 'method': args.method}, destination / 'run_state.json')
    try:
        with track(tracker):
            needs_calibration = args.method not in {'fp', 'uniform', 'dynamic'} and not (
                args.method == 'static' and args.activation_bits == 32)
            with phase('calibration_loading'):
                if calibration_loader is None and needs_calibration:
                    progress_write('Preparing calibration dataset...')
                    if getattr(args, 'origin', False):
                        from SAPQ.origin import build_calibration_loader as build_origin_loader
                        _, calibration_loader, calibration_metadata = build_origin_loader(args)
                    else:
                        _, calibration_loader, calibration_metadata = build_calibration_loader(args)
                elif calibration_loader is not None and calibration_metadata is None:
                    raise ValueError('Injected calibration_loader requires calibration_metadata')
                elif calibration_metadata is None:
                    calibration_metadata = {'dataset': args.dataset, 'selection': 'not_required',
                                            'num_trajectories': 0, 'num_samples': 0,
                                            'reason': 'This method/configuration does not consume offline calibration data.'}
            with phase('model_loading'):
                if model is None:
                    progress_write(f'Loading full-precision model: {args.model_path}')
                    model = _load_model(args.model_path, device)
                model.to(device).eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
            args._calibration_metadata = calibration_metadata
            print('Calibration: ' + json.dumps({key: calibration_metadata.get(key) for key in
                  ('dataset', 'num_trajectories', 'num_samples', 'selection', 'fingerprint')}), flush=True)
            print(f'Quantizing with {args.method} on {device}; output: {destination}', flush=True)
            if args.method == 'fp':
                from .common import QuantizationResult
                result = QuantizationResult(model, {
                    'method': 'fp', 'weight_bits': 32, 'activation_bits': 32,
                    'execution': 'floating_point', 'packed_integer_weights': False,
                    'quantized_layers': [], 'calibration_samples': 0, 'calibration_batches': 0,
                })
            else:
                implementation = importlib.import_module(METHOD_MODULES[args.method])
                result = implementation.quantize(model, calibration_loader, args, device, work_dir)
            metadata = {
                'method': args.method, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'source_revision': _revision(), 'implementation_sha256': _implementation_hashes(),
                'source_model_path': str(Path(args.model_path).resolve()),
                'settings': {key: value for key, value in vars(args).items() if not key.startswith('_')},
                'calibration': calibration_metadata, 'torch_version': str(torch.__version__),
                'elapsed_quantization_seconds': time.perf_counter() - started,
                'costs_file': 'costs.json', 'evaluation_performed': False,
            }
            with phase('artifact_saving'):
                metadata = save_quantized_artifact(result, destination, metadata,
                                                  overwrite=args.overwrite or bool(args.resume))
                counter('artifact_bytes_written', sum((destination / name).stat().st_size for name in
                                                      ('compressed_model.pt', 'quantization_state.pt', 'metadata.json')))
            if not args.keep_work_dir:
                with phase('cache_cleanup'):
                    try:
                        shutil.rmtree(work_dir)
                    except OSError as cleanup_error:
                        tracker.session['cache_cleanup_error'] = str(cleanup_error)
            succeeded = True
            final_status = 'completed'
            atomic_json({'status': 'completed', 'method': args.method}, destination / 'run_state.json')
    except BaseException as error:
        final_status = 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed'
        atomic_json({'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     'method': args.method, 'error': f'{type(error).__name__}: {error}'},
                    destination / 'run_state.json')
        raise
    finally:
        tracker.close(final_status)
        if not succeeded:
            print(f'Preparation/checkpoints retained at {destination}; work: {work_dir}', flush=True)
    metadata['costs_summary'] = tracker.snapshot()['cumulative']
    atomic_json(metadata, destination / 'metadata.json')
    print(f"Recorded optimization time: {metadata['costs_summary']['optimization_seconds']:.3f}s; costs: {destination / 'costs.json'}", flush=True)
    print(f"Saved {destination / 'compressed_model.pt'}", flush=True)
    evaluation_time = (15 if args.dataset == 'wave.Gaussians' else 20) if getattr(args, 'origin', False) else args.final_time
    print('Evaluate separately with: ' + shlex.join([
        'python', '-m', 'evaluation.evaluate', '--model_path', str(destination),
        '--dataset', args.dataset, '--data_path', str(args.data_path),
        '--final_time', str(evaluation_time), '--DEV', args.DEV]), flush=True)
    return destination, metadata


def parse_args(argv=None, *, default_method=None, defaults=None, family=None):
    """Parse optional file defaults, then command-line overrides, then method variants."""
    parser = build_parser(default_method, defaults, family=family)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config")
    config_parser.add_argument("--method")
    config_parser.add_argument("--variant")
    config_parser.add_argument("--origin", action="store_true")
    config_parser.add_argument("--resume")
    explicit = config_parser.parse_known_args(argv)[0]
    config_path = explicit.config
    saved_defaults = {}
    if explicit.resume:
        from .experiments import resume_defaults
        saved_defaults = resume_defaults(explicit.resume, explicit.method or default_method)
    configured_method = saved_defaults.get('method')
    if config_path:
        try:
            configured_method = json.loads(Path(config_path).read_text(encoding="utf-8")).get("method")
        except (OSError, ValueError, AttributeError):
            pass  # The normal config parser below provides the detailed error.
    if (explicit.method or configured_method or default_method) == "sbpq":
        from SBPQ.defaults import DEFAULTS
        parser.set_defaults(**{**DEFAULTS, **(defaults or {})})
    elif (explicit.method or configured_method or default_method) == "brecq":
        parser.set_defaults(save_every=1000)
    if saved_defaults:
        actions = {action.dest: action for action in parser._actions}
        parser.set_defaults(**{key: value for key, value in saved_defaults.items() if key in actions})
        if 'method' in saved_defaults:
            actions['method'].required = False
    values = {}
    if config_path:
        try:
            values = json.loads(Path(config_path).read_text(encoding="utf-8"))
            if not isinstance(values, dict):
                raise ValueError("Configuration must be a JSON object")
            actions = {action.dest: action for action in parser._actions}
            for key, value in values.items():
                if key not in actions or key in {"help", "config"}:
                    raise ValueError(f"Unknown configuration option: {key}")
                action = actions[key]
                if isinstance(action, argparse._StoreTrueAction) and not isinstance(value, bool):
                    raise ValueError(f"Configuration option {key} must be boolean")
                if action.type in (int, float) and (value is None or isinstance(value, bool)):
                    raise ValueError(f"Configuration option {key} must be numeric, not null/boolean")
                if action.type is int and isinstance(value, float) and not value.is_integer():
                    raise ValueError(f"Configuration option {key} must be an integer")
                if action.type is not None:
                    value = action.type(value)
                if action.choices is not None and value not in action.choices:
                    raise ValueError(f"Invalid {key}: {value!r}; expected one of {action.choices}")
                values[key] = value
            parser.set_defaults(**values)
            if "method" in values:
                actions["method"].required = False
        except (OSError, ValueError, TypeError) as error:
            parser.error(str(error))
    if family == "sapq" and (explicit.origin or values.get("origin", False)):
        for key, value in {
            "model_path": os.environ.get("PPQ_MODEL_PATH", "models/NS-PwC-L"),
            "data_path": os.environ.get("PPQ_DATA_PATH", "dataset/NS-PwC"),
        }.items():
            if key not in values and key not in (defaults or {}):
                parser.set_defaults(**{key: value})
    args = parser.parse_args(argv)
    allowed = set(VARIANTS[family].values()) if family in VARIANTS else {family}
    if family is not None and args.method not in allowed:
        parser.error(f"{family} entry cannot run method {args.method}; use that method's entry")
    variant = getattr(args, "variant", None)
    if variant is not None:
        variant_method = VARIANTS[family][variant]
        if explicit.method is not None:
            if explicit.variant is not None and explicit.method != variant_method:
                parser.error("Explicit --method and --variant select different methods")
            # An explicit method wins over a variant supplied only by config.
        elif explicit.variant is None and "method" in values and values["method"] != variant_method:
            parser.error("Configuration method and variant select different methods")
        else:
            args.method = variant_method
    if family in VARIANTS:
        args.variant = next(key for key, method in VARIANTS[family].items() if method == args.method)
    if getattr(args, "origin", False):
        from SAPQ.origin import apply_defaults
        apply_defaults(args)
    return parser, args


def main(argv=None, *, default_method=None, defaults=None, family=None):
    parser, args = parse_args(argv, default_method=default_method, defaults=defaults, family=family)
    try:
        run_quantization(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))


def method_main(family, argv=None):
    """Shared lifecycle used by each method's single launcher."""
    if family not in METHODS:
        raise ValueError(f"Unknown method family: {family}")
    return main(argv, default_method=family, family=family)


if __name__ == "__main__":
    main()
