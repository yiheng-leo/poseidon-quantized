"""Evaluate quantized Poseidon artifacts together and reuse matching results."""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path

import torch

from quantization.calibration import _fingerprint, dataset_metadata
from quantization.checkpoints import atomic_json
from . import evaluate


def file_identity(path):
    info = path.stat()
    return {
        "path": str(path.resolve()), "bytes": info.st_size,
        "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns,
    }


def artifact_identity(model_path):
    path = Path(model_path).resolve()
    weights = path / "compressed_model.pt" if path.is_dir() else path
    metadata = weights.with_name("metadata.json")
    return {
        "weights": file_identity(weights),
        "metadata": file_identity(metadata) if metadata.is_file() else None,
    }


def checkpoint_identity(checkpoint):
    if not checkpoint:
        return None
    path = Path(checkpoint).resolve(strict=True)
    if path.is_file():
        return [file_identity(path)]
    files = sorted(p for p in path.iterdir() if p.is_file() and (
        p.name == "config.json" or p.suffix in {".bin", ".safetensors"}
        or p.name.endswith(".index.json")
    ))
    if not files:
        raise ValueError(f"No checkpoint files found in {path}")
    return [file_identity(p) for p in files]


def evaluation_protocol(args, dataset):
    device = torch.device(args.DEV)
    root = Path(__file__).resolve().parents[2]
    sources = (
        "evaluation/poseidon/evaluate.py", "evaluation/poseidon/physics.py",
        "evaluation/poseidon/PDE_calc.py", "evaluation/reporting.py",
        "scOT/model.py", "scOT/trainer.py",
    )
    return {
        "protocol": "safe-svd", "dataset": args.dataset, "split": args.split,
        "data_path": str(Path(args.data_path).resolve()), "samples": len(dataset),
        "dataset_source": dataset_metadata(dataset),
        "final_time": args.final_time, "initial_time": args.initial_time,
        "ar_steps": args.ar_steps, "eval_obj": args.eval_obj,
        "num_trajectories": 64 if args.num_trajectories is None else args.num_trajectories,
        "batch_size": args.batch_size, "physics_batch_size": args.physics_batch_size,
        "num_workers": args.num_workers, "checkpoint": checkpoint_identity(args.checkpoint),
        "device": str(device), "torch_version": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "implementation": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                           for name in sources},
    }


def load_report(output, dataset):
    if not output.is_file():
        return {"schema_version": 1, "dataset": dataset, "protocols": {}}
    report = json.loads(output.read_text())
    if (not isinstance(report, dict) or report.get("schema_version") != 1 or report.get("dataset") != dataset
            or not isinstance(report.get("protocols"), dict)):
        raise ValueError(f"Expected a batch evaluation report for {dataset}: {output}")
    return report


def run_batch(args, output):
    evaluate.validate_args(args)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for path in args.models:
        identity = artifact_identity(path)
        artifacts[identity["weights"]["path"]] = identity

    # Serialize updates so concurrent jobs cannot overwrite completed entries or
    # evaluate the same model and baseline twice.
    with output.with_suffix(output.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        dataset = evaluate.open_test_dataset(args)
        protocol = evaluation_protocol(args, dataset)
        key = _fingerprint(protocol)
        report = load_report(output, args.dataset)
        group = report["protocols"].setdefault(
            key, {"settings": protocol, "origin_metrics": None, "models": {}})
        pending, skipped = [], []
        for path, identity in artifacts.items():
            previous = group["models"].get(path)
            if (previous and previous.get("artifact") == identity
                    and previous.get("result", {}).get("metrics")):
                skipped.append(path)
            else:
                pending.append(path)
        if not pending:
            return {"profile": key, "evaluated": [], "skipped": skipped}

        origin = group["origin_metrics"]
        if args.checkpoint and (origin is None or args.eval_obj == "oncomp"):
            print("Evaluating shared original checkpoint", flush=True)
            dataset, origin = evaluate.evaluate_origin(args, dataset, cached_metrics=origin)
            group["origin_metrics"] = origin
            atomic_json(report, output)

        for path in pending:
            print(f"Evaluating {path}", flush=True)
            result = evaluate.evaluate_compressed(args, dataset, path, origin)
            if artifact_identity(path) != artifacts[path]:
                raise RuntimeError(f"Artifact changed during evaluation: {path}")
            group["models"][path] = {"artifact": artifacts[path], "result": result}
            atomic_json(report, output)
        return {"profile": key, "evaluated": pending, "skipped": skipped}
