#!/usr/bin/env python3
"""Run one v2 NS-PwC 8-bit gamma candidate inside a Bash tmux session.

Activate poseidon-quant first. This supervisor only launches the existing PPQ
entry point, records provenance, and removes reconstructible caches belonging
to layers preceding the most recently completed layer. It changes no training
settings, sampling, or quantization implementation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prune_completed_caches(run_dir, epochs):
    history_file = run_dir / "history.json"
    if not history_file.exists():
        return [], None
    history = json.loads(history_file.read_text())
    completed = list(dict.fromkeys(row["unit"] for row in history if row["epoch"] == epochs))
    removed = []
    # Retain the most recent completed layer too. Every earlier layer has
    # necessarily been passed by the checkpoint cursor, including on resume.
    for index, unit in enumerate(completed[:-1]):
        directory = run_dir / "work" / f"ppq-unit-{index}"
        manifest = directory / "manifest.json"
        if directory.is_symlink() or not manifest.is_file():
            continue
        identity = json.loads(manifest.read_text()).get("identity", {})
        if identity.get("unit") == unit:
            shutil.rmtree(directory)
            removed.append({"index": index, "unit": unit})
    return removed, history[-1] if history else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", required=True,
                        choices=("3e-7", "1e-6", "3e-6", "1e-5", "3e-5"))
    parser.add_argument("--gpu_uuid", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--wait_for_idle", action="store_true",
                        help="Wait without allocating CUDA memory until GPU memory use is below 1000 MiB.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    name = f"ppq-layerwise-traj16-ft10-bit8-default-gamma{args.gamma}"
    run_dir = root / "quantization_artifacts/NS-PwC/sbpq-led-v2/tuning" / name
    log_dir = root / "logs/poseidon/NS-PwC/sbpq-led-v2/tuning"
    log_dir.mkdir(parents=True, exist_ok=True)
    lock = (log_dir / f"{name}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run_dir / "experiment.json").exists() or (run_dir / "compressed_model.pt").exists():
        raise FileExistsError(f"Existing experiment must be inspected/resumed explicitly: {run_dir}")
    launch_path = log_dir / f"{name}.launch.json"
    if launch_path.exists():
        raise FileExistsError(f"Launch already registered: {launch_path}")
    inventory_command = [
        "nvidia-smi", f"--id={args.gpu_uuid}",
        "--query-gpu=index,uuid,name,memory.used,memory.total", "--format=csv,noheader,nounits",
    ]
    queue_path = log_dir / f"{name}.queue.json"
    queue = {"created_at_utc": utc(), "host": os.uname().nodename,
             "gpu_uuid": args.gpu_uuid, "tmux_session": args.session,
             "supervisor_pid": os.getpid(), "gamma": float(args.gamma),
             "run_dir": str(run_dir), "protocol": "sbpq-led-v2"}
    while True:
        inventory = subprocess.check_output(inventory_command, text=True).strip()
        if int(inventory.split(",")[-2].strip()) < 1000:
            break
        if not args.wait_for_idle:
            raise RuntimeError(f"GPU is no longer idle: {inventory}")
        queue.update(status="waiting_for_idle_gpu", checked_at_utc=utc(), gpu_inventory=inventory)
        write_json(queue_path, queue)
        print(f"{utc()} waiting for idle GPU: {inventory}", flush=True)
        time.sleep(30)
    model = "/cs/research/vecg/EasyPDE/model-store/poseidon/fluids.incompressible.PiecewiseConstants"
    data = ("/cs/research/vecg/EasyPDE/quant/"
            "Structural-Aware-Probablistic-Quantization-for-Foundational-Physical-Models/"
            "poseidon/camlab-ethz/down_streams")
    command = [
        sys.executable, "-u", "-m", "PPQ.ppq",
        "--config", "configs/ppq/layerwise.json", "--variant", "layerwise",
        "--model_path", model, "--dataset", "fluids.incompressible.PiecewiseConstants",
        "--data_path", data, "--traj_num", "16", "--random_calibration", "n",
        "--calibration_seed", "42", "--final_time", "10", "--time_back", "0",
        "--batch_size", "2", "--num_workers", "0", "--activation_bits", "32",
        "--seed", "1", "--DEV", "cuda", "--target_bits", "8", "--init_bits", "8",
        "--mc_samples", "10", "--epochs", "5", "--learning_rate", "9.1e-4",
        "--eta", "1e-6", "--percentile", "1e-4", "--gamma", args.gamma,
        "--save_every", "1000", "--save_path", str(run_dir),
    ]
    sources = ["PPQ/ppq.py", "PPQ/quantize.py", "PPQ/loss.py", "scOT/model.py",
               "quantization/runner.py", "quantization/common.py", "quantization/probabilistic.py",
               "quantization/cache.py", "quantization/calibration.py", "configs/ppq/layerwise.json"]
    record = {
        "protocol": "sbpq-led-v2", "purpose": "NS-PwC 8-bit PPQ layerwise gamma coarse sweep",
        "created_at_utc": utc(), "host": os.uname().nodename, "gpu_uuid": args.gpu_uuid,
        "gpu_inventory_before_launch": inventory, "tmux_session": args.session,
        "supervisor_pid": os.getpid(), "command": command, "run_dir": str(run_dir),
        "log": str(log_dir / f"{name}.log"), "gamma": float(args.gamma),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "implementation_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources},
        "supervisor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cache_housekeeping": "Prune only layers before the latest checkpointed epoch-5 layer; retain calibration/current layer/checkpoint.",
    }
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    env["PYTHONUNBUFFERED"] = "1"
    status_path = log_dir / f"{name}.supervisor.json"
    with Path(record["log"]).open("x") as log:
        log.write(json.dumps(record, indent=2) + "\n")
        log.flush()
        child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        record["pid"] = child.pid
        write_json(launch_path, record)
        if args.wait_for_idle:
            queue.update(status="started", checked_at_utc=utc(), pid=child.pid)
            write_json(queue_path, queue)
        print(json.dumps(record, indent=2), flush=True)
        pruned = []
        while child.poll() is None:
            housekeeping_error = None
            try:
                removed, last_epoch = prune_completed_caches(run_dir, 5)
            except (OSError, ValueError, KeyError, TypeError) as error:
                # A housekeeping problem must never stop the training process.
                removed, last_epoch = [], None
                housekeeping_error = f"{type(error).__name__}: {error}"
                print(f"{utc()} cache housekeeping: {housekeeping_error}", flush=True)
            pruned.extend(removed)
            status = {"checked_at_utc": utc(), "status": "running", "pid": child.pid,
                      "last_saved_epoch": last_epoch, "pruned_layer_caches": pruned}
            if housekeeping_error:
                status["housekeeping_error"] = housekeeping_error
            write_json(status_path, status)
            if removed:
                print(f"{utc()} pruned {len(removed)} completed-layer cache(s)", flush=True)
            time.sleep(30)
        status = {"checked_at_utc": utc(), "status": "exited", "pid": child.pid,
                  "returncode": child.returncode, "pruned_layer_caches": pruned}
        if child.returncode == 0:
            metadata = json.loads((run_dir / "metadata.json").read_text())
            bits = metadata["quantization"]["avg_weight_bits"]
            status.update(status="completed", avg_weight_bits=bits, matches_8bit_budget=7.9 <= bits <= 8)
        write_json(status_path, status)
        print(json.dumps(status, indent=2), flush=True)
    return child.returncode


if __name__ == "__main__":
    sys.exit(main())
