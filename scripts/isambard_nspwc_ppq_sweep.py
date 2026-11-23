#!/usr/bin/env python3
"""Search one NS-PwC PPQ response curve for 2, 3, 4 and 8 weighted bits."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import subprocess
import sys
from pathlib import Path

from run_ppq_nspwc_bit8_gamma import prune_completed_caches


DATASET = "fluids.incompressible.PiecewiseConstants"
ROOT = Path("results/NS-PwC")
SWEEP = ROOT / "ppq-layerwise-traj16-ft10-b32-sweep"
TARGETS = (2, 3, 4, 8)
GAMMAS = (
    "1e-9", "3e-9", "1e-8", "2e-8", "3e-8", "5e-8", "7e-8", "1e-7",
    "1.5e-7", "2e-7", "3e-7", "5e-7", "7e-7", "1.5e-6", "2e-6",
    "3e-6", "5e-6", "1e-5", "3e-5", "1e-4", "3e-4", "1e-3", "3e-3", "1e-2",
)
SETTINGS = {
    "dataset": DATASET, "seed": 1, "traj_num": 16, "random_calibration": "n",
    "calibration_seed": 42, "final_time": 10, "time_back": 0,
    "batch_size": 32, "num_workers": 0, "activation_bits": 32,
    "target_bits": 8, "init_bits": 8, "mc_samples": 10, "epochs": 5,
    "learning_rate": 9.1e-4, "eta": 1e-6, "percentile": 1e-4,
    "model_path": f"models/poseidon/{DATASET}",
    "data_path": "poseidon/camlab-ethz/down_streams",
}


def run_dir(gamma):
    mantissa, exponent = f"{float(gamma):.7e}".split("e")
    return SWEEP / f"gamma-{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"


def verify_settings(settings):
    for key, expected in SETTINGS.items():
        actual = settings.get(key)
        if key in {"model_path", "data_path"}:
            actual, expected = Path(actual).resolve(), Path(expected).resolve()
        if actual != expected:
            raise ValueError(f"PPQ sweep {key} differs: {actual!r} != {expected!r}")
    if settings.get("method") != "ppq-layerwise":
        raise ValueError("Sweep requires PPQ layerwise")


def read_point(path):
    state, meta = path / "run_state.json", path / "metadata.json"
    if not (state.is_file() and meta.is_file() and (path / "compressed_model.pt").is_file()):
        return None
    if json.loads(state.read_text()).get("status") != "completed":
        return None
    record = json.loads(meta.read_text())
    settings = record["settings"]
    verify_settings(settings)
    bits = record["quantization"]["avg_weight_bits"]
    if not math.isfinite(bits):
        raise ValueError(f"Nonfinite final average bits in {meta}")
    return {"gamma": settings["gamma"], "avg_weight_bits": bits, "model_path": str(path)}


def completed_points():
    paths = [ROOT / "ppq-traj16-ft10", *sorted(SWEEP.glob("gamma-*"))]
    return sorted((point for path in paths if (point := read_point(path))),
                  key=lambda point: point["gamma"])


def run_candidate(gamma):
    directory = run_dir(gamma)
    SWEEP.mkdir(parents=True, exist_ok=True)
    with directory.with_name(directory.name + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if read_point(directory):
            return
        command = [sys.executable, "-u", "-m", "PPQ.ppq"]
        experiment = directory / "experiment.json"
        if experiment.is_file():
            settings = json.loads(experiment.read_text())["settings"]
            verify_settings(settings)
            if not math.isclose(settings["gamma"], float(gamma), rel_tol=1e-7):
                raise ValueError(f"Saved gamma differs from {gamma}: {experiment}")
            command += ["--resume", str(directory), "--save_every", "1000"]
        else:
            command += ["--config", "configs/ppq/layerwise.json", "--variant", "layerwise",
                        "--DEV", "cuda", "--gamma", str(gamma), "--save_every", "1000",
                        "--save_path", str(directory)]
            for key, value in SETTINGS.items():
                command.extend((f"--{key}", str(value)))
        print(f"Running gamma={gamma} in {directory}", flush=True)
        with subprocess.Popen(command) as child:
            while True:
                try:
                    status = child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        prune_completed_caches(directory, SETTINGS["epochs"])
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        print(f"PPQ cache cleanup: {error}", file=sys.stderr, flush=True)
                else:
                    if status:
                        raise subprocess.CalledProcessError(status, command)
                    break
        if not read_point(directory):
            raise RuntimeError(f"PPQ did not export a completed model: {directory}")


def select(points, target):
    return max((p for p in points if target - 0.1 <= p["avg_weight_bits"] <= target),
               key=lambda p: p["avg_weight_bits"], default=None)


def next_candidate(points, target):
    if not points:
        raise ValueError("No completed PPQ candidates")
    pairs = [
        (left, right)
        for left, right in zip(points, points[1:])
        if left["avg_weight_bits"] > target and right["avg_weight_bits"] < target - 0.1
    ]
    if pairs:
        left, right = min(pairs, key=lambda pair: pair[1]["gamma"] / pair[0]["gamma"])
        fraction = (left["avg_weight_bits"] - (target - 0.05)) / (
            left["avg_weight_bits"] - right["avg_weight_bits"])
        fraction = min(0.8, max(0.2, fraction))
        gamma = left["gamma"] * (right["gamma"] / left["gamma"]) ** fraction
    elif all(p["avg_weight_bits"] < target - 0.1 for p in points):
        gamma = points[0]["gamma"] / 10
    elif all(p["avg_weight_bits"] > target for p in points):
        gamma = points[-1]["gamma"] * 10
    else:
        raise ValueError(f"No gamma bracket for {target} bits; inspect completed points")
    token = format(gamma, ".8g")
    if any(math.isclose(float(token), p["gamma"], rel_tol=1e-7) for p in points):
        raise ValueError(f"Gamma resolution exhausted for {target} bits")
    return token


def refine(target):
    for _ in range(4):
        points = completed_points()
        if select(points, target):
            break
        run_candidate(next_candidate(points, target))
    selected = select(completed_points(), target)
    print(json.dumps({"target_bits": target, "matched": selected}), flush=True)


def report(selected_only):
    points = completed_points()
    selections = {target: select(points, target) for target in TARGETS}
    if selected_only:
        for point in selections.values():
            if point:
                print(point["model_path"])
        return
    print(json.dumps({"points": points, "selections": selections}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    coarse = commands.add_parser("coarse")
    coarse.add_argument("index", type=int, choices=range(len(GAMMAS)))
    refinement = commands.add_parser("refine")
    refinement.add_argument("target", type=int, choices=TARGETS)
    summary = commands.add_parser("report")
    summary.add_argument("--selected-only", action="store_true")
    args = parser.parse_args()
    if args.command == "coarse":
        run_candidate(GAMMAS[args.index])
    elif args.command == "refine":
        refine(args.target)
    else:
        report(args.selected_only)


if __name__ == "__main__":
    main()
