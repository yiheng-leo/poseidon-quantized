#!/usr/bin/env python3
"""Finite, shared-filesystem GPU waiting queue for the sbpq-led-v2 experiments.

Workers use only the Python standard library while waiting. Quantization and
evaluation run through the existing repository entry points, one child per GPU.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import time

PROTOCOL = "sbpq-led-v2"
SERVER_ROOT = Path("/cs/research/vecg/EasyPDE/quant/poseidon-quantized")
DATA_ROOT = ("/cs/research/vecg/EasyPDE/quant/"
             "Structural-Aware-Probablistic-Quantization-for-Foundational-Physical-Models/"
             "poseidon/camlab-ethz/down_streams")
MODEL_ROOT = Path("/cs/research/vecg/EasyPDE/model-store/poseidon")
DATASETS = {"NS-PwC": "fluids.incompressible.PiecewiseConstants",
            "CE-RM": "fluids.compressible.RichtmyerMeshkov", "Wave-Gauss": "wave.Gaussians"}
GAMMAS = {2: ["2.5e-2", "5e-2", "1e-1"], 3: ["7.5e-3", "1.25e-2", "2e-2"],
          4: ["2e-3", "3e-3", "5e-3"], 8: ["3e-6", "1e-5", "3e-5"]}
MODULES = {"uniform": "UniformQ.uniformq", "sbpq": "SBPQ.sbpq",
           "brecq": "BRECQ.brecq", "ppq-layerwise": "PPQ.ppq"}
CONFIGS = {"uniform": "configs/uniform/default.json", "sbpq": "configs/sbpq/default.json",
           "brecq": "configs/brecq/default.json", "ppq-layerwise": "configs/ppq/layerwise.json"}
TERMINAL = {"completed", "failed", "interrupted", "external_failed"}
GAMMA_PLACEHOLDER = "__SELECTED_GAMMA__"


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".queue-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def group_id(label, bits):
    return f"{label}-bit{bits}"


def make_job(root, label, method, bits, seed, *, gamma=None, tuning=False, external=False):
    root = Path(root)
    tag, random, cal_seed = ("default", "n", 42) if seed == 1 else (f"seed{seed}", "y", seed)
    name = f"{method}-traj16-ft10-bit{bits}-{tag}"
    if tuning:
        name += f"-gamma{gamma}"
    job_id = f"{label}__{name}"
    parent = Path(label) / PROTOCOL
    if tuning:
        parent /= "tuning"
    run_dir = root / "quantization_artifacts" / parent / name
    log_dir = root / "logs/poseidon" / parent
    expected = {"method": method, "dataset": DATASETS[label], "model_path": str(MODEL_ROOT / DATASETS[label]),
                "data_path": DATA_ROOT, "traj_num": 16, "random_calibration": random,
                "calibration_seed": cal_seed, "final_time": 10, "time_back": 0,
                "batch_size": 2, "num_workers": 0, "activation_bits": 32, "seed": seed, "DEV": "cuda"}
    options = {key: value for key, value in expected.items() if key != "method"}
    command = ["{python}", "-u", "-m", MODULES[method], "--config", CONFIGS[method]]
    priority = 60 if tuning and bits != 8 else 30 if tuning else {"uniform": 10, "ppq-layerwise": 15,
                                                                "sbpq": 20, "brecq": 40}[method]
    gamma_group = None
    if method == "sbpq":
        options.update(target_bits=bits, init_bits=8, mc_samples=10, total_steps=800,
                       learning_rate=3.75e-5, eta=1e-6, percentile=1e-4, prior_scale=1,
                       likelihood_scale=1, beta_kappa=150, delta_bits=2, minimum_bits=1,
                       maximum_bits=16, sobolev_order=2 if label == "NS-PwC" else 1,
                       sobolev_norm="l1", sobolev_order_weights="1", task_loss_weight=1,
                       sobolev_loss_weight=1)
        save_every = 100
        if bits != 8:
            priority = 50
    elif method == "ppq-layerwise":
        command += ["--variant", "layerwise"]
        options.update(target_bits=bits, init_bits=8, mc_samples=10, epochs=5,
                       learning_rate=9.1e-4, eta=1e-6, percentile=1e-4)
        options["gamma"] = float(gamma) if gamma is not None else GAMMA_PLACEHOLDER
        gamma_group = None if tuning else group_id(label, bits)
        save_every = 1000
    elif method == "brecq":
        options.update(weight_bits=bits, reconstruction_iterations=20000, rec_batchsize=32,
                       reconstruction_lr=1e-3)
        save_every = 1000
    else:
        options.update(weight_bits=bits)
        save_every = 1000
    expected.update(options)
    if method == "sbpq":
        expected["sobolev_order_weights"] = 1.0  # Normalized by validate_settings.
    for key, value in options.items():
        command += [f"--{key}", str(value)]
    command += ["--save_every", str(save_every), "--save_path", str(run_dir)]
    evaluation = []
    output = root / "quantization_evaluation" / parent / f"{name}.json"
    if not tuning:
        evaluation = ["{python}", "-u", "-m", "evaluation.evaluate", "--protocol", "safe-svd",
                      "--split", "test", "--model_path", str(run_dir), "--checkpoint", expected["model_path"],
                      "--dataset", DATASETS[label], "--data_path", DATA_ROOT, "--final_time", "10",
                      "--eval_obj", "ongt", "--batch_size", "8", "--physics_batch_size", "10",
                      "--num_workers", "0", "--DEV", "cuda", "--output_path", str(output)]
    return {"id": job_id, "label": label, "method": method, "bits": bits, "seed": seed,
            "calibration": tag, "kind": "tuning" if tuning else "formal", "gamma": gamma,
            "gamma_group": gamma_group, "tuning_group": group_id(label, bits) if tuning else None,
            "external": external, "priority": priority, "minimum_gpu_memory_mib": 14000,
            "expected": expected, "quantize": command, "evaluate": evaluation, "save_every": save_every,
            "run_dir": str(run_dir), "log": str(log_dir / f"{name}.log"),
            "eval_log": str(log_dir / f"{name}-eval.log"), "eval_output": str(output),
            "legacy_supervisor": str(log_dir / f"{name}.supervisor.json") if external else None,
            "legacy_launch": str(log_dir / f"{name}.launch.json") if external else None}


def build_plan(root):
    jobs = []
    for label in DATASETS:
        for method in MODULES:
            for seed in (1, 7, 42):
                jobs.append(make_job(root, label, method, 8, seed))
        for bits in (2, 3, 4):
            for seed in (1, 7, 42):
                jobs.append(make_job(root, label, "sbpq", bits, seed))
    groups = []
    for label, bits in [("NS-PwC", b) for b in (2, 3, 4, 8)] + [("CE-RM", 8), ("Wave-Gauss", 8)]:
        groups.append({"id": group_id(label, bits), "label": label, "bits": bits,
                       "max_additional_candidates": 12, "match_lower": bits - 0.1, "match_upper": bits})
        for gamma in GAMMAS[bits]:
            jobs.append(make_job(root, label, "ppq-layerwise", bits, 1, gamma=gamma, tuning=True,
                                 external=label == "NS-PwC" and bits == 8))
    assert len(jobs) == len({job["id"] for job in jobs}) == 81
    return {"schema_version": 1, "protocol": PROTOCOL, "created_at_utc": utc(), "root": str(root),
            "poll_seconds": 30, "maximum_idle_gpu_memory_mib": 1000, "minimum_free_disk_gib": 50,
            "low_bit_ppq_scope": ["NS-PwC"], "evaluation_protocol": {"protocol": "safe-svd", "split": "test",
             "final_time": 10, "eval_obj": "ongt", "batch_size": 8, "physics_batch_size": 10, "num_workers": 0},
            "counts": {"formal_8bit": 36, "formal_lowbit_sbpq": 27, "lowbit_ppq_coarse": 9,
                       "auxiliary_8bit_ppq_coarse": 6, "existing_8bit_ppq_coarse": 3},
            "groups": groups, "jobs": jobs}


def all_jobs(queue, plan):
    return plan["jobs"] + [read_json(path) for path in sorted((queue / "additional").glob("*.json"))]


def state_path(queue, job):
    return queue / "states" / f'{job["id"]}.json'


def selection_path(queue, group):
    return queue / "selections" / f"{group}.json"


def resolve_job(queue, job):
    job = json.loads(json.dumps(job))
    if job["gamma_group"]:
        selected = read_json(selection_path(queue, job["gamma_group"]))
        if not selected or selected.get("status") != "selected":
            return None
        if selected.get("group") != job["gamma_group"] or selected.get("bits") != job["bits"]:
            raise ValueError("PPQ gamma selection belongs to a different dataset/bit group")
        value = selected["gamma"]
        if not (job["bits"] - 0.1 <= selected["avg_weight_bits"] <= job["bits"]):
            raise ValueError("Selected PPQ gamma does not match its group's bit budget")
        job["gamma_selection"] = selected
        job["expected"]["gamma"] = value
        job["quantize"] = [str(value) if arg == GAMMA_PLACEHOLDER else arg for arg in job["quantize"]]
    return job


def verify_settings(job, settings):
    for key, value in job["expected"].items():
        actual = settings.get(key)
        if key in {"model_path", "data_path"} and actual:
            actual, value = str(Path(actual).resolve()), str(Path(value).resolve())
        if actual != value:
            raise ValueError(f'{job["id"]}: {key} differs: {actual!r} != {value!r}')


def completed_artifact(job):
    directory = Path(job["run_dir"])
    state = read_json(directory / "run_state.json", {})
    if state.get("status") != "completed" or not (directory / "compressed_model.pt").is_file():
        return None
    metadata = read_json(directory / "metadata.json")
    if metadata is None:
        return None
    verify_settings(job, metadata["settings"])
    quantization = metadata["quantization"]
    bits = quantization.get("avg_weight_bits", quantization.get("weight_bits"))
    if not isinstance(bits, (int, float)) or not math.isfinite(bits):
        raise ValueError(f'Final average bits missing: {job["id"]}')
    return {"avg_weight_bits": bits, "metadata": str(directory / "metadata.json"),
            "model_sha256": metadata.get("model_sha256"), "metadata_sha256": digest(directory / "metadata.json")}


def next_gamma(points, bits):
    """Match the budget, bracket by gamma, then bisect in log space."""
    matched = [point for point in points if bits - 0.1 <= point["avg_weight_bits"] <= bits]
    if matched:
        return "selected", max(matched, key=lambda point: point["avg_weight_bits"])
    ordered = sorted(points, key=lambda point: point["gamma"])
    if any(right["avg_weight_bits"] > left["avg_weight_bits"] + 0.05
           for left, right in zip(ordered, ordered[1:])):
        return "needs_attention", "Observed bits are not monotone in gamma; inspect completed runs."
    if all(point["avg_weight_bits"] > bits for point in points):
        candidate = ordered[-1]["gamma"] * 10
    elif all(point["avg_weight_bits"] < bits - 0.1 for point in points):
        candidate = ordered[0]["gamma"] / 10
    else:
        pairs = [(left, right) for left, right in zip(ordered, ordered[1:])
                 if left["avg_weight_bits"] > bits and right["avg_weight_bits"] < bits - 0.1]
        if not pairs:
            return "needs_attention", "No consistent gamma bracket was found."
        left, right = min(pairs, key=lambda pair: pair[1]["gamma"] / pair[0]["gamma"])
        candidate = math.sqrt(left["gamma"] * right["gamma"])
    if not 1e-12 <= candidate <= 1e3:
        return "needs_attention", "Gamma expansion exceeded the configured bounds [1e-12, 1e3]."
    mantissa, exponent = f"{candidate:.7e}".split("e")
    token = mantissa.rstrip("0").rstrip(".") + "e" + str(int(exponent))
    if any(math.isclose(float(token), point["gamma"], rel_tol=1e-7) for point in points):
        return "needs_attention", "Gamma interval exhausted numerical precision."
    return "candidate", token


def reconcile(queue, plan):
    """Reconcile legacy runs and resolve gamma dependencies without GPU access."""
    with (queue / "coordinator.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        jobs = all_jobs(queue, plan)
        for job in jobs:
            if not job["external"]:
                continue
            previous = read_json(state_path(queue, job), {})
            if previous.get("status") in TERMINAL:
                continue
            artifact = completed_artifact(job)
            supervisor = read_json(job["legacy_supervisor"], {})
            launch = read_json(job["legacy_launch"], {})
            status = "completed" if artifact else "external_failed" if supervisor.get("status") == "exited" else "external_running"
            atomic_json(state_path(queue, job), {"id": job["id"], "status": status,
                        "checked_at_utc": utc(), "host": launch.get("host"), "pid": launch.get("pid"),
                        "supervisor_pid": launch.get("supervisor_pid"), "artifact": artifact,
                        "legacy_supervisor": job["legacy_supervisor"]})
        for group in plan["groups"]:
            destination = selection_path(queue, group["id"])
            previous = read_json(destination, {})
            if previous.get("status") in {"selected", "needs_attention"}:
                continue
            candidates = [job for job in jobs if job["tuning_group"] == group["id"]]
            points, pending, failures = [], [], []
            for job in candidates:
                artifact = completed_artifact(job)
                if artifact:
                    points.append({"gamma": float(job["gamma"]), "job_id": job["id"],
                                   "run_dir": job["run_dir"], **artifact})
                elif read_json(state_path(queue, job), {}).get("status") in TERMINAL:
                    failures.append(job["id"])
                else:
                    pending.append(job["id"])
            report = {"group": group["id"], "label": group["label"], "bits": group["bits"],
                      "checked_at_utc": utc(), "points": points, "pending": pending, "failures": failures}
            if failures:
                report.update(status="needs_attention", reason="A gamma candidate failed; its checkpoint/log requires inspection.")
            elif pending:
                report.update(status="waiting_candidates")
            else:
                decision, value = next_gamma(points, group["bits"])
                if decision == "selected":
                    report.update(status="selected", **value, criterion="B-0.1 <= final parameter-weighted avg_weight_bits <= B")
                elif decision == "needs_attention":
                    report.update(status=decision, reason=value)
                elif len(candidates) - len(GAMMAS[group["bits"]]) >= group["max_additional_candidates"]:
                    report.update(status="needs_attention", reason="Reached the maximum of 12 additional gamma candidates.")
                else:
                    new_job = make_job(plan["root"], group["label"], "ppq-layerwise", group["bits"],
                                       1, gamma=value, tuning=True)
                    path = queue / "additional" / f'{new_job["id"]}.json'
                    if not path.exists():
                        atomic_json(path, new_job)
                        atomic_json(state_path(queue, new_job), {"id": new_job["id"], "status": "queued",
                                    "created_at_utc": utc(), "reason": "adaptive_gamma_search"})
                    report.update(status="waiting_candidates", next_gamma=value, pending=[new_job["id"]])
            atomic_json(destination, report)


def code_hashes(root):
    paths = set()
    for directory in ("quantization", "PPQ", "SBPQ", "BRECQ", "UniformQ", "evaluation", "scOT", "physical"):
        paths.update((root / directory).rglob("*.py"))
    paths.update(root / value for value in CONFIGS.values())
    return {str(path.relative_to(root)): digest(path) for path in sorted(paths)}


def initialize(queue, root):
    queue, root = queue.resolve(), root.resolve()
    queue.mkdir(parents=True, exist_ok=True)
    with (queue / "initialize.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (queue / "plan.json").exists():
            raise FileExistsError(f"Queue already initialized; inspect it instead of replacing: {queue}")
        for name in ("states", "claims", "workers", "additional", "selections", "job_records", "gpu_locks"):
            (queue / name).mkdir(exist_ok=True)
        plan = build_plan(root)
        plan["source_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        plan["source_sha256"] = code_hashes(root)
        plan["supervisor_sha256"] = digest(__file__)
        for job in plan["jobs"]:
            if not Path(job["expected"]["model_path"]).is_dir() or not Path(DATA_ROOT).is_dir():
                raise FileNotFoundError("A required checkpoint or data directory is missing")
            if not job["external"] and Path(job["run_dir"]).exists():
                raise FileExistsError(f'Inspect existing output before queuing: {job["run_dir"]}')
            if job["external"]:
                saved = read_json(Path(job["run_dir"]) / "experiment.json")
                if not saved:
                    raise FileNotFoundError(f'Existing sweep manifest missing: {job["id"]}')
                verify_settings(job, saved["settings"])
            atomic_json(state_path(queue, job), {"id": job["id"], "created_at_utc": utc(),
                        "status": "external_running" if job["external"] else "waiting_gamma" if job["gamma_group"] else "queued"})
        atomic_json(queue / "plan.json", plan)
        reconcile(queue, plan)
        return plan


def claim_next(queue, plan, gpu_total_mib, owner):
    for job in sorted(all_jobs(queue, plan), key=lambda job: (job["priority"], job["id"])):
        if job["external"] or job["minimum_gpu_memory_mib"] > gpu_total_mib:
            continue
        state = read_json(state_path(queue, job), {})
        if state.get("status") not in {"queued", "waiting_gamma"}:
            continue
        resolved = resolve_job(queue, job)
        if resolved is None:
            continue
        try:
            (queue / "claims" / job["id"]).mkdir()
        except FileExistsError:
            continue
        atomic_json(state_path(queue, job), {"id": job["id"], "status": "claimed", **owner,
                    "claimed_at_utc": utc(), "gamma_selection": resolved.get("gamma_selection")})
        return resolved
    return None


def summary(queue, plan):
    jobs = all_jobs(queue, plan)
    states = [read_json(state_path(queue, job), {"status": "missing"}) for job in jobs]
    return {"queue": str(queue), "checked_at_utc": utc(), "counts": dict(Counter(state["status"] for state in states)),
            "total": len(jobs), "initial_counts": plan["counts"],
            "workers": [read_json(path) for path in sorted((queue / "workers").glob("*.json"))],
            "gamma_groups": [read_json(selection_path(queue, group["id"]), {}) for group in plan["groups"]]}


def gpu_inventory(uuid):
    output = subprocess.check_output(["nvidia-smi", f"--id={uuid}",
              "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
              "--format=csv,noheader,nounits"], text=True, timeout=20)
    index, uuid, name, used, total, utilization = [item.strip() for item in output.strip().split(",")]
    return {"index": int(index), "uuid": uuid, "name": name, "used_mib": int(used),
            "total_mib": int(total), "utilization_percent": int(utilization)}


def legacy_busy(plan):
    for job in plan["jobs"]:
        if not job["external"]:
            continue
        launch = read_json(job["legacy_launch"], {})
        if launch.get("host", "").split(".")[0] != socket.gethostname().split(".")[0]:
            continue
        pid = launch.get("pid")
        command = Path(f"/proc/{pid}/cmdline")
        if command.exists() and job["run_dir"].encode() in command.read_bytes():
            return True
    return False


def source_changes(plan):
    root = Path(plan["root"])
    return [name for name, expected in plan["source_sha256"].items()
            if not (root / name).is_file() or digest(root / name) != expected]


def prune_ppq_caches(job):
    if job["method"] != "ppq-layerwise":
        return []
    directory = Path(job["run_dir"])
    history = read_json(directory / "history.json", [])
    completed = list(dict.fromkeys(row["unit"] for row in history if row["epoch"] == 5))
    removed = []
    for index, unit in enumerate(completed[:-1]):
        cache = directory / "work" / f"ppq-unit-{index}"
        manifest = cache / "manifest.json"
        if cache.is_symlink() or not manifest.exists():
            continue
        if read_json(manifest).get("identity", {}).get("unit") == unit:
            shutil.rmtree(cache)
            removed.append({"index": index, "unit": unit})
    return removed


def evaluation_complete(job):
    result = read_json(job["eval_output"])
    if result is None:
        return False
    values = result.get("metrics", {})
    if not values or not all("origin" in value and "compressed" in value for value in values.values()):
        raise ValueError("Evaluation result lacks original/compressed metric pairs")
    return True


def run_job(queue, plan, job, owner, heartbeat):
    root = Path(plan["root"])
    state = {"id": job["id"], "status": "running", "started_at_utc": utc(), **owner,
             "gamma_selection": job.get("gamma_selection"), "pruned_layer_caches": []}
    record = {"job": job, "owner": owner, "source_revision": plan["source_revision"],
              "source_sha256": plan["source_sha256"], "queue_plan_sha256": digest(queue / "plan.json"),
              "queue_supervisor_sha256": digest(__file__), "stages": []}
    record_path = queue / "job_records" / f'{job["id"]}.json'
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=owner["gpu_uuid"], PYTHONUNBUFFERED="1")
    child = None

    def persist():
        state["checked_at_utc"] = utc()
        atomic_json(state_path(queue, job), state)
        atomic_json(record_path, record)
        heartbeat(status="running", job_id=job["id"], stage=state.get("stage"), child_pid=state.get("pid"))

    def execute(command, log_path, stage):
        nonlocal child
        changed = source_changes(plan)
        if changed:
            raise ValueError(f"Frozen quantization/evaluation sources changed: {changed}")
        while True:
            gpu = gpu_inventory(owner["gpu_uuid"])
            if gpu["used_mib"] < plan["maximum_idle_gpu_memory_mib"]:
                break
            state.update(stage=f"waiting_gpu_before_{stage}", gpu=gpu)
            persist()
            time.sleep(plan["poll_seconds"])
        changed = source_changes(plan)
        if changed:
            raise ValueError(f"Sources changed while waiting for GPU: {changed}")
        command = [sys.executable if item == "{python}" else item for item in command]
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        event = {"stage": stage, "command": command, "started_at_utc": utc(), "log": log_path}
        record["stages"].append(event)
        with Path(log_path).open("a") as log:
            log.write("\nQUEUE_STAGE " + json.dumps(event) + "\n")
            log.flush()
            child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            event["pid"] = child.pid
            state.update(stage=stage, pid=child.pid, command=command)
            while child.poll() is None:
                if stage == "quantization":
                    try:
                        state["pruned_layer_caches"].extend(prune_ppq_caches(job))
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        state["housekeeping_error"] = str(error)
                persist()
                time.sleep(plan["poll_seconds"])
            event.update(returncode=child.returncode, ended_at_utc=utc())
            persist()
            if child.returncode != 0:
                raise RuntimeError(f"{stage} exited with {child.returncode}; see {log_path}")
            child = None

    try:
        persist()
        artifact = completed_artifact(job)
        if artifact is None:
            command = job["quantize"]
            experiment = read_json(Path(job["run_dir"]) / "experiment.json")
            if experiment is not None:
                verify_settings(job, experiment["settings"])
                command = ["{python}", "-u", "-m", MODULES[job["method"]], "--resume", job["run_dir"],
                           "--save_every", str(job["save_every"])]
            elif (Path(job["run_dir"]) / "compressed_model.pt").exists():
                raise FileExistsError("Unverified existing artifact; inspect before continuing")
            execute(command, job["log"], "quantization")
            artifact = completed_artifact(job)
            if artifact is None:
                raise RuntimeError("Quantization exited without a completed model/metadata record")
        state["artifact"] = artifact
        if job["evaluate"]:
            if Path(job["eval_output"]).exists():
                raise FileExistsError("An unregistered evaluation file already exists; inspect its protocol")
            atomic_json(Path(job["eval_output"]).with_suffix(".protocol.json"),
                        {"protocol": plan["evaluation_protocol"], "command": job["evaluate"],
                         "job_id": job["id"], "source_revision": plan["source_revision"]})
            execute(job["evaluate"], job["eval_log"], "evaluation")
            if not evaluation_complete(job):
                raise RuntimeError("Evaluation exited without its JSON result")
        state.update(status="completed", ended_at_utc=utc(), stage="complete")
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=30)
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     ended_at_utc=utc(), error=f"{type(error).__name__}: {error}")
        persist()
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        print(f'{utc()} FAILED {job["id"]}: {error}', flush=True)
    finally:
        persist()


def worker(queue, uuid, session):
    plan = read_json(queue / "plan.json")
    if plan is None:
        raise FileNotFoundError("Initialize the queue first")
    owner = {"host": socket.gethostname(), "gpu_uuid": uuid, "tmux_session": session,
             "worker_pid": os.getpid()}
    worker_file = queue / "workers" / f'{owner["host"]}-{uuid}.json'
    with (queue / "gpu_locks" / f"{uuid}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def heartbeat(**values):
            atomic_json(worker_file, {**owner, "checked_at_utc": utc(), **values})

        def stopped(*_):
            raise KeyboardInterrupt("Worker stopped")

        signal.signal(signal.SIGTERM, stopped)
        print(f"{utc()} worker {owner} waiting on {queue}", flush=True)
        try:
            while True:
                try:
                    reconcile(queue, plan)
                    gpu = gpu_inventory(uuid)
                    if legacy_busy(plan) or gpu["used_mib"] >= plan["maximum_idle_gpu_memory_mib"]:
                        heartbeat(status="waiting_gpu", gpu=gpu)
                    elif shutil.disk_usage(plan["root"]).free < plan["minimum_free_disk_gib"] * 1024 ** 3:
                        heartbeat(status="waiting_disk_space", gpu=gpu)
                    else:
                        changed = source_changes(plan)
                        if changed:
                            heartbeat(status="source_changed", changed_files=changed)
                        else:
                            job = claim_next(queue, plan, gpu["total_mib"], owner)
                            if job is not None:
                                print(f'{utc()} claimed {job["id"]}', flush=True)
                                run_job(queue, plan, job, owner, heartbeat)
                                continue
                            counts = summary(queue, plan)["counts"]
                            if sum(count for status, count in counts.items() if status not in TERMINAL) == 0:
                                heartbeat(status="finished", job_counts=counts)
                                return
                            heartbeat(status="waiting_dependencies", gpu=gpu, job_counts=counts)
                except Exception as error:
                    heartbeat(status="worker_error", error=f"{type(error).__name__}: {error}")
                    print(f"{utc()} worker check: {error}", flush=True)
                time.sleep(plan["poll_seconds"])
        except KeyboardInterrupt:
            heartbeat(status="stopped")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "init", "worker", "status"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--queue", type=Path, default=SERVER_ROOT / "logs/poseidon/queues/sbpq-led-v2-20261119")
    parser.add_argument("--gpu_uuid")
    parser.add_argument("--session", default="poseidon-v2-wait")
    args = parser.parse_args()
    if args.action == "plan":
        result = build_plan(args.root)
    elif args.action == "init":
        result = initialize(args.queue, args.root)
    elif args.action == "worker":
        if not args.gpu_uuid:
            parser.error("worker requires --gpu_uuid")
        worker(args.queue, args.gpu_uuid, args.session)
        return
    else:
        result = summary(args.queue, read_json(args.queue / "plan.json"))
    if args.action in {"plan", "init"}:
        result = {"queue": str(args.queue), "counts": result["counts"], "total": len(result["jobs"]),
                  "gamma_groups": result["groups"]}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
