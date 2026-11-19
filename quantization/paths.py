"""Poseidon experiment paths shared by quantization and evaluation.

The seed label describes calibration sampling, independently of the optimizer RNG.
Numerical settings remain in experiment.json/metadata.json rather than the name.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import re


POSEIDON_METHODS = frozenset({"uniform", "sbpq", "brecq", "ppq", "ppq-layerwise"})
DATASET_LABELS = {
    "fluids.incompressible.PiecewiseConstants": "NS-PwC",
    "fluids.compressible.RichtmyerMeshkov": "CE-RM",
    "wave.Gaussians": "Wave-Gauss",
}
_COMPACT_NAME = re.compile(
    r"(?:uniform|sbpq|brecq|ppq(?:-layerwise)?)-traj\d+-ft\d+"
    r"-bit\d+(?:\.\d+)?-(?:default|seed-?\d+)\Z"
)


def _settings(value):
    return value if isinstance(value, Mapping) else vars(value)


def uses_poseidon_paths(value):
    settings = _settings(value)
    return (settings.get("model_type", "poseidon") == "poseidon"
            and settings.get("method") in POSEIDON_METHODS)


def dataset_label(dataset):
    return DATASET_LABELS.get(dataset, dataset)


def poseidon_run_name(value):
    settings = _settings(value)
    method = settings["method"]
    if not uses_poseidon_paths(settings):
        raise ValueError("Compact experiment names require a supported Poseidon method")
    bits = settings["target_bits"] if method in {"sbpq", "ppq", "ppq-layerwise"} else settings["weight_bits"]
    seed = "default" if settings["random_calibration"] == "n" else f'seed{settings["calibration_seed"]}'
    return (f'{method}-traj{settings["traj_num"]}-ft{settings["final_time"]}'
            f'-bit{bits:g}-{seed}')


def default_artifact_path(value):
    settings = _settings(value)
    return Path("quantization_artifacts") / dataset_label(settings["dataset"]) / poseidon_run_name(settings)


def default_evaluation_path(model_path, dataset):
    """Infer calibration labels from the artifact, including legacy run names.

    Original checkpoints and artifacts from other model families retain their
    existing output behavior when no experiment configuration can be inferred.
    """
    if not model_path:
        return None
    directory = Path(model_path).expanduser()
    if directory.is_file() or directory.suffix in {".pt", ".pth"}:
        directory = directory.parent
    settings = {}
    for name in ("metadata.json", "experiment.json"):
        manifest = directory / name
        if manifest.is_file():
            record = json.loads(manifest.read_text(encoding="utf-8"))
            settings.update(record.get("settings", {}))
            if record.get("method"):
                settings.setdefault("method", record["method"])
    if settings and not uses_poseidon_paths(settings):
        return None
    required = {"method", "traj_num", "final_time", "random_calibration"}
    bits_key = "target_bits" if settings.get("method") in {"sbpq", "ppq", "ppq-layerwise"} else "weight_bits"
    required.add(bits_key)
    if settings.get("random_calibration") != "n":
        required.add("calibration_seed")
    if required <= settings.keys():
        name = poseidon_run_name(settings)
    elif _COMPACT_NAME.fullmatch(directory.name):
        name = directory.name
    else:
        return None
    return Path("quantization_evaluation") / dataset_label(dataset) / (name + ".json")
