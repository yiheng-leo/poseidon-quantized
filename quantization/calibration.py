"""Deterministic, complete-trajectory calibration compatible with MySVD.

TrajectorySubset and the selection policy are adapted from
physic_model_comp/poseidon/MySVD/helper/calibration_dataset.py (local reference
read 2026-10-23). Time settings follow MySVD/mysvd.py, not PPQ's truncated
minibatch loaders. Source repository is not required at runtime.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


def get_dataset(*args, **kwargs):
    # Delay optional ScOT/accelerate dependencies until a real dataset is needed.
    from scOT.problems.base import get_dataset as factory
    return factory(*args, **kwargs)


class TrajectorySubset(Dataset):
    """Keep every sample, in source order, of each selected trajectory."""

    def __init__(self, dataset, trajectory_indices):
        self.source_dataset = dataset
        self.selected_trajectory_indices = tuple(trajectory_indices)
        self.num_trajectories = len(self.selected_trajectory_indices)
        total = int(dataset.num_trajectories)
        if total <= 0 or len(dataset) % total:
            raise ValueError("Dataset must contain equally sized, complete trajectories.")
        if len(set(self.selected_trajectory_indices)) != self.num_trajectories:
            raise ValueError("trajectory_indices must not contain duplicates.")
        if any(i < 0 or i >= total for i in self.selected_trajectory_indices):
            raise IndexError("trajectory_indices outside the source dataset.")
        self.samples_per_trajectory = len(dataset) // total
        self.sample_indices = tuple(
            i * self.samples_per_trajectory + offset
            for i in self.selected_trajectory_indices
            for offset in range(self.samples_per_trajectory)
        )

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        return self.source_dataset[self.sample_indices[idx]]

    def __getattr__(self, name):
        if name == "source_dataset":
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "source_dataset"), name)


def _jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported metadata type: {type(value).__name__}")


def _fingerprint(metadata):
    encoded = json.dumps(_jsonable(metadata), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _underlying_dataset(dataset):
    while hasattr(dataset, "source_dataset") or hasattr(dataset, "dataset"):
        dataset = dataset.source_dataset if hasattr(dataset, "source_dataset") else dataset.dataset
    return dataset


def dataset_metadata(dataset):
    """Describe normalization and source identity without hashing multi-GB data.

    Cache identity includes source-code SHA256, normalization, file identity,
    nanosecond mtime/ctime and 64 KiB content probes (head/middle/tail). This is a
    local cache guard, not a cryptographic checksum of the entire data archive.
    """
    source = _underlying_dataset(dataset)
    info = {
        "dataset_class": f"{type(source).__module__}.{type(source).__qualname__}",
        "normalization": _jsonable(getattr(source, "constants", {})),
        "normalization_attributes": {
            key: _jsonable(getattr(source, key))
            for key in ("mean", "std", "mean_pressure") if hasattr(source, key)
        },
        "resolution": _jsonable(getattr(source, "resolution", None)),
        "transpose": bool(getattr(source, "transpose", False)),
        "label_description": getattr(source, "label_description", None),
        "pixel_mask": _jsonable(getattr(source, "pixel_mask", None)),
        "split_start": int(getattr(source, "start", 0)),
    }
    source_files = set()
    for cls in type(source).__mro__:
        try:
            filename = inspect.getsourcefile(cls)
        except TypeError:
            filename = None
        if filename and Path(filename).is_file():
            source_files.add(Path(filename).resolve())
    info["implementation_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_files)
    }
    file_paths = set()
    reader = getattr(source, "reader", None)
    for filename in (getattr(reader, "filename", None), getattr(source, "file_path", None)):
        if isinstance(filename, (str, Path)) and Path(filename).is_file():
            file_paths.add(Path(filename).resolve())
    info["data_files"] = []
    for path in sorted(file_paths):
        st = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for offset in sorted({0, max(0, st.st_size // 2 - 32768), max(0, st.st_size - 65536)}):
                stream.seek(offset)
                digest.update(str(offset).encode())
                digest.update(stream.read(65536))
        info["data_files"].append({
            "path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns,
            "ctime_ns": st.st_ctime_ns, "inode": st.st_ino, "device": st.st_dev,
            "content_probes_sha256": digest.hexdigest(),
        })
    return info


def build_calibration_loader(args, dataset_factory=None):
    """Return (dataset, sequential full loader, cache-safe selection metadata)."""
    count = int(getattr(args, "traj_num", 64))
    seed = int(getattr(args, "calibration_seed", 42))
    random_flag = getattr(args, "random_calibration", "n")
    if random_flag not in ("y", "n"):
        raise ValueError("random_calibration must be 'y' or 'n'.")
    random_selection = random_flag == "y"
    if count <= 0:
        raise ValueError("traj_num must be positive.")
    final_time = int(getattr(args, "final_time", 20))
    time_back = int(getattr(args, "time_back", 0))
    horizon = final_time - time_back
    elliptic = "elliptic" in args.dataset
    if not elliptic and (time_back < 0 or horizon <= 0):
        raise ValueError("Require 0 <= time_back < final_time.")
    kwargs = {"data_path": str(args.data_path)}
    if not elliptic:
        kwargs.update(fix_input_to_time_step=None, time_step_size=1, max_num_time_steps=horizon)
    factory = dataset_factory or get_dataset
    source = factory(
        dataset=args.dataset, which="train",
        num_trajectories=-1 if random_selection else count, **kwargs,
    )
    total = int(source.num_trajectories)
    if count > total:
        raise ValueError(f"Cannot sample {count} trajectories from {total} training trajectories.")
    selected = random.Random(seed).sample(range(total), count) if random_selection else list(range(count))
    dataset = TrajectorySubset(source, selected) if random_selection else source
    if len(dataset) % count:
        raise ValueError("Calibration data do not contain complete trajectories.")
    metadata = {
        "schema": "mysvd-complete-trajectory-calibration-v1",
        "dataset": args.dataset, "split": "train", "data_path": str(Path(args.data_path).resolve()),
        "selection": "seeded_random_without_replacement" if random_selection else "fixed_prefix",
        "calibration_seed": seed, "num_trajectories": count,
        "trajectory_indices": selected, "num_samples": len(dataset),
        "absolute_trajectory_indices": [int(getattr(source, "start", 0)) + i for i in selected],
        "samples_per_trajectory": len(dataset) // count,
        "final_time": final_time, "time_back": time_back,
        "time_pairs": _jsonable(getattr(source, "time_indices", None)) if not elliptic else None,
        "time_settings": {k: v for k, v in kwargs.items() if k != "data_path"},
        "loader": {"batch_size": args.batch_size, "num_workers": args.num_workers,
                   "shuffle": False, "drop_last": False},
        "selection_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source": dataset_metadata(source),
        "reference": "MySVD/helper/calibration_dataset.py and MySVD/mysvd.py (2026-10-23)",
    }
    metadata["fingerprint"] = _fingerprint(metadata)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                        shuffle=False, drop_last=False)
    return dataset, loader, metadata
