# Adapted from SafeSVD/VICON; see VENDORED.md and LICENSE-*.txt in this directory.
"""Seeded calibration-dataset helpers for VICON MySVD."""

from __future__ import annotations

import copy
import hashlib
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, Subset, get_worker_info


def _stream_seed(seed: int, category: str, worker: int) -> int:
    # Python's hash() is salted independently in each process.
    key = f"vicon-calibration-v1:{seed}:{category}:{worker}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63)


class _RepeatableViconStream(IterableDataset):
    """Reset only calibration streams; leave the upstream training loader alone.

    Each iterator owns a shallow copy with fresh RNGs and paths. This also
    prevents a partially consumed profiling pass from advancing Fisher or
    compression, and works with worker processes and concurrent iterators.
    """

    def __init__(self, dataset, category: str, seed: int):
        for name in ("dp", "rng", "tc_rng"):
            if not hasattr(dataset, name):
                raise TypeError(f"VICON calibration stream {category} lacks {name}")
        self.dataset = dataset
        self.category = category
        self.seed = seed
        # Sort logical relative paths, not symlink targets or filesystem order.
        folder = dataset.cfg.types[category].folder
        self.dp = sorted(dataset.dp, key=lambda path: os.path.relpath(path, folder))

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        seed = _stream_seed(self.seed, self.category, worker_id)
        dataset = copy.copy(self.dataset)
        dataset.dp = list(self.dp)
        dataset.num_workers = worker.num_workers if worker is not None else 0
        dataset.worker_id = worker_id
        dataset.rng = np.random.default_rng(seed)
        dataset.tc_rng = torch.Generator().manual_seed(seed)
        # Upstream _init_rng returns immediately because rng is initialized.
        yield from dataset


def calibration_cache_tag(
    calibration_num: int,
    random_calibration: bool = False,
    calibration_seed: int = 0,
) -> str:
    """Keep caches produced from different calibration samples isolated."""
    tag = f"cali{calibration_num}"
    if random_calibration:
        tag += f"_random_seed{calibration_seed}"
    return tag


def build_calibration_loaders(
    datasets: dict,
    *,
    calibration_num: int,
    batch_size: int,
    num_workers: int,
    random_calibration: bool = False,
    calibration_seed: int = 0,
) -> dict:
    """
    Build repeatable calibration loaders, independently of prior traversals.

    For map-style datasets, the subset is materialized once and iterated without
    shuffling. Consequently, profiling, Fisher collection, rank selection, and
    compression all see exactly the same calibration examples.

    VICON streams always use calibration_seed, including random_calibration=False.
    Every traversal restarts the same category/worker RNGs and sorted file list.
    Keep batch size and worker count fixed when comparing calibration batches.
    Changing the sampling policy requires fresh input/Fisher/rank caches.
    """
    if calibration_num <= 0:
        raise ValueError("calibration_num must be positive")
    if not isinstance(calibration_seed, int) or calibration_seed < 0:
        raise ValueError("calibration_seed must be a non-negative integer")

    rng = random.Random(calibration_seed)
    loaders = {}
    for dataset_type, dataset in datasets.items():
        calibration_dataset = dataset
        if isinstance(dataset, IterableDataset):
            calibration_dataset = _RepeatableViconStream(dataset, dataset_type, calibration_seed)
        elif random_calibration:
            try:
                dataset_size = len(dataset)
            except (AttributeError, NotImplementedError, TypeError):
                dataset_size = None

            if dataset_size is not None:
                sample_count = min(calibration_num, dataset_size)
                indices = rng.sample(range(dataset_size), sample_count)
                calibration_dataset = Subset(dataset, indices)

        loaders[dataset_type] = DataLoader(
            calibration_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            generator=torch.Generator().manual_seed(_stream_seed(calibration_seed, dataset_type, -1)),
        )
    return loaders


def describe_calibration_selection(
    datasets: dict,
    *,
    calibration_num: int,
    random_calibration: bool = False,
    calibration_seed: int = 0,
) -> str:
    counts = {}
    stream_datasets = []
    for dataset_type, dataset in datasets.items():
        try:
            counts[dataset_type] = min(calibration_num, len(dataset))
        except (AttributeError, NotImplementedError, TypeError):
            counts[dataset_type] = "stream"
            stream_datasets.append(dataset_type)

    if stream_datasets:
        return (
            "repeatable VICON calibration streams (sampling=v1, sorted files, "
            f"reset each pass, seed={calibration_seed}, stream={stream_datasets}, "
            f"map-style random_subset={random_calibration}, counts={counts})"
        )
    if random_calibration:
        return f"seeded random subsets (seed={calibration_seed}, counts={counts})"
    return f"fixed dataset prefixes (counts={counts})"
