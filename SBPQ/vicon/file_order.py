# Adapted from SafeSVD/VICON; see VENDORED.md and LICENSE-*.txt in this directory.
"""Replay a recorded VICON file enumeration under an independent data root.

The research loader shuffles the filesystem's initial order rather than a
sorted order. Copying the same files can therefore change its sample stream.
Manifests contain paths relative to each dataset folder, never source paths.
They establish ordering only; they do not certify equal HDF5 contents.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate VICON file-order key: {key!r}")
        result[key] = value
    return result


def _relative_name(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"VICON file-order entries must be relative POSIX paths: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value or value == ".":
        raise ValueError(f"VICON file-order entries must be canonical relative paths: {value!r}")
    return value


def apply_vicon_file_order(datasets, manifest_path, split="train"):
    """Apply a version-1 manifest before iteration and return its SHA-256.

    Schema: ``{"version": 1, "splits": {"train": {"NS2D": ["a_train.h5"]}}}``.
    Categories must match the loader's category order exactly. Each file list
    must cover precisely the existing ``dataset.dp`` entries relative to
    ``dataset.cfg.types[category].folder``. All datasets are validated before
    any path order is changed, and existing local path strings are preserved.
    """
    raw = Path(manifest_path).read_bytes()
    manifest = json.loads(raw, object_pairs_hook=_unique_keys)
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("Unsupported VICON file-order manifest version; expected 1")
    splits = manifest.get("splits")
    orders = splits.get(split) if isinstance(splits, dict) else None
    if not isinstance(orders, dict):
        raise ValueError(f"VICON file-order manifest has no {split!r} split")
    if list(orders) != list(datasets):
        raise ValueError(
            "VICON file-order categories/order do not match the configured datasets: "
            f"manifest={list(orders)}, configured={list(datasets)}"
        )

    replacements = []
    for category, dataset in datasets.items():
        if getattr(dataset, "rng", None) is not None or getattr(dataset, "tc_rng", None) is not None:
            raise ValueError(f"VICON file order for {category} must be set before dataset iteration")
        names = orders[category]
        if not isinstance(names, list):
            raise ValueError(f"VICON file-order list is invalid for {category}")
        names = [_relative_name(name) for name in names]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate VICON file-order paths for {category}")
        try:
            folder = Path(os.path.abspath(dataset.cfg.types[category].folder))
            paths = dataset.dp
        except (AttributeError, KeyError, TypeError) as error:
            raise ValueError(f"VICON dataset {category} does not expose its folder and file paths") from error
        local = {}
        for path in paths:
            # Keep the logical local path, including symlinks; resolving a link
            # could unnecessarily replace the user's configured dataset root.
            absolute = Path(os.path.abspath(path))
            try:
                name = absolute.relative_to(folder).as_posix()
            except ValueError as error:
                raise ValueError(f"VICON file for {category} is outside its configured folder: {path}") from error
            if name in local:
                raise ValueError(f"Duplicate local VICON file for {category}: {name}")
            if not absolute.is_file():
                raise FileNotFoundError(f"VICON file for {category} does not exist: {path}")
            local[name] = path
        if set(names) != set(local):
            missing = sorted(set(names) - set(local))
            extra = sorted(set(local) - set(names))
            raise ValueError(
                f"VICON file-order set mismatch for {category}: "
                f"missing={missing}, extra={extra}"
            )
        replacements.append((dataset, [local[name] for name in names]))

    for dataset, paths in replacements:
        dataset.dp = paths
    return hashlib.sha256(raw).hexdigest()
