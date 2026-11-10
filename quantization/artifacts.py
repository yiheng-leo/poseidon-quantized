"""Standalone model artifacts, using MySVD's compressed_model.pt convention."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from torch import nn

from .common import QuantizationResult
from . import compat
from .progress import progress


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} into artifact metadata")


def _sha256(path, *, show_progress=False):
    digest = hashlib.sha256()
    with open(path, "rb") as stream, progress(
            total=os.path.getsize(path), desc="Checksum saved model", unit="B",
            unit_scale=True, unit_divisor=1024, enabled=show_progress) as checksum:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            checksum.update(len(chunk))
    return digest.hexdigest()


def _atomic_save(value, destination, *, json_file=False):
    destination = Path(destination)
    fd, temporary = tempfile.mkstemp(prefix=".saving-", dir=destination.parent)
    os.close(fd)
    try:
        if json_file:
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(value, stream, indent=2, default=_json_default, allow_nan=False)
                stream.write("\n")
        else:
            with open(temporary, "wb") as stream:
                torch.save(value, stream)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_quantized_artifact(result: QuantizationResult, path, metadata, *, overwrite=False):
    """Save a self-contained CPU model; metadata is written last as completion record."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model_file = path / "compressed_model.pt"
    if model_file.exists() and not overwrite:
        raise FileExistsError(f"Artifact already exists: {model_file}; use a new --save_path or --overwrite.")
    with progress(total=3, desc="Save quantization artifact", unit="file") as files:
        files.set_postfix(file=model_file.name, refresh=False)
        result.model.eval().cpu()
        _atomic_save(result.model, model_file)
        files.update()
        files.set_postfix(file="quantization_state.pt", refresh=False)
        _atomic_save(result.state, path / "quantization_state.pt")
        files.update()
        combined = {
            **metadata,
            "schema_version": 1,
            "artifact_type": "poseidon_quantization",
            "model_file": model_file.name,
            "model_sha256": _sha256(model_file, show_progress=True),
            "quantization": result.metadata,
        }
        files.set_postfix(file="metadata.json", refresh=False)
        _atomic_save(combined, path / "metadata.json", json_file=True)
        files.update()
    return combined


def load_quantized_artifact(path, device="cpu"):
    """Load a trusted local artifact without its original checkpoint or calibration data."""
    path = Path(path)
    model_file = path / "compressed_model.pt" if path.is_dir() else path
    metadata_file = model_file.parent / "metadata.json"
    metadata = {}
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        expected = metadata.get("model_sha256")
        if expected is not None and _sha256(model_file) != expected:
            raise ValueError(f"Artifact checksum does not match metadata: {model_file}")
    # Full module serialization deliberately follows MySVD. Load only trusted files.
    model = torch.load(model_file, map_location="cpu", weights_only=False, pickle_module=compat)
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected a saved model, got {type(model).__name__}: {model_file}")
    return model.to(device).eval(), metadata
