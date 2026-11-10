"""Atomic training checkpoints and deterministic resume bookkeeping."""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def fingerprint(value):
    """Hash batch order, tensor shapes/dtypes and bytes without retaining copies."""
    digest = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(f"tensor:{tensor.dtype}:{tuple(tensor.shape)}".encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict{")
            for key in sorted(item):
                visit(key)
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            digest.update(f"sequence:{len(item)}[".encode())
            for entry in item:
                visit(entry)
            digest.update(b"]")
        else:
            digest.update(f"{type(item).__name__}:{item!r};".encode())

    visit(value)
    return digest.hexdigest()


def rng_state(device):
    numpy_state = np.random.get_state()
    # Store only standard Python types and tensors so existing evaluators can
    # still read trainer states with torch.load(weights_only=True).
    numpy_state = (numpy_state[0], numpy_state[1].tolist(), int(numpy_state[2]),
                   int(numpy_state[3]), float(numpy_state[4]))
    state = {"python": random.getstate(), "numpy": numpy_state,
             "torch_cpu": torch.get_rng_state(), "device_type": device.type}
    if device.type == "cuda":
        state["torch_device"] = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        state["torch_device"] = torch.mps.get_rng_state()
    return state


def restore_rng(state, device):
    if state["device_type"] != device.type:
        raise ValueError("Exact resume requires the same device type as the checkpoint.")
    random.setstate(state["python"])
    name, values, position, has_gaussian, cached_gaussian = state["numpy"]
    np.random.set_state((name, np.asarray(values, dtype=np.uint32), position, has_gaussian, cached_gaussian))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["torch_device"].cpu(), device)
    elif device.type == "mps":
        torch.mps.set_rng_state(state["torch_device"].cpu())


def atomic_save(value, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def resolve_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        periodic = path / "sbpq_checkpoint.pt"
        path = periodic if periodic.is_file() else path / "sbpq_trainer_state.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    return path
