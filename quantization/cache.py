"""Fresh, complete calibration snapshots; no fixed-batch truncation or stale cache reuse."""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import tempfile
import torch
from .common import move_to_device
from .progress import progress

class CalibrationCache:
    """A fresh disk snapshot of every supplied calibration batch, never an old cache."""

    def __init__(self, loader, directory):
        self.paths = []
        self.samples = 0
        if loader is None:
            raise ValueError("This method requires a nonempty calibration loader")
        for index, batch in enumerate(progress(loader, desc="Cache calibration")):
            batch = move_to_device(batch, "cpu")
            value = batch if torch.is_tensor(batch) else batch.get("pixel_values")
            if not torch.is_tensor(value) or value.ndim == 0:
                raise ValueError("Calibration batches need a batched pixel_values tensor")
            path = Path(directory) / f"calibration-{index:06d}.pt"
            torch.save(batch, path)
            self.paths.append(path)
            self.samples += value.shape[0]
        if not self.paths or not self.samples:
            raise ValueError("Calibration loader is empty")

    def __iter__(self):
        for path in self.paths:
            yield torch.load(path, map_location="cpu", weights_only=False)

    def __len__(self):
        return len(self.paths)


@contextmanager
def calibration_cache(loader, work_dir, *, required=True):
    if not required:
        yield None, None
        return
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="baseline-calibration-", dir=work_dir) as directory:
        yield CalibrationCache(loader, directory), directory


def record_calibration(metadata, calibration):
    if calibration is not None:
        metadata.update(calibration_samples=calibration.samples,
                        calibration_batches=len(calibration.paths))
