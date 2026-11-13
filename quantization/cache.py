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


class PersistentCalibration:
    """Content-verified records, reusable after interruption without FP recomputation.

    Source batches are always read and hashed, so changed data cannot silently
    reuse old FP targets. Derived records are reused only after their payload
    hash is verified. A partial manifest commits a bounded preparation prefix.
    """
    def __init__(self, loader, directory, identity, *, transform=None, build_record=None):
        from .checkpoints import atomic_json, atomic_save, cpu_tree, fingerprint
        from .costs import counter
        import json
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest_path = self.directory / 'manifest.json'
        saved = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        if saved and saved['identity'] != identity:
            raise ValueError('Calibration cache model/settings identity changed')
        entries = saved['entries'] if saved else []
        self.paths, self.samples, self.batch_sizes = [], 0, []
        digests = []
        for index, source in enumerate(progress(loader, desc='Verify/cache calibration')):
            batch = cpu_tree(transform(source) if transform else source)
            value = batch if torch.is_tensor(batch) else batch.get('pixel_values')
            if value is None or not len(value):
                raise ValueError('Calibration requires a nonempty batched input')
            digest = fingerprint(batch)
            digests.append(digest)
            path = self.directory / f'batch-{index:06d}.pt'
            if index < len(entries):
                entry = entries[index]
                if entry['input'] != digest:
                    raise ValueError(f'Calibration data changed at batch {index}')
                if path.exists():
                    record = torch.load(path, map_location='cpu', weights_only=True)
                    if fingerprint(record) != entry['payload']:
                        raise ValueError(f'Calibration cache is corrupt: {path}')
                    counter('calibration_records_reused')
                else:
                    record = {'batch': batch, **cpu_tree(build_record(batch) if build_record else {})}
                    if fingerprint(record) != entry['payload']:
                        raise ValueError('Rebuilt calibration target differs from the saved cache')
                    atomic_save(record, path)
                    counter('cache_bytes_written', path.stat().st_size)
            else:
                if saved and saved.get('complete'):
                    raise ValueError('Calibration batch count increased')
                record = {'batch': batch, **cpu_tree(build_record(batch) if build_record else {})}
                atomic_save(record, path)
                counter('cache_bytes_written', path.stat().st_size)
                entries.append({'input': digest, 'payload': fingerprint(record), 'samples': len(value)})
                counter('calibration_records_written')
                if (index + 1) % 32 == 0:
                    atomic_json({'identity': identity, 'entries': entries, 'complete': False}, manifest_path)
            self.paths.append(path)
            self.batch_sizes.append(len(value))
            self.samples += len(value)
        if not self.paths or (saved and len(self.paths) < len(entries)):
            raise ValueError('Calibration batch count decreased or is empty')
        self.digest = fingerprint(digests)
        atomic_json({'identity': identity, 'entries': entries, 'complete': True,
                     'digest': self.digest, 'samples': self.samples}, manifest_path)

    def __len__(self):
        return len(self.paths)

    def __iter__(self):
        for path in self.paths:
            yield torch.load(path, map_location='cpu', weights_only=True)['batch']
