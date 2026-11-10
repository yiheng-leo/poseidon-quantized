"""Consistent terminal and batch-job progress for quantization workflows."""
from __future__ import annotations

import os
import sys

from tqdm.auto import tqdm


def progress(iterable=None, *, desc, unit="batch", total=None, leave=True,
             enabled=True, **kwargs):
    """Report completed work without consuming or reordering the iterable.

    Keep bars enabled in redirected Slurm logs too. Set TQDM_DISABLE=1 to silence
    them; throttled refreshes keep nested calibration/MC loops inexpensive.
    Unknown-length calibration iterators show counts and speed without an ETA.
    """
    disabled = not enabled or os.environ.get("TQDM_DISABLE", "").lower() in {"1", "true", "yes", "on"}
    return tqdm(iterable, desc=desc, unit=unit, total=total, leave=leave,
                dynamic_ncols=True, mininterval=1.0, disable=disabled, **kwargs)


def progress_write(message):
    """Print a durable summary without overwriting a live progress bar."""
    tqdm.write(message)
    sys.stdout.flush()
