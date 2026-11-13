"""CPU batch snapshots that support cyclic optimization without retaining RAM copies."""
from collections.abc import Sequence
from pathlib import Path
import torch


class DiskSequence(Sequence):
    def __init__(self, paths, key=None):
        self.paths = tuple(Path(path) for path in paths)
        self.key = key

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return type(self)(self.paths[index], key=self.key)
        value = torch.load(self.paths[index], map_location="cpu", weights_only=True)
        return value[self.key] if self.key else value
