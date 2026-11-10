from argparse import Namespace
import importlib.util
from pathlib import Path
import random

import pytest
import torch
from torch.utils.data import Dataset

from quantization.calibration import TrajectorySubset, build_calibration_loader


class ToyTrajectories(Dataset):
    def __init__(self, dataset, which, num_trajectories, data_path, **kwargs):
        self.num_trajectories = 10 if num_trajectories == -1 else num_trajectories
        self.kwargs = kwargs
        self.which = which
        self.data_path = data_path
        self.constants = {"mean": torch.tensor([0.0]), "std": torch.tensor([1.0]), "time": 20}
        self.resolution = 4
        self.label_description = "[u]"
        self.start = 0
        if "elliptic" not in dataset:
            n = kwargs["max_num_time_steps"]
            self.time_indices = [(i, j) for i in range(n+1) for j in range(i, n+1)]
        self.multiplier = len(getattr(self, "time_indices", [None]))

    def __len__(self):
        return self.num_trajectories * self.multiplier

    def __getitem__(self, idx):
        return {"index": idx, "trajectory": idx // self.multiplier}


def args(**kwargs):
    values = dict(dataset="fluids.incompressible.PiecewiseConstants", traj_num=3,
                  random_calibration="n", calibration_seed=42, final_time=4,
                  time_back=1, data_path="dataset", batch_size=7, num_workers=0)
    return Namespace(**(values | kwargs))


def test_complete_prefix_all_pairs_and_no_batch_truncation():
    dataset, loader, metadata = build_calibration_loader(args(), ToyTrajectories)
    assert dataset.kwargs == dict(fix_input_to_time_step=None, time_step_size=1, max_num_time_steps=3)
    assert metadata["time_pairs"] == [[i, j] for i in range(4) for j in range(i, 4)]
    assert metadata["num_samples"] == 30
    assert metadata["trajectory_indices"] == [0, 1, 2]
    assert torch.cat([batch["index"] for batch in loader]).tolist() == list(range(30))


def test_random_matches_reference_algorithm_and_preserves_rng():
    random.seed(123)
    before = random.getstate()
    dataset, loader, metadata = build_calibration_loader(args(random_calibration="y"), ToyTrajectories)
    selected = random.Random(42).sample(range(10), 3)
    assert metadata["trajectory_indices"] == selected
    assert list(dataset.sample_indices) == [i*10+j for i in selected for j in range(10)]
    assert torch.cat([b["index"] for b in loader]).tolist() == list(dataset.sample_indices)
    assert random.getstate() == before


def test_direct_reference_helper_equivalence():
    reference = Path("/Users/hcj/UCL/Final Year Project/Scientific-Foundational-Model-compression/physic_model_comp/poseidon/MySVD/helper/calibration_dataset.py")
    if not reference.exists():
        pytest.skip("Optional local MySVD reference is unavailable.")
    spec = importlib.util.spec_from_file_location("mysvd_calibration_reference", reference)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for randomized in (False, True):
        ours, _, metadata = build_calibration_loader(args(random_calibration="y" if randomized else "n"), ToyTrajectories)
        theirs = module.build_calibration_dataset(
            ToyTrajectories, dataset_name=args().dataset, num_trajectories=3,
            random_calibration=randomized, calibration_seed=42, data_path="dataset",
            fix_input_to_time_step=None, time_step_size=1, max_num_time_steps=3,
        )
        assert [ours[i] for i in range(len(ours))] == [theirs[i] for i in range(len(theirs))]
        assert ours.constants == theirs.constants


def test_elliptic_has_no_time_kwargs():
    dataset, _, metadata = build_calibration_loader(args(dataset="elliptic.poisson.Gaussians"), ToyTrajectories)
    assert dataset.kwargs == {}
    assert len(dataset) == 3
    assert metadata["time_pairs"] is None


def test_fingerprint_tracks_selection_horizon_normalization_and_source_file(tmp_path):
    data = tmp_path / "data.nc"
    data.write_bytes(b"example dataset")
    def factory(**kwargs):
        ds = ToyTrajectories(**kwargs)
        ds.file_path = str(data)
        return ds
    base = args(random_calibration="y", data_path=str(tmp_path))
    first = build_calibration_loader(base, factory)[2]["fingerprint"]
    assert first == build_calibration_loader(base, factory)[2]["fingerprint"]
    assert first != build_calibration_loader(args(**(vars(base) | {"calibration_seed": 5})), factory)[2]["fingerprint"]
    assert first != build_calibration_loader(args(**(vars(base) | {"time_back": 0})), factory)[2]["fingerprint"]
    def normalized_factory(**kwargs):
        ds = factory(**kwargs)
        ds.constants["std"] = torch.tensor([2.0])
        return ds
    assert first != build_calibration_loader(base, normalized_factory)[2]["fingerprint"]
    data.write_bytes(b"changed dataset")
    assert first != build_calibration_loader(base, factory)[2]["fingerprint"]


def test_invalid_selections_fail():
    with pytest.raises(ValueError):
        build_calibration_loader(args(traj_num=0), ToyTrajectories)
    with pytest.raises(ValueError):
        build_calibration_loader(args(traj_num=11, random_calibration="y"), ToyTrajectories)
    with pytest.raises(ValueError):
        build_calibration_loader(args(time_back=4), ToyTrajectories)
    with pytest.raises(ValueError):
        TrajectorySubset(ToyTrajectories(**dict(dataset="elliptic", which="train", num_trajectories=2, data_path="")), [0, 0])
