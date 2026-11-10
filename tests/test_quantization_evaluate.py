from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from evaluation import evaluate, metrics, rollout


class ToyDataset(Dataset):
    constants = {"mean": 0.0, "std": 1.0, "time": 1.0}
    label_description = "[u]"
    resolution = 4
    transpose = False
    start = 50
    num_trajectories = 100  # Training capacity deliberately differs from split size.

    def __len__(self):
        return 5

    def __getitem__(self, index):
        x = torch.full((1, 4, 4), float(index))
        return {"pixel_values": x, "labels": x+1, "time": torch.tensor(1.0)}


class AddTime(nn.Module):
    def __init__(self, output_channels=None):
        super().__init__()
        self.calls = []
        self.output_channels = output_channels

    def forward(self, pixel_values, time=None, return_dict=True, **kwargs):
        assert "labels" not in kwargs and "pixel_mask" not in kwargs
        self.calls.append(pixel_values.clone())
        channels = self.output_channels or pixel_values.shape[1]
        delta = time[:, None, None, None] if time is not None else 1
        return SimpleNamespace(output=pixel_values[:, :channels]+delta)


def test_autoregressive_no_gt_leak_and_correct_lead_time():
    model = AddTime(output_channels=2)
    batch = {
        "pixel_values": torch.zeros(2, 3, 4, 4),
        "labels": torch.full((2, 2, 4, 4), 999.0),
        "pixel_mask": torch.tensor([[False, True], [False, True]]),
        "time": torch.tensor([6.0, 6.0]),
    }
    batch["pixel_values"][:, 1] = 7
    batch["pixel_values"][:, 2] = 42
    result = rollout.autoregressive_predict(model, batch, device="cpu", ar_steps=3)
    assert torch.all(result[:, 0] == 6)
    assert torch.all(result[:, 1] == 7)
    assert torch.all(model.calls[1][:, 0] == 2)
    assert torch.all(model.calls[2][:, 2] == 42)
    batch["labels"].fill_(-555)
    assert torch.equal(result, rollout.autoregressive_predict(model, batch, device="cpu", ar_steps=3))


def test_sample_weighting_with_partial_final_batch():
    ds = ToyDataset()
    class Zeros(nn.Module):
        def forward(self, pixel_values, **kwargs):
            return torch.zeros_like(pixel_values)
    result = evaluate.evaluate_model(Zeros(), DataLoader(ds, batch_size=2), dataset=ds,
                                     dataset_name="reaction_diffusion.AllenCahn", device="cpu")
    assert result["num_samples"] == 5
    assert result["metrics"]["L1"] == 48  # 16 * mean(1, 2, 3, 4, 5)
    assert result["metrics"]["RelL1"] == 100
    assert result["metrics"]["sobolev_s01"] == 3


def test_tiny_nonzero_target_relative_error_matches_scot_definition():
    target = torch.full((1, 1, 4, 4), 1e-14, dtype=torch.float64)
    values = metrics.per_sample_metrics(torch.zeros_like(target), target, ToyDataset(),
                                       "reaction_diffusion.AllenCahn")
    assert values["RelL1"].item() == 100


def test_known_velocity_divergence_and_curl():
    # y,x convention with transpose=True: u=x, v=y, div=2, curl=0.
    coords = torch.arange(4, dtype=torch.float64) / 4
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    prediction = torch.stack([xx, yy]).unsqueeze(0)
    ds = SimpleNamespace(constants={"mean": [0, 0], "std": [1, 1]},
                         label_description="[u,v]", transpose=True)
    values = metrics.per_sample_metrics(prediction, prediction.clone(), ds, "fluids.incompressible.PiecewiseConstants")
    assert values["continuity"].item() == 2
    assert values["vorticity"].item() == 0
    assert values["sobolev_s01"].item() == 0
    assert values["RelL1"].item() == 0


def test_normalization_wave_and_poisson():
    x = torch.ones(1, 2, 4, 4)
    ds = SimpleNamespace(constants={"mean": 1, "std": 2, "mean_c": 10, "std_c": 20})
    denorm = metrics.denormalize_output(x, ds, "wave.Layer")
    assert denorm[0, 0, 0, 0] == 3
    assert denorm[0, 1, 0, 0] == 30
    poisson = SimpleNamespace(constants={"mean_solution": 4, "std_solution": 5})
    assert torch.all(metrics.denormalize_output(x[:, :1], poisson, "elliptic.poisson.Gaussians") == 9)


def test_normalization_pressure_offset_tracer_forcing_and_wrapper():
    constants = {"mean": [1, 2, 3, 4], "std": [2, 2, 2, 2],
                 "tracer_mean": 10, "tracer_std": 3,
                 "mean_forcing": 20, "std_forcing": 4}
    compressible = SimpleNamespace(constants=constants, mean_pressure=5,
                                   label_description="[rho],[u,v],[p],[tracer]")
    x = torch.ones(1, 5, 4, 4)
    out = metrics.denormalize_output(x, compressible, "fluids.compressible.Riemann.tracer")
    assert out[0, :, 0, 0].tolist() == [3, 4, 5, 11, 1]
    ns = SimpleNamespace(constants=constants, label_description="[rho],[u,v],[p],[tracer]")
    out = metrics.denormalize_output(x, ns, "fluids.incompressible.PiecewiseConstants.tracer")
    assert out[0, 4, 0, 0] == 13
    forcing = SimpleNamespace(constants=constants, label_description="[rho],[u,v],[p],[g]")
    out = metrics.denormalize_output(x, forcing, "fluids.incompressible.forcing.KolmogorovFlow")
    assert out[0, 4, 0, 0] == 24
    wrapped = SimpleNamespace(dataset=SimpleNamespace(mean=2, std=3))
    assert torch.all(metrics.denormalize_output(x[:, :1], wrapped, "elliptic.Helmholtz.time") == 5)


def test_complete_eval_split_and_explicit_prefix(monkeypatch):
    calls = []
    def factory(**kwargs):
        calls.append(kwargs)
        return ToyDataset()
    monkeypatch.setattr(evaluate, "get_dataset", factory)
    args = Namespace(dataset="wave.Layer", data_path="example", split="test", ar_steps=3,
                     initial_time=0, final_time=15, num_trajectories=None, batch_size=2, num_workers=0)
    ds, loader, meta = evaluate.build_evaluation_loader(args)
    assert len(ds) == 5 and sum(len(b["labels"]) for b in loader) == 5
    assert calls[0]["num_trajectories"] == -1
    assert calls[0]["fix_input_to_time_step"] == 0
    assert meta["trajectory_indices"] == [0, 1, 2, 3, 4]
    args.num_trajectories = 3
    ds, _, meta = evaluate.build_evaluation_loader(args)
    assert len(ds) == 3 and meta["num_samples"] == 3
    args.num_trajectories = 6
    with pytest.raises(ValueError):
        evaluate.build_evaluation_loader(args)


def test_empty_and_nonfinite_fail():
    ds = ToyDataset()
    with pytest.raises(ValueError, match="empty"):
        evaluate.evaluate_model(AddTime(), [], dataset=ds, dataset_name="wave.Layer", device="cpu")
    class Invalid(nn.Module):
        def forward(self, pixel_values, **kwargs):
            return pixel_values * float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        evaluate.evaluate_model(Invalid(), DataLoader(ds, batch_size=2), dataset=ds,
                                dataset_name="reaction_diffusion.AllenCahn", device="cpu")
