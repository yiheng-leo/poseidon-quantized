from copy import deepcopy
from types import SimpleNamespace
import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from SBPQ.resume import atomic_save, resolve_checkpoint
from SBPQ.trainer import SBPQTrainer


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, pixel_values, **kwargs):
        return SimpleNamespace(output=self.linear(pixel_values).reshape(-1, 2, 1, 1))


def setup_trainer(tmp_path, device="cpu", altered_batch=False, **overrides):
    # Rebuild identical inputs regardless of ambient RNG consumed during restart.
    generator = torch.Generator().manual_seed(30)
    torch.manual_seed(40)
    model = TinyModel().to(device)
    batches = [{"pixel_values": torch.randn(2, 3, generator=generator),
                "labels": torch.zeros(2, 2, 1, 1)} for _ in range(3)]
    clean = []
    with torch.no_grad():
        for batch in batches:
            clean.append(model(pixel_values=batch["pixel_values"].to(device)).output.cpu())
    if altered_batch:
        batches[0]["pixel_values"][0, 0] += 1
    beta_path = tmp_path / "beta.pt"
    torch.save({"block_beta_parameters": {"block": {"alpha": torch.tensor(5.), "beta": torch.tensor(5.)}}}, beta_path)
    settings = dict(model=model, frozen_batches=batches, clean_network_outputs=clean,
                    ranges_dict={"linear": {"weight_ranges": torch.ones(2)}},
                    layer_to_block={"linear": "block"}, beta_parameter_path=beta_path,
                    initial_bits=8, minimum_bits=1, maximum_bits=16,
                    learning_rate=1e-4, num_mc_samples=3, eta=1e-6,
                    channel_weights={"linear": torch.full((2,), 3.)}, device=device)
    settings.update(overrides)
    return SBPQTrainer(**settings)


def seed_all(value):
    torch.manual_seed(value)
    random.seed(value)
    np.random.seed(value)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_resume_matches_uninterrupted_updates_history_optimizer_and_rng(tmp_path, device):
    continuous = setup_trainer(tmp_path, device)
    seed_all(123)
    continuous.optimize(8)
    expected_state = deepcopy(continuous.get_state_dict())
    expected_probe = (random.random(), np.random.rand(), torch.rand(5), torch.rand(5, device=device))

    first = setup_trainer(tmp_path, device)
    seed_all(123)
    path = tmp_path / "run" / "sbpq_checkpoint.pt"
    first.optimize(4, checkpoint_path=path, save_every=2, checkpoint_metadata={"dataset": "same"})
    assert torch.load(path, weights_only=True)["completed_steps"] == 4
    assert resolve_checkpoint(path.parent) == path
    resumed = setup_trainer(tmp_path, device)
    seed_all(999)
    resumed.load(path, expected_metadata={"dataset": "same"})
    assert len(resumed.history) == 4
    resumed.optimize(8, checkpoint_path=path, save_every=2)
    actual = resumed.get_state_dict()
    for key in ("step_size_state_dict", "optimizer_state_dict", "history", "completed_steps"):
        torch.testing.assert_close(actual[key], expected_state[key], rtol=0, atol=0)
    probe = (random.random(), np.random.rand(), torch.rand(5), torch.rand(5, device=device))
    torch.testing.assert_close(probe, expected_probe, rtol=0, atol=0)
    assert [entry["batch_index"] for entry in resumed.history] == [0, 1, 2, 0, 1, 2, 0, 1]
    before = deepcopy(resumed.get_step_sizes())
    resumed.optimize(8)  # Completed target: no extra optimizer updates.
    torch.testing.assert_close(resumed.get_step_sizes(), before, rtol=0, atol=0)
    with pytest.raises(ValueError, match="below completed"):
        resumed.optimize(7)


@pytest.mark.parametrize("change", ["batch", "eta", "metadata"])
def test_resume_rejects_mismatches_before_modifying_steps(tmp_path, change):
    original = setup_trainer(tmp_path)
    original.optimize(2)
    path = tmp_path / "checkpoint.pt"
    original.save(path, {"dataset": "first"})
    other = setup_trainer(tmp_path, altered_batch=change == "batch", **({"eta": 1e-5} if change == "eta" else {}))
    before = deepcopy(other.get_step_sizes())
    with pytest.raises(ValueError, match="differs|differ"):
        other.load(path, expected_metadata={"dataset": "second" if change == "metadata" else "first"})
    torch.testing.assert_close(other.get_step_sizes(), before, rtol=0, atol=0)


def test_failed_atomic_write_keeps_last_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pt"
    atomic_save({"completed_steps": 100}, path)
    original = path.read_bytes()
    def fail(value, stream):
        stream.write(b"incomplete")
        raise OSError("disk failure")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk failure"):
        atomic_save({"completed_steps": 200}, path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".checkpoint-*"))


def test_interruption_keeps_periodic_completed_step(tmp_path, monkeypatch):
    trainer = setup_trainer(tmp_path)
    path = tmp_path / "sbpq_checkpoint.pt"
    train_step = trainer.train_step
    def interrupt(batch_index):
        if len(trainer.history) == 3:
            raise KeyboardInterrupt()
        return train_step(batch_index)
    monkeypatch.setattr(trainer, "train_step", interrupt)
    with pytest.raises(KeyboardInterrupt):
        trainer.optimize(8, checkpoint_path=path, save_every=2)
    saved = torch.load(path, weights_only=False)
    assert saved["completed_steps"] == 2  # Third update was after the last save.
    restarted = setup_trainer(tmp_path)
    restarted.load(path)
    restarted.optimize(8)
    assert len(restarted.history) == 8


def test_legacy_checkpoint_can_continue_with_warning(tmp_path):
    trainer = setup_trainer(tmp_path)
    trainer.optimize(2)
    state = trainer.get_state_dict()
    for key in ("rng_state", "training_signature", "completed_steps", "resume_version"):
        state.pop(key)
    path = tmp_path / "legacy.pt"
    atomic_save(state, path)
    restarted = setup_trainer(tmp_path)
    with pytest.warns(RuntimeWarning, match="Legacy checkpoint"):
        restarted.load(path)
    restarted.optimize(3)
    assert len(restarted.history) == 3


def test_runner_resume_only_executes_remaining_steps(tmp_path, monkeypatch):
    from SBPQ.poseidon.config import SBPQConfig
    from SBPQ.poseidon import run_sbpq_network_global as runner
    torch.set_num_threads(1)
    cfg = SBPQConfig(repo_root=tmp_path, num_optimization_steps=2, calib_steps=1, val_steps=1,
                     sensitivity_batches=1, max_blocks=1, num_mc_samples=2, save_every=1)
    cfg.model_path = str(tmp_path / "original")
    from scOT.model import ScOT, ScOTConfig
    torch.manual_seed(42)
    template = ScOT(ScOTConfig(image_size=8, patch_size=2, embed_dim=8, depths=[1, 1], num_heads=[2, 4],
                               window_size=2, skip_connections=[1, 1], num_channels=4, num_out_channels=4,
                               use_conditioning=True, drop_path_rate=0)).eval()
    batch = {"pixel_values": torch.randn(2, 4, 8, 8), "labels": torch.randn(2, 4, 8, 8),
             "time": torch.tensor([0.1, 0.2]), "pixel_mask": torch.tensor([[False, False, False, True]] * 2)}
    loader = SimpleNamespace(dataset=SimpleNamespace(constants={"mean": [0.8, 0, 0, 0], "std": [0.31, 0.39, 0.35, 0.18]}))
    monkeypatch.setattr(runner, "SBPQConfig", lambda: cfg)
    monkeypatch.setattr(runner, "load_poseidon_model", lambda **kwargs: (deepcopy(template), torch.device("cpu")))
    monkeypatch.setattr(runner, "build_poseidon_loaders", lambda **kwargs:
                        (loader, loader, lambda: iter([batch]), lambda: iter([batch])))
    runner.main(["--save_every", "1"])
    checkpoints = list(tmp_path.rglob("sbpq_checkpoint.pt"))
    assert len(checkpoints) == 1
    cfg.num_optimization_steps = 4
    runner.main(["--resume", str(checkpoints[0].parent), "--total_steps", "4"])
    saved = torch.load(checkpoints[0], weights_only=False)
    assert saved["completed_steps"] == len(saved["history"]) == 4
    assert len(list(tmp_path.rglob("sbpq_checkpoint.pt"))) == 1
    assert len(json.loads((checkpoints[0].parent / "history.json").read_text())) == 4
    cfg.num_optimization_steps = None
    runner.main(["--resume", str(checkpoints[0])])  # Saved total remains four, not default epochs * batches.
    assert torch.load(checkpoints[0], weights_only=False)["completed_steps"] == 4
