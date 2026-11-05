from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from SBPQ.beta_prior import compute_layer_beta_prior
from SBPQ.poseidon.likelihood import compute_network_mc_likelihood_single_batch as likelihood
from SBPQ.poseidon.evaluation import evaluate_poseidon_with_weight_steps
from SBPQ.trainer import freeze_model_parameters


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(64, 64)
        self.batch_sizes = []

    def forward(self, pixel_values, time=None, labels=None, pixel_mask=None):
        self.batch_sizes.append(len(pixel_values))
        return SimpleNamespace(output=self.linear(pixel_values).tanh())


def data(model, count=2):
    return {"pixel_values": torch.randn(count, 1, 2, 64, dtype=model.linear.weight.dtype),
            "labels": torch.randn(count, 1, 2, 64, dtype=model.linear.weight.dtype)}


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_replay_matches_stacked_loss_gradients_adam_and_rng(device):
    torch.manual_seed(12)
    template = Model().double().to(device)
    batch = data(template)
    with torch.no_grad():
        target = template(pixel_values=batch["pixel_values"].to(device)).output.cpu()

    def run(replay):
        model = deepcopy(template)
        if replay:
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
            freeze_model_parameters(model)
        step = nn.Parameter(torch.full((64,), 0.025, dtype=torch.float64, device=device))
        optimizer = torch.optim.AdamW([step], lr=1e-4, weight_decay=0)
        torch.manual_seed(21)
        records = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            value = likelihood(model, {"linear": step}, [batch], [target], 0,
                               num_mc_samples=5, eta=1e-6, device=device, memory_efficient=replay)
            prior = compute_layer_beta_prior(step, torch.ones_like(step), step.new_tensor(4.),
                                             step.new_tensor(6.), 1., 16.)
            state = torch.get_rng_state()
            gpu_state = torch.cuda.get_rng_state(device) if device == "cuda" else None
            (value + prior).backward()
            assert torch.equal(state, torch.get_rng_state())
            if gpu_state is not None:
                assert torch.equal(gpu_state, torch.cuda.get_rng_state(device))
            gradient = step.grad.clone()
            optimizer.step()
            records.append((value.detach(), prior.detach(), gradient, step.detach().clone(),
                            torch.rand(5, device=device)))
        if replay:
            assert all(p.grad is None and not p.requires_grad for p in model.parameters())
        assert not model.linear._forward_hooks
        return records

    torch.testing.assert_close(run(True), run(False), rtol=1e-10, atol=1e-10)


def test_checkpoint_reduces_saved_tensor_storage():
    model = Model().requires_grad_(False)
    batch = data(model)
    target = torch.zeros_like(batch["labels"])

    def count(replay):
        sizes = []
        def pack(tensor):
            sizes.append(tensor.numel())
            return tensor
        step = nn.Parameter(torch.full((64,), 0.01))
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            value = likelihood(model, {"linear": step}, [batch], [target], 0,
                               num_mc_samples=5, device="cpu", memory_efficient=replay)
        value.backward()
        return sum(sizes)

    assert count(True) < count(False) / 10


def test_validation_chunks_keep_original_batch_means():
    model = Model().double().eval()
    batches = [data(model, 5), data(model, 3)]
    settings = dict(model=model, dataloader=batches, weight_step_sizes={"linear": torch.full((64,), 0.02)},
                    layer_names=["linear"], device="cpu")
    expected = evaluate_poseidon_with_weight_steps(**settings, forward_batch_size=None)
    model.batch_sizes.clear()
    actual = evaluate_poseidon_with_weight_steps(**settings, forward_batch_size=2)
    assert model.batch_sizes == [2, 2, 1, 2, 1]
    assert actual == pytest.approx(expected, rel=1e-12)
    assert not model.linear._forward_hooks


def test_real_poseidon_checkpoint_gradients_match():
    from scOT.model import ScOT, ScOTConfig
    torch.manual_seed(19)
    model = ScOT(ScOTConfig(image_size=8, patch_size=2, embed_dim=8, depths=[1, 1], num_heads=[2, 4],
                            window_size=2, skip_connections=[1, 1], num_channels=4, num_out_channels=4,
                            use_conditioning=True, drop_path_rate=0)).eval().requires_grad_(False)
    batch = {"pixel_values": torch.randn(2, 4, 8, 8), "labels": torch.randn(2, 4, 8, 8),
             "time": torch.tensor([0.1, 0.2]), "pixel_mask": torch.tensor([[False, False, False, True]] * 2)}
    with torch.no_grad():
        target = model(**batch).output
    results = []
    for replay in (False, True):
        steps = {name: nn.Parameter(torch.full((layer.out_features,), 0.01))
                 for name, layer in model.named_modules() if isinstance(layer, nn.Linear)
                 and "norm" not in name and "time" not in name}
        torch.manual_seed(29)
        value = likelihood(model, steps, [batch], [target], 0, num_mc_samples=5, eta=1e-6,
                           device="cpu", memory_efficient=replay)
        value.backward()
        results.append((value.detach(), {name: step.grad for name, step in steps.items()}))
        assert not any(module._forward_hooks for module in model.modules())
    torch.testing.assert_close(results[0], results[1], rtol=2e-5, atol=1e-4)


def test_final_steps_survive_validation_failure(tmp_path, monkeypatch):
    from scOT.model import ScOT, ScOTConfig
    from SBPQ.poseidon.config import SBPQConfig
    from SBPQ.poseidon import run_sbpq_network_global as runner
    cfg = SBPQConfig(repo_root=tmp_path, num_optimization_steps=2, calib_steps=1, val_steps=1,
                     sensitivity_batches=1, max_blocks=1, num_mc_samples=2)
    cfg.model_path = str(tmp_path / "model")
    model = ScOT(ScOTConfig(image_size=8, patch_size=2, embed_dim=8, depths=[1, 1], num_heads=[2, 4],
                            window_size=2, skip_connections=[1, 1], num_channels=4, num_out_channels=4,
                            use_conditioning=True, drop_path_rate=0)).eval()
    batch = {"pixel_values": torch.randn(2, 4, 8, 8), "labels": torch.randn(2, 4, 8, 8),
             "time": torch.tensor([0.1, 0.2]), "pixel_mask": torch.tensor([[False, False, False, True]] * 2)}
    dataset = SimpleNamespace(constants={"mean": [0.8, 0, 0, 0], "std": [0.31, 0.39, 0.35, 0.18]})
    loader = SimpleNamespace(dataset=dataset)
    monkeypatch.setattr(runner, "SBPQConfig", lambda: cfg)
    monkeypatch.setattr(runner, "load_poseidon_model", lambda **kwargs: (model, torch.device("cpu")))
    monkeypatch.setattr(runner, "build_poseidon_loaders", lambda **kwargs:
                        (loader, loader, lambda: iter([batch]), lambda: iter([batch])))
    def fail(**kwargs):
        raise RuntimeError("validation failure")
    monkeypatch.setattr(runner, "evaluate_poseidon_with_weight_steps", fail)
    with pytest.raises(RuntimeError, match="validation failure"):
        runner.main()
    saved = list(tmp_path.rglob("sbpq_trainer_state.pt"))
    assert len(saved) == 1
    state = torch.load(saved[0], weights_only=False)
    assert state["optimized_step_sizes"] and len(state["history"]) == 2
    assert (saved[0].parent / "history.json").is_file()
