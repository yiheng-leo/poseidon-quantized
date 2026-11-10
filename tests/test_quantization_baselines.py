"""Numerical, calibration-coverage, and standalone-artifact baseline checks."""
from argparse import Namespace
import copy
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from quantization.layers import QuantizedLinear
from quantization.common import QuantizationResult
import importlib
import importlib.util


def quantize_baseline(model, calibration_loader, args, device, work_dir):
    if args.method == "fp":
        return QuantizationResult(model.to(device).eval(), {"method": "fp", "execution": "floating_point"})
    modules = {"uniform": "UniformQ", "dynamic": "DynamicQ", "static": "StaticQ",
               "brecq": "BRECQ", "repq": "RepQ", "ptq4vit": "PTQ4ViT"}
    return importlib.import_module(modules[args.method] + ".quantize").quantize(
        model, calibration_loader, args, device, work_dir)
from quantization.common import forward_model


class ToyPhysicalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(3, 5), nn.GELU(), nn.Linear(5, 2))

    def forward(self, pixel_values, time=None):
        return self.network(pixel_values)


def arguments(method, **overrides):
    values = dict(method=method, weight_bits=4, activation_bits=32, seed=7,
                  reconstruction_iterations=2, rec_batchsize=2, batch_size=2,
                  ptq_search_rounds=1, ptq_candidates=4)
    if method in {"dynamic", "static", "repq", "ptq4vit"}:
        values["activation_bits"] = 4
    return Namespace(**(values | overrides))


def calibration():
    return [{"pixel_values": torch.tensor([[0.2, -0.1, 0.5], [0.8, 0.3, -0.2]]),
             "labels": torch.tensor([[0.7, -0.2], [0.1, 0.9]])},
            {"pixel_values": torch.tensor([[2.4, -1.3, 0.6]]),
             "labels": torch.tensor([[0.2, -0.4]])}]


@pytest.mark.parametrize("method", ["fp", "uniform", "dynamic", "static", "repq", "ptq4vit", "brecq"])
def test_baseline_finite_standalone_and_complete_calibration(method, tmp_path):
    torch.manual_seed(3)
    model = ToyPhysicalModel().eval()
    before = copy.deepcopy(model)
    result = quantize_baseline(model, calibration(), arguments(method), "cpu", tmp_path)
    sample = calibration()[0]
    prediction = forward_model(result.model, sample, "cpu")
    assert prediction.shape == (2, 2)
    assert torch.isfinite(prediction).all()
    if method in {"static", "repq", "ptq4vit", "brecq"}:
        assert result.metadata["calibration_samples"] == 3
        assert result.metadata["calibration_batches"] == 2
    if method == "fp":
        torch.testing.assert_close(prediction, forward_model(before, sample, "cpu"))
    else:
        assert result.metadata["execution"] == "fake_quantization"
    if method == "ptq4vit":
        assert result.metadata["twin_uniform_layers"] == ["network.2"]
    checkpoint = tmp_path / "compressed_model.pt"
    torch.save(result.model, checkpoint)
    restored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    torch.testing.assert_close(prediction, forward_model(restored, sample, "cpu"), rtol=0, atol=0)
    # No hooks, transient calibration records or vendor quantizers are needed.
    assert not any(module._forward_hooks or module._forward_pre_hooks for module in restored.modules())
    assert not list(tmp_path.glob("baseline-calibration-*"))


def test_saved_linear_loads_in_fresh_process_without_source_checkpoint(tmp_path):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.GELU(), nn.Linear(4, 2))
    result = quantize_baseline(model, None, arguments("dynamic"), "cpu", tmp_path)
    sample = torch.randn(3, 3)
    checkpoint = tmp_path / "compressed_model.pt"
    torch.save(result.model, checkpoint)
    torch.save(sample, tmp_path / "sample.pt")
    command = [sys.executable, "-c",
               "import sys,torch; from pathlib import Path; p=Path(sys.argv[1]); "
               "m=torch.load(p/'compressed_model.pt',map_location='cpu',weights_only=False); "
               "x=torch.load(p/'sample.pt',weights_only=True); torch.save(m(x),p/'prediction.pt')",
               str(tmp_path)]
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1], env=os.environ.copy())
    actual = torch.load(tmp_path / "prediction.pt", weights_only=True)
    torch.testing.assert_close(actual, result.model(sample), atol=0, rtol=0)


def test_static_range_includes_last_small_batch(tmp_path):
    model = ToyPhysicalModel()
    result = quantize_baseline(model, calibration(), arguments("static"), "cpu", tmp_path)
    expected = torch.tensor([2.4, 1.3, 0.6]) / 7
    torch.testing.assert_close(result.model.network[0].input_scale, expected)


@pytest.mark.parametrize("bits", [4, 8, 16])
def test_uniform_matches_existing_ppq_dynamic_step_grid(bits, tmp_path):
    # Independent oracle for the original PPQ.metrics grid; production does
    # not import archived training implementations.
    def compute_dynamic_stepsizes(model, names, num_bits, device):
        return {name: 2.0 * module.weight.detach().abs().amax(1) / (2 ** num_bits - 1)
                for name, module in model.named_modules() if name in names}
    torch.manual_seed(12)
    model = ToyPhysicalModel().eval()
    names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    steps = compute_dynamic_stepsizes(model, names, num_bits=bits, device="cpu")
    reference = copy.deepcopy(model)
    with torch.no_grad():
        for name, module in reference.named_modules():
            if name in steps:
                step = steps[name].view(-1, 1)
                module.weight.copy_(torch.round(module.weight / step) * step)
    result = quantize_baseline(model, None, arguments("uniform", weight_bits=bits), "cpu", tmp_path)
    sample = calibration()[0]
    torch.testing.assert_close(forward_model(result.model, sample, "cpu"),
                               forward_model(reference, sample, "cpu"), rtol=0, atol=0)


@pytest.mark.parametrize("bits", [4, 8])
def test_dynamic_matches_existing_dynamicq_weights_and_activations(bits, tmp_path):
    reference_file = Path(__file__).resolve().parents[1] / "archive/main/DynamicQ/dynamic_weight_activation.py"
    spec = importlib.util.spec_from_file_location("dynamicq_reference", reference_file)
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    make_dynamic_weight_activation_quantized_copy = reference_module.make_dynamic_weight_activation_quantized_copy
    torch.manual_seed(12)
    model = ToyPhysicalModel().eval()
    names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    reference = make_dynamic_weight_activation_quantized_copy(model, names, bitwidth=bits, device="cpu")
    result = quantize_baseline(model, None, arguments("dynamic", weight_bits=bits, activation_bits=bits), "cpu", tmp_path)
    sample = calibration()[0]
    torch.testing.assert_close(forward_model(result.model, sample, "cpu"),
                               forward_model(reference, sample, "cpu"), rtol=0, atol=0)


def test_repq_transform_preserves_fp_function_before_rounding(tmp_path):
    torch.manual_seed(2)
    model = ToyPhysicalModel().eval()
    reference = copy.deepcopy(model)
    result = quantize_baseline(model, calibration(), arguments("repq", weight_bits=32), "cpu", tmp_path)
    for module in result.model.modules():
        if isinstance(module, QuantizedLinear):
            module.activation_bits = 32
    sample = calibration()[0]
    torch.testing.assert_close(forward_model(result.model, sample, "cpu"),
                               forward_model(reference, sample, "cpu"), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("method,overrides,match", [
    ("uniform", {"activation_bits": 8}, "weight-only"),
    ("repq", {"activation_bits": 32}, "activation"),
    ("brecq", {"weight_bits": 16}, "BRECQ"),
    ("static", {"weight_bits": 1}, "weight_bits"),
])
def test_invalid_settings_fail_before_consuming_loader(method, overrides, match, tmp_path):
    def forbidden_loader():
        raise AssertionError("Invalid settings must fail before calibration")
        yield
    with pytest.raises(ValueError, match=match):
        quantize_baseline(ToyPhysicalModel(), forbidden_loader(), arguments(method, **overrides), "cpu", tmp_path)


@pytest.mark.parametrize("method", ["static", "repq", "ptq4vit", "brecq"])
def test_empty_calibration_is_rejected(method, tmp_path):
    with pytest.raises(ValueError, match="empty"):
        quantize_baseline(ToyPhysicalModel(), [], arguments(method), "cpu", tmp_path)


def test_ptq4vit_requires_nonzero_label_gradient(tmp_path):
    model = ToyPhysicalModel().eval()
    batches = calibration()
    for batch in batches:
        batch["labels"] = model(batch["pixel_values"]).detach()
    with pytest.raises(ValueError, match="gradients are all zero"):
        quantize_baseline(model, batches, arguments("ptq4vit"), "cpu", tmp_path)


def test_brecq_handles_zero_and_constant_rows_and_rebatches_all_samples(tmp_path):
    model = ToyPhysicalModel().eval()
    with torch.no_grad():
        model.network[0].weight[0].zero_()
        model.network[0].weight[1].fill_(0.25)
        model.network[0].weight[2].fill_(-0.25)
    result = quantize_baseline(model, calibration(), arguments("brecq", rec_batchsize=3), "cpu", tmp_path)
    assert result.metadata["reconstruction_cached_batches"] == 1
    torch.testing.assert_close(result.model.network[0].weight[0], torch.zeros(3))
    assert torch.isfinite(forward_model(result.model, calibration()[0], "cpu")).all()


def test_brecq_rebatching_is_streaming_and_preserves_last_batch():
    from BRECQ.reconstruction import _brecq_calibration_batches
    consumed = []

    def source():
        for index in range(7):
            consumed.append(index)
            yield {"pixel_values": torch.full((1, 3), float(index))}

    stream = _brecq_calibration_batches(source(), 3)
    first = next(stream)
    assert consumed == [0, 1, 2]
    torch.testing.assert_close(first["pixel_values"][:, 0], torch.tensor([0., 1., 2.]))
    rest = list(stream)
    assert [batch["pixel_values"].shape[0] for batch in rest] == [3, 1]
    assert consumed == list(range(7))


@pytest.mark.parametrize("method", ["brecq", "repq", "ptq4vit"])
def test_real_small_scot_forward_and_roundtrip(method, tmp_path):
    pytest.importorskip("transformers")
    from scOT.model import ScOT, ScOTConfig
    torch.manual_seed(7)
    model = ScOT(ScOTConfig(image_size=8, patch_size=2, num_channels=3, num_out_channels=3,
                            embed_dim=8, depths=[1, 1], num_heads=[2, 4], window_size=2,
                            skip_connections=[True, True], use_conditioning=True,
                            residual_model="convnext", drop_path_rate=0.0)).eval()
    batches = [{"pixel_values": torch.randn(1, 3, 8, 8), "time": torch.tensor([0.5]),
                "labels": torch.randn(1, 3, 8, 8)}]
    result = quantize_baseline(model, batches, arguments(method, reconstruction_iterations=1,
                                                        ptq_candidates=2), "cpu", tmp_path)
    actual = forward_model(result.model, batches[0], "cpu")
    assert actual.shape == (1, 3, 8, 8)
    assert torch.isfinite(actual).all()
    checkpoint = tmp_path / "small-scot.pt"
    torch.save(result.model, checkpoint)
    restored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    torch.testing.assert_close(actual, forward_model(restored, batches[0], "cpu"), rtol=0, atol=0)
    torch.save(batches[0], tmp_path / "batch.pt")
    subprocess.run([sys.executable, "-c",
                    "import sys,torch; from pathlib import Path; "
                    "from quantization.common import forward_model; p=Path(sys.argv[1]); "
                    "m=torch.load(p/'small-scot.pt',map_location='cpu',weights_only=False); "
                    "b=torch.load(p/'batch.pt',weights_only=True); "
                    "torch.save(forward_model(m,b,'cpu'),p/'fresh.pt')", str(tmp_path)],
                   check=True, cwd=Path(__file__).resolve().parents[1], env=os.environ.copy())
    torch.testing.assert_close(actual, torch.load(tmp_path / "fresh.pt", weights_only=True), rtol=0, atol=0)
