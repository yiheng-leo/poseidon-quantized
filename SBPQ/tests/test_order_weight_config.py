from types import SimpleNamespace

import pytest
import torch

from SBPQ.poseidon.config import SBPQConfig
from SBPQ.poseidon.run_sbpq_network_global import build_sensitivity_loss_function, select_candidate_ranges
from SBPQ.poseidon.sobolev import get_order_weights


@pytest.mark.parametrize("variable", ["SBPQ_SOBOLEV_ORDER_WEIGHTS", "SBPQ_SOB_WEIGHTS"])
def test_scalar_environment_matches_explicit_second_order_loss(monkeypatch, variable):
    monkeypatch.delenv("SBPQ_SOBOLEV_ORDER_WEIGHTS", raising=False)
    monkeypatch.delenv("SBPQ_SOB_WEIGHTS", raising=False)
    monkeypatch.setenv("SBPQ_SOBOLEV_ORDER", "2")
    monkeypatch.setenv(variable, "1")
    scalar = SBPQConfig()
    assert scalar.sobolev_order_weights == 1.0
    assert get_order_weights(2, scalar.sobolev_order_weights) == [1.0, 1.0, 1.0]
    monkeypatch.setenv(variable, "1,1,1")
    vector = SBPQConfig()
    dataset = SimpleNamespace(constants={"mean": [0, 0, 0, 0], "std": [1, 1, 1, 1]})
    torch.manual_seed(8)
    prediction = torch.randn(2, 4, 8, 8, requires_grad=True)
    reference = torch.randn_like(prediction)
    first = build_sensitivity_loss_function(scalar, dataset)(prediction, reference)
    second = build_sensitivity_loss_function(vector, dataset)(prediction, reference)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(first, prediction)[0],
                               torch.autograd.grad(second, prediction)[0], rtol=0, atol=0)


def test_invalid_explicit_weights_fail_before_model_loading(monkeypatch):
    monkeypatch.setenv("SBPQ_SOBOLEV_ORDER", "2")
    monkeypatch.setenv("SBPQ_SOBOLEV_ORDER_WEIGHTS", "1,1")
    with pytest.raises(ValueError, match="exactly 3"):
        SBPQConfig()


def test_superset_range_cache_keeps_only_selected_layers_and_order():
    ranges = {"layer.b": {"weight_ranges": torch.ones(2)},
              "unselected.norm": {"weight_ranges": torch.ones(4)},
              "layer.a": {"weight_ranges": torch.ones(3)}}
    result = select_candidate_ranges(ranges, ["layer.a", "layer.b"])
    assert list(result) == ["layer.b", "layer.a"]  # Original noise sampling order.
    assert result["layer.a"] is ranges["layer.a"]
    assert "unselected.norm" in ranges  # Do not modify the shared cache.
    with pytest.raises(ValueError, match="missing 1 selected"):
        select_candidate_ranges(ranges, ["missing"])
