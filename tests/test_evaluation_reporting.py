"""Compact reports must use achieved bits and retain exact metric values."""
import pytest

from evaluation.reporting import build_results, final_average_weight_bits


@pytest.mark.parametrize("quantization,expected", [
    ({"avg_weight_bits": 7.612345, "target_bits": 8, "initial_bits": 9}, 7.612345),
    ({"weight_bits": 4}, 4),
    ({"effective_bits": 4.2}, 4.2),
    ({"method": "fp", "weight_bits": 32}, 32),
    ({"target_bits": 8, "initial_bits": 9}, None),
    ({}, None),
])
def test_final_bits_are_actual_quantization_statistics(quantization, expected):
    assert final_average_weight_bits({"quantization": quantization}) == expected


def test_report_keeps_values_and_percent_changes_only():
    result = build_results(
        avg_weight_bits=7.612345,
        origin_metrics={"base_loss": 0.125, "continuity": 0.0, "model_size": 999,
                        "linear_size": 500, "rel_base_loss": "-"},
        compressed_metrics={"base_loss": 0.15625, "continuity": 0.01, "model_size": 999,
                            "linear_size": 500, "rel_base_loss": "-"},
    )
    assert result == {
        "avg_weight_bits": 7.612345,
        "metrics": {
            "base_loss": {"origin": 0.125, "compressed": 0.15625, "change_percent": 25.0},
            "continuity": {"origin": 0.0, "compressed": 0.01, "change_percent": None},
        },
    }


def test_origin_only_report_has_no_compressed_or_change_values():
    assert build_results(avg_weight_bits=32, origin_metrics={"base_loss": 0.125}) == {
        "avg_weight_bits": 32, "metrics": {"base_loss": {"origin": 0.125}},
    }


def test_endpoint_cli_uses_the_same_compact_output(tmp_path, monkeypatch):
    import json
    from evaluation import evaluate
    from quantization import artifacts

    monkeypatch.setattr(artifacts, "load_quantized_artifact", lambda *a: (object(), {
        "quantization": {"avg_weight_bits": 6.5}, "calibration": {"indices": list(range(1000))},
    }))
    monkeypatch.setattr(evaluate, "build_evaluation_loader", lambda *a: (object(), [], {}))
    monkeypatch.setattr(evaluate, "evaluate_model", lambda *a, **kw: {
        "num_samples": 100, "metrics": {"L1": 1.23456789, "RelL1": 5.6789},
    })
    output = tmp_path / "metrics.json"
    result = evaluate.main([
        "--protocol", "endpoint", "--model_path", "model.pt", "--dataset", "wave.Layer",
        "--data_path", "data", "--output", str(output),
    ])
    assert json.loads(output.read_text()) == result == {
        "avg_weight_bits": 6.5,
        "metrics": {"L1": {"compressed": 1.23456789}, "RelL1": {"compressed": 5.6789}},
    }
