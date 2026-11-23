import json

from evaluation.evaluate import main
from evaluation.poseidon import batch


def test_batch_reuses_baseline_and_skips_only_matching_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    weights = [tmp_path / f"model-{i}.pt" for i in range(3)]
    for path in weights:
        path.write_text(path.name)
    checkpoint = tmp_path / "origin"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    monkeypatch.setattr(batch.evaluate, "open_test_dataset", lambda args: [1, 2])
    monkeypatch.setattr(batch, "dataset_metadata", lambda _: {"dataset": "fixture"})
    calls = []

    def origin(_args, dataset, **_kwargs):
        calls.append("origin")
        return dataset, {"base_loss": 1.0}

    def compressed(_args, _dataset, path, baseline):
        calls.append(path)
        assert baseline == {"base_loss": 1.0}
        return {"avg_weight_bits": 8.0, "metrics": {"base_loss": {"compressed": 1.1}}}

    monkeypatch.setattr(batch.evaluate, "evaluate_origin", origin)
    monkeypatch.setattr(batch.evaluate, "evaluate_compressed", compressed)
    settings = [
        "--dataset", "fluids.incompressible.PiecewiseConstants",
        "--data_path", str(tmp_path), "--checkpoint", str(checkpoint),
    ]

    first, second, third = map(str, weights)
    result = main([*settings, "--models", first, second])
    assert result["evaluated"] == [first, second]
    assert calls == ["origin", first, second]
    result = main([*settings, "--models", first, second, third])
    assert result["skipped"] == [first, second]
    assert result["evaluated"] == [third]
    assert calls == ["origin", first, second, third]

    weights[1].write_text("replaced artifact")
    result = main([*settings, "--models", second])
    assert result["evaluated"] == [second]
    assert calls == ["origin", first, second, third, second]

    other = main([*settings, "--final_time", "10", "--models", first])
    assert other["profile"] != result["profile"]
    assert calls[-2:] == ["origin", first]
    report = json.loads((tmp_path / "results/NS-PwC/result.json").read_text())
    assert set(report["protocols"][result["profile"]]["models"]) == {first, second, third}
