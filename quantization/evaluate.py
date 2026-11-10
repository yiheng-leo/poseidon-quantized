"""Compatibility entry; use ``python -m evaluation.evaluate`` for evaluation."""
from evaluation.evaluate import (
    build_evaluation_loader,
    build_parser,
    evaluate_model,
    main,
)
from evaluation.metrics import denormalize_output, per_sample_metrics
from evaluation.rollout import autoregressive_predict

__all__ = [
    "autoregressive_predict", "build_evaluation_loader", "build_parser",
    "denormalize_output", "evaluate_model", "main", "per_sample_metrics",
]


if __name__ == "__main__":
    main()
