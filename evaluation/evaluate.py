"""Evaluate Poseidon artifacts with SAFE-SVD's all-time-pairs physical protocol.

Use --checkpoint to also report the original model and percentage increases.
The previous fixed-endpoint evaluator is available with --protocol endpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from quantization.calibration import dataset_metadata, _jsonable, _fingerprint, _underlying_dataset
from evaluation.metrics import denormalize_output, per_sample_metrics
from evaluation.rollout import autoregressive_predict


def get_dataset(*args, **kwargs):
    from scOT.problems.base import get_dataset as factory
    return factory(*args, **kwargs)


class _PrefixDataset(Dataset):
    def __init__(self, source, length):
        self.source_dataset = source
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)
        return self.source_dataset[index]

    def __getattr__(self, name):
        if name == "source_dataset":
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, "source_dataset"), name)


def build_evaluation_loader(args):
    if args.split not in ("val", "test"):
        raise ValueError("Evaluation split must be val or test.")
    static = "elliptic" in args.dataset or "steady" in args.dataset
    if args.ar_steps < 1:
        raise ValueError("ar_steps must be positive.")
    if static and args.ar_steps != 1:
        raise ValueError("Autoregressive rollout is undefined for a static problem.")
    kwargs = {"data_path": str(args.data_path)}
    if not static:
        if args.initial_time < 0 or args.final_time <= args.initial_time:
            raise ValueError("Require 0 <= initial_time < final_time.")
        horizon = args.final_time - args.initial_time
        if horizon % args.ar_steps:
            raise ValueError("final_time - initial_time must be divisible by ar_steps.")
        kwargs.update(fix_input_to_time_step=args.initial_time, time_step_size=horizon,
                      max_num_time_steps=1)
    source = get_dataset(dataset=args.dataset, which=args.split, num_trajectories=-1, **kwargs)
    # ScOT num_trajectories describes training capacity even for val/test splits.
    # Each static sample or fixed-endpoint sample is exactly one held-out trajectory.
    available = len(source)
    requested = getattr(args, "num_trajectories", None)
    if requested is not None and (requested < 1 or requested > available):
        raise ValueError(f"num_trajectories must be in [1, {available}], or omitted for the full split.")
    count = available if requested is None else requested
    dataset = source if count == available else _PrefixDataset(source, count)
    metadata = {
        "schema": "quantized-artifact-evaluation-v1", "dataset": args.dataset,
        "split": args.split, "available_trajectories": available,
        "num_trajectories": count, "num_samples": len(dataset),
        "trajectory_indices": list(range(count)),
        "absolute_trajectory_indices": [int(getattr(_underlying_dataset(source), "start", 0)) + i
                                        for i in range(count)],
        "time_pairs": None if static else [[args.initial_time, args.final_time]],
        "ar_steps": args.ar_steps, "source": dataset_metadata(source),
    }
    metadata["fingerprint"] = _fingerprint(metadata)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False)
    return dataset, loader, metadata


@torch.no_grad()
def evaluate_model(model, loader, *, dataset, dataset_name, device, ar_steps=1):
    model.to(device).eval()
    totals, count = {}, 0
    for batch in loader:
        pred = autoregressive_predict(model, batch, device=device, ar_steps=ar_steps)
        target = batch["labels"].to(device=device)
        # The model can predict fewer channels than the dataset input/labels.
        target = target[:, :pred.shape[1]]
        values = per_sample_metrics(pred, target, dataset, dataset_name)
        for key, value in values.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite {key} encountered during evaluation.")
            totals[key] = totals.get(key, 0.0) + value.sum().item()
        count += pred.shape[0]
    if not count:
        raise ValueError("Evaluation dataset is empty.")
    return {"num_samples": count, "metrics": {key: total/count for key, total in totals.items()}}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("safe-svd", "endpoint"), default="safe-svd")
    parser.add_argument("--model_path", "--model", dest="model_path",
                        help="Saved quantization artifact directory or full model file.")
    parser.add_argument("--checkpoint", help="Original fine-tuned checkpoint for baseline comparison.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_path", "--data-root", dest="data_path", required=True)
    parser.add_argument("--DEV", "--device", dest="DEV", default="cpu")
    parser.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=256)
    parser.add_argument("--physics_batch_size", "--physics-batch-size", dest="physics_batch_size",
                        type=int, default=10)
    parser.add_argument("--eval_obj", "--eval-obj", dest="eval_obj",
                        choices=("ongt", "oncomp"), default="ongt")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--initial_time", type=int, default=0)
    parser.add_argument("--final_time", "--final-time", dest="final_time", type=int, default=20)
    parser.add_argument("--ar_steps", type=int, default=1)
    parser.add_argument("--num_trajectories", "--num-trajectories", dest="num_trajectories", type=int,
                        default=None, help="SAFE-SVD dataset argument (default 64, not a test prefix); "
                        "in endpoint mode, optional held-out trajectory prefix.")
    parser.add_argument("--output_path", "--output", dest="output_path", default=None)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.protocol == "safe-svd":
        from evaluation.poseidon.evaluate import run
        result = run(args)
        serialized = json.dumps(_jsonable(result), indent=2, allow_nan=False)
        if args.output_path:
            output = Path(args.output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(serialized + "\n")
        print(serialized)
        return result
    if not args.model_path:
        parser.error("--protocol endpoint requires --model_path")
    if args.checkpoint or args.eval_obj != "ongt":
        parser.error("--checkpoint and --eval_obj oncomp require --protocol safe-svd")
    from quantization.artifacts import load_quantized_artifact
    model, artifact_meta = load_quantized_artifact(args.model_path, args.DEV)
    dataset, loader, _ = build_evaluation_loader(args)
    result = evaluate_model(model, loader, dataset=dataset, dataset_name=args.dataset,
                            device=args.DEV, ar_steps=args.ar_steps)
    from evaluation.reporting import build_results, final_average_weight_bits
    result = build_results(avg_weight_bits=final_average_weight_bits(artifact_meta),
                           compressed_metrics=result["metrics"])
    serialized = json.dumps(result, indent=2, allow_nan=False)
    if args.output_path:
        output = Path(args.output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n")
    print(serialized)
    return result


if __name__ == "__main__":
    main()
