"""Independent VICON evaluation with the SafeSVD research/paired sampling rules.

The original loss and physics equations are retained, using the local SBPQ
model/data adapters. No training optimizer, plotting code or external checkout
is loaded. Results use the repository's compact bits/metric/change format.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import islice
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from evaluation.reporting import build_results, final_average_weight_bits
from quantization.artifacts import load_quantized_artifact
from quantization.checkpoints import atomic_json
from quantization.common import move_to_device, resolve_device
from .checkpoints import load_compressed_state
from .data_utils import load_config, prepare_pairs
from .dataset import all_datasets
from .model_utils import load_model, model_forward
from .physics import average_metrics, physics_metrics
from .sobolev import query_fields, valid_mask, masked_mean


def batch_factory(cfg, *, batch_size, num_workers=0, max_batches=None, protocol="research"):
    """Research advances the same datasets across loss/physics and both models."""
    datasets = all_datasets(cfg.datasets, num_workers, int(cfg.test_seed), "test")
    for category, dataset in datasets.items():
        if not dataset.dp:
            raise FileNotFoundError(f"No VICON test HDF5 files for {category}: {cfg.datasets.types[category].folder}")

    def batches():
        selected = deepcopy(datasets) if protocol == "paired" else datasets
        for category, dataset in selected.items():
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
            for payload in islice(loader, max_batches):
                yield category, payload
    return batches


def evaluate_model(model, batches, cfg, device):
    """Keep separate loss and physical-metric passes, as in the research code."""
    model.to(device).eval()
    losses, label_energies, physical = {}, {}, {}
    with torch.inference_mode():
        for category, payload in batches():
            batch = prepare_pairs(move_to_device(payload[1], device), cfg)
            prediction = query_fields(model_forward(model, batch, device), batch)
            label = query_fields(batch["labels"], batch)
            mask = valid_mask(batch)
            if not mask.any():
                raise ValueError(f"VICON test batch for {category} has no valid query cells")
            loss = float(masked_mean((prediction - label).square(), mask))
            channel_validity = batch["channel_mask"][:, :, :-1]
            # Preserve the original evaluator's relative-loss denominator.
            energy = float((label.square() * channel_validity).sum()
                           / (channel_validity.sum() * label.shape[-1] * label.shape[-2] + 1e-12))
            losses.setdefault(category, []).append(loss)
            label_energies.setdefault(category, []).append(energy)
        for category, payload in batches():
            batch = prepare_pairs(move_to_device(payload[1], device), cfg)
            prediction = query_fields(model_forward(model, batch, device), batch)
            label = query_fields(batch["labels"], batch)
            mean, std = batch["mean"][:, :, :-1], batch["std"][:, :, :-1]
            values = physics_metrics(category, prediction * std + mean, label * std + mean)
            physical.setdefault(category, []).append(values)
    if not losses:
        raise ValueError("VICON evaluation dataset is empty")
    mean = lambda values: sum(values) / len(values)
    return {
        category: {**average_metrics(physical.get(category, [])), "base_loss": mean(values),
                   "rel_base_loss": mean(values) / (mean(label_energies[category]) + 1e-12)}
        for category, values in losses.items()
    }


def run(args):
    device = resolve_device(args.DEV)
    model_args = SimpleNamespace(model_path=str(Path(args.checkpoint).expanduser().resolve()),
                                 data_path=str(Path(args.data_path).expanduser().resolve()),
                                 config_dir=args.config_dir, config_name=args.config_name,
                                 config_override=args.config_override, num_workers=args.num_workers)
    cfg = load_config(model_args)
    batches = batch_factory(cfg, batch_size=args.batch_size, num_workers=args.num_workers,
                            max_batches=args.max_batches, protocol=args.protocol)
    print(f"VICON test evaluation: protocol={args.protocol}; batch_size={args.batch_size}", flush=True)
    origin_model = load_model(model_args, device, explicit_attention=False)
    origin = evaluate_model(origin_model, batches, cfg, device)
    origin_model.cpu()
    del origin_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    compressed, artifact_metadata = None, {}
    if args.model_path:
        path = Path(args.model_path).expanduser().resolve()
        metadata_path = path / "metadata.json" if path.is_dir() else path.parent / "metadata.json"
        if metadata_path.is_file():
            artifact_metadata = json.loads(metadata_path.read_text())
        if path.is_dir() or path.suffix == ".pt":
            model, _ = load_quantized_artifact(path, device=device)
        else:
            model = load_compressed_state(load_model(model_args, device), path).to(device).eval()
        compressed = evaluate_model(model, batches, cfg, device)
    average_bits = final_average_weight_bits(artifact_metadata) if args.model_path else 32.0
    return {
        "avg_weight_bits": average_bits,
        "metrics": {
            category: build_results(avg_weight_bits=average_bits, origin_metrics=metrics,
                                    compressed_metrics=compressed[category] if compressed is not None else None)["metrics"]
            for category, metrics in origin.items()
        },
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--ckpt_path", required=True, help="Original VICON checkpoint.")
    parser.add_argument("--model_path", "--model", help="SBPQ artifact directory, full .pt model, or .pth state dict.")
    parser.add_argument("--data_path", "--data-root", required=True)
    parser.add_argument("--DEV", "--device", default="cuda")
    parser.add_argument("--batch_size", "--batch-size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_batches", "--max-batches", type=int, help="Optional test batch cap per category/pass.")
    parser.add_argument("--config_dir", "--config-dir")
    parser.add_argument("--config_name", "--config-name", default="default")
    parser.add_argument("--config_override", "--config-override", action="append", default=[])
    parser.add_argument("--protocol", choices=("research", "paired"), default="research")
    parser.add_argument("--output_path", "--output")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.num_workers < 0 or (args.max_batches is not None and args.max_batches < 1):
        parser.error("batch_size/max_batches must be positive; num_workers must be nonnegative")
    result = run(args)
    if args.output_path:
        atomic_json(result, args.output_path)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return result
