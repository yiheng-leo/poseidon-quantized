"""SafeSVD calibration streams and the original VICON bc normalization."""
from __future__ import annotations

from collections import OrderedDict
import importlib
import inspect
from itertools import islice
from pathlib import Path

import torch

from SBPQ.resume import fingerprint
from .model_utils import configure_source


def load_config(args):
    if hasattr(args, "_vicon_config"):
        return args._vicon_config
    root = configure_source(args.safe_svd_root)
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict
    directory = Path(args.config_dir or root / "VICON/configs").expanduser().resolve()
    with initialize_config_dir(config_dir=str(directory), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=args.config_override)
    with open_dict(cfg):
        cfg.dataset_workers = args.num_workers
        for name, dataset_config in cfg.datasets.types.items():
            for field in ("folder", "path", "data_path"):
                if field not in dataset_config:
                    continue
                path = Path(dataset_config[field]).expanduser()
                explicit = any(override.lstrip("+").startswith((f"datasets.types.{name}.{field}=",
                                                                          f"datasets.{name}.{field}="))
                               for override in args.config_override)
                if args.data_path is not None and not explicit:
                    relative = Path(path.name) if path.is_absolute() or ".." in path.parts else path
                    path = Path(args.data_path) / relative
                elif not path.is_absolute():
                    path = Path.cwd() / path
                dataset_config[field] = str(path.resolve())
    args._vicon_config = cfg
    return cfg


def prepare_pairs(pairs, cfg):
    """Match VICON Trainer._data_preprocess without creating a training optimizer."""
    initial, end, channel_mask = pairs
    channel_mask = channel_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
    if cfg.loss.scale == "bc":
        mean = torch.mean(initial, dim=(1, -2, -1), keepdim=True)
        std = torch.std(initial, dim=(1, -2, -1), keepdim=True)
        mean = torch.where(channel_mask == 0, 0, mean)
        std = torch.where(channel_mask == 0, 1, std)
        mean[:, :, -1] = 0
        std[:, :, -1] = 1
        if cfg.model.preprocess.scheme == "scale_to_one":
            std = torch.where(std < cfg.model.preprocess.eps, 1, std)
        elif cfg.model.preprocess.scheme == "scale_to_eps":
            std = torch.where(std < cfg.model.preprocess.eps, cfg.model.preprocess.eps, std)
        else:
            raise ValueError(f"Unsupported VICON preprocess scheme: {cfg.model.preprocess.scheme}")
    elif cfg.loss.scale == "none":
        mean = torch.zeros_like(initial[:, :1, :, :1, :1])
        std = torch.ones_like(mean)
    else:
        raise ValueError(f"Unsupported VICON loss.scale: {cfg.loss.scale}")
    if not 0 <= cfg.loss.min_ex < end.shape[1]:
        raise ValueError("VICON loss.min_ex must leave at least one query pair")
    return {"pixel_values": (initial - mean) / std, "labels": (end - mean) / std,
            "channel_mask": channel_mask, "mean": mean, "std": std,
            "min_ex": int(cfg.loss.min_ex)}


class CalibrationBatches:
    """Traverse each category with SafeSVD's floor(calibration_num / batch_size) cap."""

    def __init__(self, loaders, cfg, max_batches, metadata):
        self.loaders, self.cfg = loaders, cfg
        self.max_batches, self.metadata = max_batches, metadata

    def __iter__(self):
        digests, counts, batches = [], OrderedDict(), 0
        for category, loader in self.loaders.items():
            counts[category] = 0
            for payload in islice(loader, self.max_batches):
                batch = prepare_pairs(payload[1], self.cfg)
                batch.update(category=category, time_in=payload[2], time_out=payload[3], delta_t=payload[4])
                counts[category] += len(batch["pixel_values"])
                batches += 1
                digests.append(fingerprint(batch))
                yield batch
            if not counts[category]:
                raise ValueError(f"VICON calibration category {category} is empty")
        self.metadata.update(num_samples=sum(counts.values()), category_samples=dict(counts),
                             num_batches=batches, fingerprint=fingerprint(digests))

    def __len__(self):
        return len(self.loaders) * self.max_batches


def build_calibration_loader(args):
    from omegaconf import OmegaConf
    cfg = load_config(args)
    dataset_module = importlib.import_module("VICON.src.dataset")
    from safe_svd.helper.collector.vicon.vicon_calibration_dataset import build_calibration_loaders
    from safe_svd.helper.collector.vicon.vicon_file_order import apply_vicon_file_order
    dataset_seed = args.calibration_seed if args.random_calibration == "y" else cfg.train_seed
    values = (cfg.datasets, 0, dataset_seed, "train")
    datasets = (dataset_module.all_datasets(*values, cfg.rollout)
                if "rollout_cfg" in inspect.signature(dataset_module.all_datasets).parameters
                else dataset_module.all_datasets(*values))
    for category, dataset in datasets.items():
        if hasattr(dataset, "dp") and not dataset.dp:
            raise FileNotFoundError(f"No VICON training HDF5 files for {category}: {cfg.datasets.types[category].folder}")
    order_digest = None
    if args.vicon_file_order != "filesystem":
        order_digest = apply_vicon_file_order(datasets, args.vicon_file_order)
    loaders = build_calibration_loaders(
        datasets, calibration_num=args.calibration_num, batch_size=args.batch_size,
        num_workers=args.num_workers, random_calibration=args.random_calibration == "y",
        calibration_seed=args.calibration_seed,
    )
    metadata = {
        "dataset": "vicon", "split": "train", "selection": "safe_svd_vicon_calibration_loaders",
        "categories": list(loaders), "requested_samples_per_category": args.calibration_num,
        "max_batches_per_category": args.calibration_num // args.batch_size,
        "calibration_seed": args.calibration_seed, "random_calibration": args.random_calibration,
        "file_order_sha256": order_digest, "num_trajectories": None, "num_samples": None,
        "loader": {"batch_size": args.batch_size, "num_workers": args.num_workers, "shuffle": False},
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    loader = CalibrationBatches(loaders, cfg, args.calibration_num // args.batch_size, metadata)
    return loader, metadata
