"""Strict checkpoint loading and hook-visible VICON transformer execution."""
from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import sys

import torch
from torch import nn


def source_root(value=None):
    """Find the SafeSVD checkout, which includes its experimental VICON source."""
    value = value or os.environ.get("SAFE_SVD_ROOT")
    if value:
        candidates = [Path(value).expanduser()]
    else:
        project = Path(__file__).resolve().parents[2]
        candidates = [project / "SafeSVD", project.parent / "SafeSVD", project.parent / "SAFE-SVD-v0"]
    for candidate in candidates:
        if (candidate / "VICON/src/models.py").is_file() and (candidate / "safe_svd").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "SafeSVD's experimental VICON source is required; pass --safe_svd_root /path/to/SafeSVD "
        "(containing VICON/ and safe_svd/)."
    )


def configure_source(value):
    root = source_root(value)
    # This repository also has an evaluation/ package. Import only VICON and
    # safe_svd here; the independent evaluator is launched as a separate script.
    if str(root) not in sys.path:
        sys.path.append(str(root))
    for package in ("VICON", "safe_svd"):
        module = sys.modules.get(package)
        spec = importlib.util.find_spec(package) if module is None else module.__spec__
        locations = list(spec.submodule_search_locations or ()) if spec is not None else []
        if any(not Path(location).resolve().is_relative_to(root) for location in locations):
            raise ValueError(f"{package} resolves to another checkout; start a fresh process with the selected SafeSVD source")
    return root


class HookableEncoderLayer(nn.TransformerEncoderLayer):
    """Use PyTorch's unfused equations so hooks reach every Linear projection.

    A real module class also survives torch.save/load, unlike a bound local
    forward replacement. Parameters and state-dict names remain unchanged.
    """

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x


def expose_attention(model):
    from safe_svd.helper.collector.vicon.vicon_attention import convert_attn_to_linear
    convert_attn_to_linear(model, verbose=False)
    for module in model.modules():
        if isinstance(module, nn.TransformerEncoderLayer):
            module.__class__ = HookableEncoderLayer
        if isinstance(module, nn.TransformerEncoder):
            module.enable_nested_tensor = False
            module.use_nested_tensor = False
    return model.eval()


def load_model(args, device):
    from .data_utils import load_config
    configure_source(args.safe_svd_root)
    cfg = load_config(args)
    models = importlib.import_module("VICON.src.models")
    constructor = models.ICON_CROPPED if cfg.model.type == "crop" else models.ICON_UNCROPPED
    checkpoint = Path(args.model_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"VICON checkpoint not found: {checkpoint}; pass --ckpt_path")
    model = constructor(cfg.model)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    model.load_state_dict({name.removeprefix("module."): value for name, value in state.items()}, strict=True)
    return expose_attention(model.to(device))


def model_forward(model, batch, device):
    return model(tuple(batch[key].to(device) for key in ("pixel_values", "labels", "channel_mask")))
