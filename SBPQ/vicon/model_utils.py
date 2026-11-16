"""Strict checkpoint loading and hook-visible VICON transformer execution."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


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
    from .attention import convert_attn_to_linear
    convert_attn_to_linear(model, verbose=False)
    for module in model.modules():
        if isinstance(module, nn.TransformerEncoderLayer):
            module.__class__ = HookableEncoderLayer
        if isinstance(module, nn.TransformerEncoder):
            module.enable_nested_tensor = False
            module.use_nested_tensor = False
    return model.eval()


def load_model(args, device, *, explicit_attention=True):
    from .data_utils import load_config
    from .models import ICON_CROPPED, ICON_UNCROPPED
    cfg = load_config(args)
    constructor = ICON_CROPPED if cfg.model.type == "crop" else ICON_UNCROPPED
    checkpoint = Path(args.model_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"VICON checkpoint not found: {checkpoint}; pass --ckpt_path")
    model = constructor(cfg.model)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload.get("state_dict", payload))
    model.load_state_dict({name.removeprefix("module."): value for name, value in state.items()}, strict=True)
    model.to(device).eval()
    return expose_attention(model) if explicit_attention else model


def model_forward(model, batch, device):
    return model(tuple(batch[key].to(device) for key in ("pixel_values", "labels", "channel_mask")))
