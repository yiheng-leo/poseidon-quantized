"""VICON projections and complete encoder layers form SBPQ structural blocks."""
from collections import OrderedDict

import torch
from torch import nn


def build_block_mapping(model):
    blocks = OrderedDict()
    for name in ("pre_proj", "post_proj"):
        if isinstance(getattr(model, name, None), nn.Linear):
            blocks[name] = model.get_submodule(name)
    for name, module in model.named_modules():
        if isinstance(module, nn.TransformerEncoderLayer):
            blocks[name] = module
    layers, mapping, counts = OrderedDict(), {}, {}
    for block_name, block in blocks.items():
        counts[block_name] = 0
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                full_name = f"{block_name}.{name}" if name else block_name
                layers[full_name] = module
                mapping[full_name] = block_name
                counts[block_name] += module.weight.numel()
    if not layers or not any(isinstance(block, nn.TransformerEncoderLayer) for block in blocks.values()):
        raise ValueError("Expected VICON pre/post projections and transformer encoder layers")
    channel_weights = {name: torch.full((layer.out_features,), float(layer.in_features))
                       for name, layer in layers.items()}
    return blocks, layers, mapping, counts, channel_weights
