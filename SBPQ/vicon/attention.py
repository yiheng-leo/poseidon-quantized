# Adapted from SafeSVD/VICON; see VENDORED.md and LICENSE-*.txt in this directory.
"""VICON attention preparation copied locally from MySVD/convert_attn.py."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

class ExplicitMultiheadAttention(nn.Module):
    """
    Drop-in replacement for nn.MultiheadAttention that uses four separate
    nn.Linear layers (q_proj, k_proj, v_proj, out_proj) instead of a fused
    in_proj_weight parameter.

    This makes every projection visible to code that iterates over nn.Linear
    modules (e.g., SVD profiling, Fisher collection, weight replacement).
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.0, bias: bool = True,
                 batch_first: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # ── Compatibility attributes ──
        # nn.TransformerEncoderLayer.forward inspects these on self.self_attn
        # BEFORE calling self_attn(...) to decide fast-path vs standard-path.
        # Setting them to None forces the standard path (which just calls forward).
        self.in_proj_weight = None
        self.in_proj_bias = None
        self._qkv_same_embed_dim = True

    @classmethod
    def from_native(cls, mha: nn.MultiheadAttention) -> "ExplicitMultiheadAttention":
        """
        Create an ExplicitMultiheadAttention from a native nn.MultiheadAttention,
        copying all weights exactly.
        """
        if mha.kdim != mha.embed_dim or mha.vdim != mha.embed_dim or mha.bias_k is not None or mha.bias_v is not None or mha.add_zero_attn:
            raise ValueError("VICON conversion requires equal Q/K/V dimensions without extra bias/zero tokens")
        embed_dim = mha.embed_dim
        num_heads = mha.num_heads
        dropout = mha.dropout
        bias = mha.in_proj_bias is not None
        batch_first = mha.batch_first

        module = cls(embed_dim, num_heads, dropout=dropout, bias=bias,
                     batch_first=batch_first)
        parameter = next(mha.parameters())
        module = module.to(device=parameter.device, dtype=parameter.dtype)
        module.train(mha.training)

        # Split the fused in_proj_weight [3*E, E] -> three [E, E]
        with torch.no_grad():
            if mha.in_proj_weight is not None:
                q_w, k_w, v_w = mha.in_proj_weight.chunk(3, dim=0)
                module.q_proj.weight.copy_(q_w)
                module.k_proj.weight.copy_(k_w)
                module.v_proj.weight.copy_(v_w)
            elif hasattr(mha, 'q_proj_weight'):
                # Rare: separate projection weights (when kdim != vdim)
                module.q_proj.weight.copy_(mha.q_proj_weight)
                module.k_proj.weight.copy_(mha.k_proj_weight)
                module.v_proj.weight.copy_(mha.v_proj_weight)

            if bias and mha.in_proj_bias is not None:
                q_b, k_b, v_b = mha.in_proj_bias.chunk(3, dim=0)
                module.q_proj.bias.copy_(q_b)
                module.k_proj.bias.copy_(k_b)
                module.v_proj.bias.copy_(v_b)

            # out_proj is already nn.Linear in native MHA
            module.out_proj.weight.copy_(mha.out_proj.weight)
            if mha.out_proj.bias is not None:
                module.out_proj.bias.copy_(mha.out_proj.bias)

        return module

    def forward(self, query, key=None, value=None,
                key_padding_mask=None, need_weights=False,
                attn_mask=None, is_causal=False):
        """
        Mimics nn.MultiheadAttention.forward signature.
        For VICON, the TransformerEncoder calls self_attn(src, src, src, ...).
        """
        if key is None:
            key = query
        if value is None:
            value = query

        # Handle batch_first
        if self.batch_first:
            # input: [B, T, E] -> we work in [T, B, E] internally
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        T, B, E = query.shape
        S = key.shape[0]

        # Project
        q = self.q_proj(query)  # [T, B, E]
        k = self.k_proj(key)    # [S, B, E]
        v = self.v_proj(value)  # [S, B, E]

        # Reshape for multi-head: [T, B, nH, hD] -> [B, nH, T, hD]
        # (using B*nH as batch dim for bmm)
        nH = self.num_heads
        hD = self.head_dim

        q = q.view(T, B, nH, hD).permute(1, 2, 0, 3)  # [B, nH, T, hD]
        k = k.view(S, B, nH, hD).permute(1, 2, 0, 3)  # [B, nH, S, hD]
        v = v.view(S, B, nH, hD).permute(1, 2, 0, 3)  # [B, nH, S, hD]

        # ── Prepare attn_mask for F.scaled_dot_product_attention ──
        # SDPA expects mask shape [B, nH, T, S] or [T, S] or [1, 1, T, S]
        # Native MHA bool masks use True=blocked; convert to additive masks
        # because SDPA's bool masks use True=allowed.
        # Float mask: additive
        sdpa_mask = None
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill(attn_mask, float('-inf'))
            else:
                attn_mask = attn_mask.to(dtype=q.dtype)
            if attn_mask.dim() == 2:
                # [T, S] -> [1, 1, T, S]  works for both bool and float
                sdpa_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                # [B*nH, T, S] -> [B, nH, T, S]
                sdpa_mask = attn_mask.view(B, nH, T, S)
            else:
                raise ValueError("attn_mask must have 2 or 3 dimensions")

        # Merge key_padding_mask into sdpa_mask
        if key_padding_mask is not None:
            # key_padding_mask: [B, S] -> [B, 1, 1, S]
            kpm = key_padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
            if kpm.dtype == torch.bool:
                kpm = torch.zeros_like(kpm, dtype=q.dtype).masked_fill(kpm, float('-inf'))
            else:
                kpm = kpm.to(dtype=q.dtype)
            sdpa_mask = kpm if sdpa_mask is None else sdpa_mask + kpm

        # F.scaled_dot_product_attention uses Flash/Memory-efficient kernels
        # -> never materializes the full [B, nH, T, S] attention matrix
        dropout_p = self.dropout if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )  # [B, nH, T, hD]

        # Reshape back: [B, nH, T, hD] -> [T, B, E]
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(T, B, E)

        # Output projection
        attn_output = self.out_proj(attn_output)

        if self.batch_first:
            attn_output = attn_output.transpose(0, 1)  # [B, T, E]

        if need_weights:
            # SDPA doesn't return attention weights; would need manual fallback.
            # In practice VICON never requests weights, so just warn.
            import warnings
            warnings.warn("need_weights=True not supported with SDPA path, returning None")
        return attn_output, None


def convert_attn_to_linear(model: nn.Module, *, verbose=True) -> Dict[str, nn.MultiheadAttention]:
    """
    Find all nn.MultiheadAttention modules in the model and replace them
    with ExplicitMultiheadAttention (which uses separate nn.Linear layers).

    Args:
        model: The model to convert (modified in-place).

    Returns:
        A dict mapping module paths to the original nn.MultiheadAttention modules,
        in case you want to revert later.
    """
    originals = {}

    # Collect targets first to avoid modifying dict during iteration
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            targets.append((name, module))

    for name, mha in targets:
        explicit = ExplicitMultiheadAttention.from_native(mha)
        # Move to same device
        device = next(mha.parameters()).device
        explicit = explicit.to(device)

        # Replace in the parent module
        _set_module_by_name(model, name, explicit)
        originals[name] = mha
        if verbose:
            print(f"[convert_attn] Replaced {name}: MultiheadAttention -> ExplicitMultiheadAttention")

    if verbose:
        print(f"[convert_attn] Converted {len(targets)} attention modules total.")
    return originals


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module):
    """Set a submodule by its dot-separated name path."""
    parts = name.split(".")
    parent = model
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    if parts[-1].isdigit():
        parent[int(parts[-1])] = new_module
    else:
        setattr(parent, parts[-1], new_module)

