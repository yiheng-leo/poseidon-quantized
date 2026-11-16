"""Shared SBPQ Beta construction, retaining the original numerical implementation.

These routines operate on scalar sensitivities and parameter counts only. Keep
the historical import paths working while exposing a model-independent API to
new adapters.
"""
from SBPQ.poseidon.beta_parameter_builder import (
    build_poseidon_beta_parameters as build_beta_parameters,
)
from SBPQ.resume import atomic_save


def save_beta_parameters(parameters, path, metadata=None):
    atomic_save({"block_beta_parameters": parameters, "metadata": metadata or {}}, path)

__all__ = ["build_beta_parameters", "save_beta_parameters"]
