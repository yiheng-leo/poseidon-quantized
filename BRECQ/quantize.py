"""Poseidon BRECQ adapter: MSE scales, AdaRound reconstruction, materialized weights."""
from __future__ import annotations
import copy
import tempfile
import torch
from quantization.common import QuantizationResult, forward_model
from quantization.progress import progress
from quantization.layers import _replace, _bits, prepare_method
from quantization.cache import calibration_cache, record_calibration
from .quantizers import _initialize_brecq_weight_quantizer
from .reconstruction import _QuantStateModel, _reconstruct_brecq_block

def _brecq(model, layers, calibration, args, weight_bits, activation_bits, device, directory):
    if activation_bits != 32:
        raise ValueError("The Poseidon BRECQ adapter currently supports weight-only reconstruction; set activation_bits=32")
    if not 2 <= weight_bits <= 8:
        raise ValueError("BRECQ's AdaRound quantizer requires weight_bits in 2..8")
    iterations = int(getattr(args, "reconstruction_iterations", 20000))
    if iterations < 1:
        raise ValueError("BRECQ reconstruction_iterations must be positive")
    from .quantizers import QuantModule

    original = copy.deepcopy(model)
    weight_parameters = {"n_bits": weight_bits, "channel_wise": True, "scale_method": "mse"}
    activation_parameters = {"n_bits": 8, "channel_wise": False, "scale_method": "max"}
    poseidon = model.__class__.__module__ == "scOT.model"
    if poseidon:
        from .poseidon import PoseidonQuantModel
        from .poseidon import QuantScOTLayer, QuantConvNeXtBlock, QuantResNetBlock
        poseidon_wrapper = PoseidonQuantModel(model, weight_parameters, activation_parameters).to(device).eval()
        qnn = _QuantStateModel(poseidon_wrapper.model).to(device).eval()
        block_types = (QuantScOTLayer, QuantConvNeXtBlock, QuantResNetBlock)
        blocks = [(name, module) for name, module in qnn.model.named_modules() if isinstance(module, block_types)]
    else:
        for name, layer in layers.items():
            model = _replace(model, name, QuantModule(layer, weight_parameters, activation_parameters))
        qnn = _QuantStateModel(model).to(device).eval()
        blocks = []
    quant_modules = {name: module for name, module in qnn.model.named_modules() if isinstance(module, QuantModule)}
    if not quant_modules:
        raise ValueError("BRECQ found no quantizable operations")
    # Initialize every weight quantizer, including layers outside reconstructed
    # blocks, directly from weights; no old scales/checkpoints are consumed.
    with torch.no_grad():
        for module in progress(quant_modules.values(), desc="BRECQ initialize weight scales", unit="layer"):
            _initialize_brecq_weight_quantizer(module)
    covered = {id(module) for _, block in blocks for module in block.modules() if isinstance(module, QuantModule)}
    blocks.extend((name, module) for name, module in quant_modules.items() if id(module) not in covered)
    reconstruction_batch_size = int(getattr(args, "rec_batchsize", 32))
    qnn.requires_grad_(False)
    cached_batches = 0
    block_progress = progress(blocks, desc="BRECQ reconstruct blocks", unit="block")
    for name, block in block_progress:
        block_progress.set_postfix(block=name, refresh=False)
        qnn.eval()
        with tempfile.TemporaryDirectory(prefix="brecq-block-", dir=directory) as block_directory:
            cached_batches = _reconstruct_brecq_block(
                qnn, block, calibration, iterations, reconstruction_batch_size, block_directory, device)
    qnn.eval()
    state = {}
    original_modules = dict(original.named_modules())
    with torch.no_grad():
        for name, module in progress(quant_modules.items(), desc="BRECQ materialize weights", unit="layer"):
            quantizer = module.weight_quantizer
            if hasattr(quantizer, "soft_targets"):
                quantizer.soft_targets = False
            quantized_weight = quantizer(module.weight).detach()
            target = original_modules[name]
            target.weight.copy_(quantized_weight)
            state[name] = {"scale": quantizer.delta.detach().cpu(),
                           "zero_point": quantizer.zero_point.detach().cpu(),
                           "rounding_alpha": getattr(quantizer, "alpha", None)}
            if state[name]["rounding_alpha"] is not None:
                state[name]["rounding_alpha"] = state[name]["rounding_alpha"].detach().cpu()
        # Verify the saved, ordinary model reproduces the reconstructed
        # wrapper on calibration input before discarding transient modules.
        qnn.set_quant_state(True, False)
        example = next(iter(calibration))
        wrapped_prediction = forward_model(qnn, example, device)
        materialized_prediction = forward_model(original.eval(), example, device)
        difference = (wrapped_prediction - materialized_prediction).abs().amax()
        if not torch.allclose(wrapped_prediction, materialized_prediction, atol=1e-6, rtol=1e-5):
            raise RuntimeError(f"BRECQ weight materialization changed predictions (max difference {float(difference):.3g})")
    return original.eval(), state, {"reconstruction_blocks": [name for name, _ in blocks],
                                    "reconstruction_iterations": iterations,
                                    "reconstruction_batch_size": reconstruction_batch_size,
                                    "reconstruction_cached_batches": cached_batches,
                                    "calibration_cache": "disk-backed complete calibration and one block at a time",
                                    "reconstruction_sampling": "uniform sample index mapped to its cached batch (batch-size-proportional sampling)",
                                    "materialization_max_abs_difference": float(difference),
                                    "quantized_layers": list(quant_modules),
                                    "scope": "Poseidon Conv2d and Linear, including conditional norm projections" if poseidon else "selected Linear"}



def quantize(model, calibration_loader, args, device, work_dir):
    weight_bits = _bits(getattr(args, "weight_bits", 8), "weight_bits")
    activation_bits = _bits(getattr(args, "activation_bits", 32), "activation_bits")
    if activation_bits != 32 or not 2 <= weight_bits <= 8:
        raise ValueError("BRECQ supports weight_bits=2..8 and activation_bits=32")
    model, layers, weight_bits, activation_bits, metadata = prepare_method(model, args, device, "brecq")
    with calibration_cache(calibration_loader, work_dir) as (calibration, directory):
        record_calibration(metadata, calibration)
        model, state, details = _brecq(model, layers, calibration, args, weight_bits, activation_bits, device, directory)
    metadata.update(details)
    metadata["algorithm"] = "BRECQ block reconstruction with AdaRound; materialized QDQ weights"
    return QuantizationResult(model.eval(), metadata, state)
