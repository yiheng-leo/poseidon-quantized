"""Parallel calibration of the Poseidon PTQ4ViT Linear adaptation."""
from __future__ import annotations
import tempfile
from quantization.common import QuantizationResult
from quantization.progress import progress
from quantization.layers import QuantizedLinear, _replace, prepare_method
from quantization.cache import calibration_cache, record_calibration
from .hessian import _cache_ptq_layer
from .search import _ptq_weight, _ptq_search

def _ptq4vit(model, layers, calibration, weight_bits, activation_bits, device, directory, args):
    rounds = int(getattr(args, "ptq_search_rounds", 3))
    candidates = int(getattr(args, "ptq_candidates", 100))
    if rounds < 1 or candidates < 2:
        raise ValueError("PTQ4ViT requires at least one search round and two candidate intervals")
    state, replacements, twin_layers = {}, {}, []
    # Collect and optimize against the unchanged FP network (parallel PTQ),
    # then apply all replacements together.
    layer_progress = progress(layers.items(), desc="PTQ4ViT quantize layers", unit="layer")
    for name, layer in layer_progress:
        layer_progress.set_postfix(layer=name, refresh=False)
        with tempfile.TemporaryDirectory(prefix="ptq-layer-", dir=directory) as layer_directory:
            paths, twin = _cache_ptq_layer(model, name, layer, calibration, layer_directory, device)
            ws, activation_scale, negative_scale, energy = _ptq_search(
                layer, paths, weight_bits, activation_bits, twin, device, rounds, candidates)
        if twin:
            twin_layers.append(name)
        replacements[name] = QuantizedLinear(layer, _ptq_weight(layer.weight, ws, weight_bits).detach(),
                                            activation_bits, "twin" if twin else "symmetric",
                                            scale=activation_scale, negative_scale=negative_scale,
                                            signed_full_range=True)
        state[name] = {"weight_scale": ws.cpu(), "activation_scale": activation_scale.cpu(),
                       "negative_activation_scale": negative_scale.cpu(), "twin_uniform": twin,
                       "calibration_gradient_energy": energy}
    for name, replacement in progress(replacements.items(), desc="PTQ4ViT materialize layers", unit="layer"):
        model = _replace(model, name, replacement)
    return model, state, {"search_rounds": rounds, "candidate_intervals": candidates,
                          "twin_uniform_layers": twin_layers,
                          "sensitivity_metric": "squared regression-loss gradient weighted output reconstruction"}



def quantize(model, calibration_loader, args, device, work_dir):
    model, layers, weight_bits, activation_bits, metadata = prepare_method(model, args, device, "ptq4vit")
    with calibration_cache(calibration_loader, work_dir) as (calibration, directory):
        record_calibration(metadata, calibration)
        model, state, details = _ptq4vit(model, layers, calibration, weight_bits, activation_bits, device, directory, args)
    metadata.update(details)
    metadata.update(algorithm="PTQ4ViT Linear adaptation: Hessian-guided search and post-GELU twin uniform",
                    attention_matmul_quantized=False, softmax_quantized=False)
    return QuantizationResult(model.eval(), metadata, state)
