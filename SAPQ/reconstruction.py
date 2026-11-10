"""SAPQ block grouping, asymmetric inputs, and Fisher reconstruction caches."""
import torch
import torch.nn.functional as F
from quantization.common import forward_model
from quantization.progress import progress
from quantization.probabilistic import (
    _Unit, _capture_unit, _cpu, _first_tensor, _load, _reference_hooks,
    _requires_input_grad,
)

_BLOCK_TYPES = {"ScOTLayer", "ConvNeXtBlock", "ResNetBlock"}


def _block_units(model, layers):
    """Preserve ScOT residual/attention blocks; cover non-block Linear layers too."""
    modules = dict(model.named_modules())
    blocks = {name: mod for name, mod in modules.items()
              if type(mod).__name__ in _BLOCK_TYPES}
    groups = {}
    for name, layer in layers.items():
        ancestors = [block for block in blocks if name.startswith(block + ".") or block == ""]
        owner = max(ancestors, key=len) if ancestors else name
        groups.setdefault(owner, {})[name] = layer
    return [_Unit(name, modules[name], groups[name]) for name in modules if name in groups]


def _reference_prefix(model, unit, layers):
    names = list(dict(model.named_modules()))
    end = max(index for index, name in enumerate(names)
              if name == unit.name or name.startswith(unit.name + ".") or unit.name == "")
    return {name: layers[name] for name in names[:end + 1] if name in layers}


def _cache_unit(model, unit, layers, prefix_layers, reference_steps, paths, directory, device, *, block):
    """Keep only one unit's IO; block inputs use a quantized prefix (asym=True)."""
    result = []
    for index, path in enumerate(progress(paths, desc=f"Cache unit IO/Fisher: {unit.name}", leave=False)):
        record = _load(path)
        captured = _capture_unit(model, unit, record["batch"], device)
        if block:
            # The target remains the clean FP block output; inputs reflect an
            # independently quantized prefix, as in original BRECQ asym caching.
            with _reference_hooks(layers, reference_steps):
                quantized = _capture_unit(model, unit, record["batch"], device)
            captured.update(args=quantized["args"], kwargs=quantized["kwargs"])
            output = []
            handle = None
            try:
                with _reference_hooks(prefix_layers, reference_steps):
                    handle = unit.module.register_forward_hook(lambda mod, inputs, value: output.append(_first_tensor(value)))
                    pred = forward_model(model, _requires_input_grad(record["batch"], device), device)
                if len(output) != 1:
                    raise ValueError(f"Expected one execution of {unit.name!r}, got {len(output)}.")
                loss = F.mse_loss(pred, record["clean"].to(device))
                grad = torch.autograd.grad(loss, output[0], allow_unused=True)[0]
                if grad is None:
                    raise ValueError(f"No Fisher gradient reached {unit.name!r}.")
                # save_grad_data's default is abs(gradient) + damping=1.0;
                # the block energy squares this damped quantity, not raw grad.
                captured["fisher"] = _cpu(grad.abs() + 1.0)
            finally:
                if handle is not None:
                    handle.remove()
        local_path = directory / f"unit_{index:06d}.pt"
        torch.save(captured, local_path)
        result.append(local_path)
    return result
