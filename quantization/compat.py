"""Known class-path migrations for trusted full-model artifacts.

Only serialization resolves these historical names. Runtime code imports the
canonical method packages, never archived implementations.
"""
import pickle


_MODULES = {
    "quantization.baselines": "quantization.layers",
    "BRECQ.quant.quant_layer": "BRECQ.quantizers",
    "BRECQ.quant.adaptive_rounding": "BRECQ.quantizers",
    "BRECQ.quant.quant_block": "BRECQ.quantizers",
    "BRECQ.quant.poseidon_quant_model": "BRECQ.poseidon",
    "BRECQ.quant.poseidon_quant_block": "BRECQ.poseidon",
    "BRECQ.quant.vicon_quant_model": "workflows.vicon.BRECQ.quant.vicon_quant_model",
    "BRECQ.quant.vicon_quant_block": "workflows.vicon.BRECQ.quant.vicon_quant_block",
}


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return super().find_class(_MODULES.get(module, module), name)


# torch.load's legacy format reader also uses these standard pickle functions.
load = pickle.load
loads = pickle.loads
