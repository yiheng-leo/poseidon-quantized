"""Quantize Poseidon with SBPQ; evaluation is a separate command."""
from quantization.runner import method_main

if __name__ == "__main__":
    method_main("sbpq")
