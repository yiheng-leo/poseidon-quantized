"""Quantize Poseidon or SafeSVD's VICON with SBPQ; evaluate separately."""
from quantization.runner import method_main

if __name__ == "__main__":
    method_main("sbpq")
