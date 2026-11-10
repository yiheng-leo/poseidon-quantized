"""ptq4vit calibration, quantization and standalone artifact entry."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantization.runner import method_main

if __name__ == "__main__":
    method_main("ptq4vit")
