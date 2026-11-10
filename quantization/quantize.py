"""Compatibility entry; prefer python -m <Method>.<method>."""
from .runner import METHODS, PROBABILISTIC_METHODS, build_parser, validate_args, run_quantization, main

if __name__ == "__main__":
    main()
