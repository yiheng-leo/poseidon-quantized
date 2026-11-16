"""VICON source and calibration argument resolution."""
import math
from pathlib import Path


def validate_args(args):
    if args.method != "sbpq":
        raise ValueError("--model vicon currently supports --method sbpq")
    if args.calibration_num < args.batch_size or args.calibration_seed < 0:
        raise ValueError("VICON requires calibration_num >= batch_size and nonnegative calibration_seed")
    if not isinstance(args.config_override, list) or not all(isinstance(item, str) for item in args.config_override):
        raise ValueError("VICON config_override must be a list of Hydra override strings")
    root = Path(__file__).resolve().parents[2]
    args.model_name, args.dataset = "VICON", "vicon"
    if args.model_path is None:
        args.model_path = str(root / "models/VICON/model.pth")
    if args.data_path in (None, "poseidon/camlab-ethz/down_streams"):
        args.data_path = str(root / "dataset/VICON")
    args.model_path = str(Path(args.model_path).expanduser().resolve())
    args.data_path = str(Path(args.data_path).expanduser().resolve())
    bundled = Path(__file__).resolve().parent / "configs"
    args.config_dir = str(Path(args.config_dir).expanduser().resolve() if args.config_dir else bundled)
    if args.vicon_file_order != "filesystem":
        args.vicon_file_order = str(Path(args.vicon_file_order).expanduser().resolve() if args.vicon_file_order
                                    else bundled / "research_calibration_order.json")


def validate_settings(args):
    """The SBPQ bounds/weights contract, without importing the ScOT runtime."""
    if args.total_steps < 1 or args.save_every < 1:
        raise ValueError("SBPQ total_steps/save_every must be positive")
    if not 0 < args.minimum_bits < args.maximum_bits <= 30:
        raise ValueError("SBPQ requires 0 < minimum_bits < maximum_bits <= 30")
    if not args.minimum_bits <= args.init_bits <= args.maximum_bits:
        raise ValueError("SBPQ init_bits must be inside minimum_bits..maximum_bits")
    if not args.minimum_bits < args.target_bits < args.maximum_bits:
        raise ValueError("SBPQ target_bits must lie strictly inside the bit bounds")
    if args.beta_kappa <= 0 or args.delta_bits < 0 or args.likelihood_scale < 0:
        raise ValueError("SBPQ requires positive beta_kappa and nonnegative delta_bits/likelihood_scale")
    if not 0 < args.beta_epsilon < 0.5 or args.beta_relative_epsilon <= 0:
        raise ValueError("Invalid SBPQ Beta epsilon")
    if min(args.task_loss_weight, args.sobolev_loss_weight) < 0 or args.task_loss_weight + args.sobolev_loss_weight == 0:
        raise ValueError("SBPQ sensitivity weights must be nonnegative and not all zero")
    if not 0 <= args.sobolev_order <= 8:
        raise ValueError("SBPQ Sobolev order must be 0..8")
    weights = args.sobolev_order_weights
    if isinstance(weights, str):
        weights = [float(value.strip()) for value in weights.split(",")]
        weights = weights[0] if len(weights) == 1 else weights
    if isinstance(weights, (tuple, list)):
        if len(weights) != args.sobolev_order + 1:
            raise ValueError("sobolev_order_weights must contain exactly order + 1 weights")
        weights = tuple(float(value) for value in weights)
        valid = all(math.isfinite(value) and value >= 0 for value in weights)
    else:
        valid = not isinstance(weights, bool) and math.isfinite(float(weights)) and float(weights) >= 0
        weights = float(weights)
    if not valid:
        raise ValueError("Sobolev weights must be finite and nonnegative")
    args.sobolev_order_weights = weights
