"""Network-wise SBPQ for SafeSVD's VICON model and calibration protocol."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from SBPQ.beta_parameters import build_beta_parameters, save_beta_parameters
from SBPQ.resume import atomic_save, fingerprint
from SBPQ.storage import DiskSequence
from SBPQ.trainer import SBPQTrainer, freeze_model_parameters
from quantization.cache import PersistentCalibration
from quantization.checkpoints import file_sha256, resolve_resume
from quantization.common import QuantizationResult
from quantization.costs import phase
from quantization.progress import progress, progress_write
from .blocks import build_block_mapping
from .model_utils import expose_attention, model_forward
from .objectives import ViconNetworkLikelihood, compute_sensitivity, query_fields


def implementation_identity(args):
    local = Path(__file__).resolve().parents[2]
    external = Path(args.safe_svd_root)
    paths = [*sorted((local / "SBPQ/vicon").glob("*.py")),
             local / "SBPQ/beta_parameters.py", local / "SBPQ/poseidon/beta_parameter_builder.py",
             local / "SBPQ/noise.py", local / "SBPQ/trainer.py", local / "SBPQ/step_sizes.py",
             *sorted((external / "VICON/src").glob("*.py")),
             *sorted((external / "safe_svd/helper/collector/vicon").glob("*.py"))]
    return {str(path.relative_to(local)) if path.is_relative_to(local)
            else "SafeSVD/" + str(path.relative_to(external)): file_sha256(path) for path in paths}


def quantize(model, calibration_loader, args, device, work_dir):
    # The cropped research model writes position embeddings into tensor slices.
    # SafeSVD's non-inplace context keeps differentiation valid and is restored
    # before the final model is serialized. No local bound methods are saved.
    from safe_svd.helper.collector.vicon.vicon_fisher_forward import _patched_forward
    with _patched_forward(model):
        return _quantize(model, calibration_loader, args, device, work_dir)


def _quantize(model, calibration_loader, args, device, work_dir):
    model.to(device).eval()
    expose_attention(model)
    torch.set_float32_matmul_precision("high")
    destination = Path(args.save_path)
    checkpoint = destination / "sbpq_checkpoint.pt"
    if checkpoint.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"SBPQ checkpoint exists: {checkpoint}; use --resume or a new save_path")
    blocks, layers, mapping, counts, channel_weights = build_block_mapping(model)
    progress_write(f"[INFO] VICON: {len(layers)} candidate Linear layers in {len(blocks)} blocks.")
    with phase("identity_verification"):
        implementation = implementation_identity(args)
        identity = {"version": 1, "model": fingerprint(model.state_dict()), "implementation": implementation,
                    "torch_version": str(torch.__version__), "device_type": torch.device(device).type,
                    "matmul_precision": torch.get_float32_matmul_precision()}
    def clean_record(batch):
        with torch.no_grad():
            return {"clean": query_fields(model_forward(model, batch, device), batch).detach().cpu()}
    with phase("fp_output_cache"):
        snapshot = PersistentCalibration(calibration_loader, Path(work_dir) / "sbpq-calibration", identity,
                                         build_record=clean_record)
    progress_write(f"[INFO] VICON calibration cached: {snapshot.samples} samples in {len(snapshot)} batches; "
                   f"categories={args._calibration_metadata.get('category_samples')}")
    batches = DiskSequence(snapshot.paths, key="batch")
    batches.batch_sizes = snapshot.batch_sizes
    outputs = DiskSequence(snapshot.paths, key="clean")
    preparation_identity = {
        **identity, "calibration": snapshot.digest, "layer_to_block": mapping,
        "calibration_metadata": fingerprint(args._calibration_metadata),
        "loss": {key: getattr(args, key) for key in ("percentile", "sobolev_order", "sobolev_order_weights",
                  "sobolev_norm", "sobolev_transpose", "task_loss_weight", "sobolev_loss_weight")},
    }
    preparation_path = destination / "sbpq_preparation.pt"
    with phase("preparation"):
        prepared = torch.load(preparation_path, map_location="cpu", weights_only=True) if preparation_path.exists() else None
        if prepared and prepared["identity"] != preparation_identity:
            raise ValueError("VICON SBPQ preparation changed (model/data/source/loss); use a new save_path")
        if prepared is None:
            with phase("range_estimation"), torch.no_grad():
                ranges = {}
                for name, layer in progress(layers.items(), desc="VICON weight ranges", unit="layer"):
                    weight = layer.weight.detach().float()
                    value = (torch.quantile(weight, 1 - args.percentile, dim=1)
                             - torch.quantile(weight, args.percentile, dim=1)).clamp_min(1e-12)
                    ranges[name] = {"weight_ranges": value.cpu()}
            with phase("sensitivity"):
                raw = compute_sensitivity(model, batches, blocks, args, device)
            prepared = {"identity": preparation_identity, "ranges": ranges, "raw_sensitivity": raw}
            atomic_save(prepared, preparation_path)
        else:
            progress_write("[INFO] Reusing content-verified VICON ranges and sensitivity.")
        beta = build_beta_parameters(
            prepared["raw_sensitivity"], args.minimum_bits, args.maximum_bits, args.target_bits,
            args.delta_bits, args.beta_kappa, block_parameter_counts=counts,
            mean_epsilon=args.beta_epsilon, relative_epsilon=args.beta_relative_epsilon,
        )
        beta_path = destination / "sbpq_beta_parameters.pt"
        save_beta_parameters(beta, beta_path, metadata={"model": "vicon", "identity": fingerprint(preparation_identity)})
    freeze_model_parameters(model)
    with phase("optimizer_setup"):
        trainer = SBPQTrainer(
            model, batches, outputs, prepared["ranges"], mapping, beta_path,
            initial_bits=args.init_bits, minimum_bits=args.minimum_bits, maximum_bits=args.maximum_bits,
            learning_rate=args.learning_rate, num_mc_samples=args.mc_samples, eta=args.eta,
            prior_scale=args.prior_scale, likelihood_scale=args.likelihood_scale,
            beta_boundary_epsilon=args.beta_epsilon, weight_decay=0.0, gradient_clip_norm=None,
            channel_weights=channel_weights, device=device, likelihood_factory=ViconNetworkLikelihood,
        )
    resume_metadata = {"protocol": "safe-svd-vicon-sbpq-v1", "preparation_identity": fingerprint(preparation_identity)}
    if args.resume:
        _, resume_path = resolve_resume(args.resume, "sbpq")
        if resume_path:
            trainer.load(resume_path, expected_metadata=resume_metadata)
    if args.total_steps < len(batches):
        progress_write(f"[INFO] {len(batches)} calibration batches; {args.total_steps} optimization steps use a prefix. "
                       "Sensitivity uses every selected batch.")
    with phase("optimization"):
        trainer.optimize(args.total_steps, checkpoint_path=checkpoint, save_every=args.save_every,
                         checkpoint_metadata={**resume_metadata, "target_total_steps": args.total_steps})
    trainer.save(destination / "sbpq_trainer_state.pt", metadata=resume_metadata)
    (destination / "history.json").write_text(json.dumps(trainer.history, indent=2) + "\n")
    steps = {name: value.detach().cpu().clone() for name, value in trainer.get_step_sizes().items()}
    average = trainer.calculate_average_bitwidth(parameter_weighted=True)
    with phase("materialization"), torch.no_grad():
        for name, layer in layers.items():
            step = steps[name].to(layer.weight).reshape(-1, 1).clamp_min(1e-12)
            layer.weight.copy_(torch.round(layer.weight / step) * step)
    details = {
        "method": "sbpq", "model_family": "vicon", "weight_only": True, "activation_bits": 32,
        "representation": "materialized_float_fake_quantized_weights", "packed_integer_weights": False,
        "scope": "pre_proj, post_proj and all encoder Q/K/V/out/MLP Linears", "layer_names": list(layers),
        "num_layers": len(layers), "num_blocks": len(blocks), "avg_weight_bits": average,
        "initial_bits": args.init_bits, "target_bits": args.target_bits,
        "calibration_samples": snapshot.samples, "calibration_batches": len(batches),
        "calibration_storage": "persistent_verified_disk", "sensitivity_batches": len(batches),
        "completed_steps": len(trainer.history), "optimization_unique_batches": min(len(trainer.history), len(batches)),
        "optimization_batch_schedule": "cyclic_category_source_order",
        "likelihood": "masked_query_mc_log_mean_exp_output_sum_squared",
        "prior": "parameter_weighted_centered_block_beta", "sobolev_order": args.sobolev_order,
        "sobolev_units": "denormalized_physical_fields", "evaluated_test_data": False,
    }
    state = {"steps": steps, "ranges": prepared["ranges"], "sensitivity_raw": prepared["raw_sensitivity"],
             "layer_to_block": mapping, "history": list(trainer.history), "training_signature": trainer.training_signature}
    return QuantizationResult(model, details, state)
