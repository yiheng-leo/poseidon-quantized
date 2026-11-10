"""NS-PwC Table 3 optimization recipe; calibration is configured independently."""
DEFAULTS = {
    "init_bits": 8, "target_bits": 8.0, "mc_samples": 10,
    "learning_rate": 3.75e-5, "eta": 1e-6, "prior_scale": 1.0,
    "beta_kappa": 150.0, "delta_bits": 2.0, "minimum_bits": 1.0,
    "maximum_bits": 16.0, "total_steps": 800, "seed": 42,
    "sobolev_order": 2, "sobolev_order_weights": "1", "sobolev_norm": "l1",
    "task_loss_weight": 1.0, "sobolev_loss_weight": 1.0,
    "likelihood_scale": 1.0, "percentile": 1e-4, "activation_bits": 32,
}
