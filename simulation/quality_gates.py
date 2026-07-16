"""Quality gates + MLflow tracking for physics simulations."""

import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class QualityGates:
    """Physics simulation quality validation."""

    ENERGY_DRIFT_THRESHOLD = 0.01  # < 0.01%
    SPIN_NORM_DRIFT_THRESHOLD = 0.005  # < 0.005%
    MU_DEVIATION_THRESHOLD = 0.01  # < 1% magnetic moment deviation

    @staticmethod
    def validate(diagnostics: Dict) -> Tuple[bool, str, float]:
        """
        Validate physics simulation quality.

        Args:
            diagnostics: From batch_simulate() output

        Returns:
            (gates_passed, reason, quality_score)
        """
        energy_drift = diagnostics.get("energy_drift_percent", 0.0)
        spin_norm_drift = diagnostics.get("spin_norm_drift_percent", 0.0)
        mu_max_rel_dev = diagnostics.get("mu_max_rel_dev", 0.0)

        failures = []

        if energy_drift > QualityGates.ENERGY_DRIFT_THRESHOLD:
            failures.append(f"Energy drift {energy_drift:.4f}% > {QualityGates.ENERGY_DRIFT_THRESHOLD}%")

        if spin_norm_drift > QualityGates.SPIN_NORM_DRIFT_THRESHOLD:
            failures.append(f"Spin norm drift {spin_norm_drift:.6f}% > {QualityGates.SPIN_NORM_DRIFT_THRESHOLD}%")

        if mu_max_rel_dev > QualityGates.MU_DEVIATION_THRESHOLD:
            failures.append(f"Magnetic moment deviation {mu_max_rel_dev:.4f} > {QualityGates.MU_DEVIATION_THRESHOLD}")

        gates_passed = len(failures) == 0
        reason = "; ".join(failures) if failures else "All gates passed"
        quality_score = 1.0 if gates_passed else 0.0

        logger.info(f"Quality gates: {reason}")
        return gates_passed, reason, quality_score


def track_to_mlflow(run_id: str, results: Dict, hardware_info: Dict, wall_clock_time: float):
    """Log simulation results to MLflow."""
    try:
        import mlflow
    except ImportError:
        logger.warning("MLflow not available, skipping logging")
        return

    with mlflow.start_run(run_name=f"physics-sim-{run_id}"):
        # Parameters
        mlflow.log_param("batch_size", results.get("batch_size", 0))
        mlflow.log_param("num_steps", results.get("num_steps", 0))
        mlflow.log_param("hardware_used", hardware_info.get("device_name", "unknown"))
        mlflow.log_param("device_type", "cuda" if hardware_info.get("cuda_available") else "cpu")

        # Metrics
        mlflow.log_metric("physics_quality_score", results.get("quality_score", 0.0))
        mlflow.log_metric(
            "energy_drift_percent",
            results.get("diagnostics", {}).get("energy_drift_percent", 0.0),
        )
        mlflow.log_metric(
            "spin_norm_drift_percent",
            results.get("diagnostics", {}).get("spin_norm_drift_percent", 0.0),
        )
        mlflow.log_metric(
            "mu_max_rel_dev",
            results.get("diagnostics", {}).get("mu_max_rel_dev", 0.0),
        )
        mlflow.log_metric("wall_clock_time_sec", wall_clock_time)

        # Estimated cost (NVIDIA A100: ~$2/hr)
        estimated_cost = (wall_clock_time / 3600.0) * 2.0
        mlflow.log_metric("estimated_cost_usd", estimated_cost)

        # Gates passed
        gates_passed = results.get("diagnostics", {}).get("gates_passed", False)
        mlflow.log_param("gates_passed", str(gates_passed))

        logger.info(f"Logged simulation {run_id} to MLflow")
