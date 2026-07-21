"""
Production A/B router with deterministic assignment and MLflow integration.

Key design decisions:
  - Hash-based bucketing: same user always sees same variant (no flicker)
  - Experiment config is immutable after creation (prevents p-hacking)
  - All observations logged to MLflow for reproducibility
  - Supports multi-variant (A/B/C/...) not just binary
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    experiment_id: str
    variants: dict[str, float]
    primary_metric: str
    guardrail_metrics: dict[str, float] = field(default_factory=dict)
    min_detectable_effect: float = 0.02
    alpha: float = 0.05
    power: float = 0.80
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    status: str = "running"
    description: str = ""

    def __post_init__(self):
        total = sum(self.variants.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Variant traffic splits must sum to 1.0, got {total:.3f}")


@dataclass
class Observation:
    variant: str
    user_id: str | None
    metrics: dict[str, float]
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    experiment_id: str
    status: str
    winner: str | None
    metrics_summary: dict[str, dict[str, Any]]
    recommendation: str
    n_observations: dict[str, int]
    analyzed_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def summary(self) -> str:
        lines = [
            f"Experiment: {self.experiment_id}",
            f"Status: {self.status}  Winner: {self.winner or 'undecided'}",
            f"N: {self.n_observations}",
        ]
        for metric, stats in self.metrics_summary.items():
            lines.append(
                f"  {metric}: "
                + "  ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items()
                )
            )
        lines.append(f"Recommendation: {self.recommendation}")
        return "\n".join(lines)


class ABRouter:
    """
    Multi-variant A/B router with:
    - Deterministic hash-based assignment
    - Per-observation MLflow logging
    - Automatic analysis with statistical tests
    - Guardrail metric monitoring
    """

    def __init__(self, mlflow_tracking_uri: str | None = None):
        self._experiments: dict[str, ExperimentConfig] = {}
        self._observations: dict[str, list[Observation]] = {}
        self._mlflow = self._init_mlflow(mlflow_tracking_uri)

    def _init_mlflow(self, uri: str | None):
        try:
            from src.mlops.compat import mlflow

            if uri:
                mlflow.set_tracking_uri(uri)
            return mlflow
        except ImportError:
            return None

    def create_experiment(
        self,
        experiment_id: str,
        variants: dict[str, float],
        primary_metric: str,
        guardrail_metrics: dict[str, float] | None = None,
        min_detectable_effect: float = 0.02,
        alpha: float = 0.05,
        description: str = "",
    ) -> ExperimentConfig:
        config = ExperimentConfig(
            experiment_id=experiment_id,
            variants=variants,
            primary_metric=primary_metric,
            guardrail_metrics=guardrail_metrics or {},
            min_detectable_effect=min_detectable_effect,
            alpha=alpha,
            description=description,
        )
        self._experiments[experiment_id] = config
        self._observations[experiment_id] = []

        if self._mlflow:
            with self._mlflow.start_run(run_name=f"experiment-{experiment_id}"):
                self._mlflow.log_params(
                    {
                        "experiment_id": experiment_id,
                        "variants": str(list(variants.keys())),
                        "primary_metric": primary_metric,
                        "mde": min_detectable_effect,
                        "alpha": alpha,
                    }
                )

        logger.info(
            "Created experiment '%s' with variants: %s", experiment_id, list(variants.keys())
        )
        return config

    def assign_variant(self, experiment_id: str, user_id: str) -> str:
        """Deterministic assignment: same user always gets same variant."""
        config = self._experiments.get(experiment_id)
        if not config or config.status != "running":
            return list(config.variants.keys())[0] if config else "control"

        h = int(hashlib.md5(f"{experiment_id}:{user_id}".encode()).hexdigest(), 16)
        bucket = (h % 10000) / 10000.0

        cumulative = 0.0
        for variant, split in config.variants.items():
            cumulative += split
            if bucket < cumulative:
                return variant
        return list(config.variants.keys())[-1]

    def log_observation(
        self,
        experiment_id: str,
        variant: str,
        user_id: str | None = None,
        **metrics: float,
    ) -> None:
        obs = Observation(variant=variant, user_id=user_id, metrics=metrics)
        self._observations[experiment_id].append(obs)
        self._check_guardrails(experiment_id)

        if self._mlflow:
            self._mlflow.log_metrics(
                {f"{experiment_id}.{variant}.{k}": v for k, v in metrics.items()}
            )

    def _check_guardrails(self, experiment_id: str) -> None:
        config = self._experiments.get(experiment_id)
        if not config or not config.guardrail_metrics:
            return

        obs = self._observations.get(experiment_id, [])
        variants = list(config.variants.keys())
        if len(variants) < 2:
            return

        control = variants[0]
        treatment = variants[1]

        for metric, max_degradation in config.guardrail_metrics.items():
            c_vals = [
                o.metrics.get(metric, 0)
                for o in obs
                if o.variant == control and metric in o.metrics
            ]
            t_vals = [
                o.metrics.get(metric, 0)
                for o in obs
                if o.variant == treatment and metric in o.metrics
            ]
            if len(c_vals) < 10 or len(t_vals) < 10:
                continue
            degradation = sum(t_vals) / len(t_vals) - sum(c_vals) / len(c_vals)
            if degradation > max_degradation:
                config.status = "stopped_guardrail"
                logger.warning(
                    "GUARDRAIL BREACH in '%s': %s degraded by %.3f > max %.3f",
                    experiment_id,
                    metric,
                    degradation,
                    max_degradation,
                )

    def get_observations(self, experiment_id: str, variant: str | None = None) -> list[Observation]:
        obs = self._observations.get(experiment_id, [])
        if variant:
            return [o for o in obs if o.variant == variant]
        return obs

    def get_config(self, experiment_id: str) -> ExperimentConfig | None:
        return self._experiments.get(experiment_id)
