"""Physics simulation: batched RK4 + Thomas-BMT + EDM + variational sensitivities."""

from .physics_core import batch_simulate
from .quality_gates import QualityGates, track_to_mlflow

__all__ = ["batch_simulate", "QualityGates", "track_to_mlflow"]
