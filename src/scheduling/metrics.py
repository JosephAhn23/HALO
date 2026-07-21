"""Scheduling metrics and observability for fleet dispatch plans."""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class SchedulingMetrics:
    """Extract metrics from reconciliation output for MLflow + Prometheus."""

    @staticmethod
    def extract_metrics(plan: dict[str, Any]) -> dict[str, float]:
        """
        Extract quantitative metrics from a dispatch plan.

        Args:
            plan: Output from FleetReconciler.reconcile()

        Returns:
            dict with metric_name -> value for MLflow logging
        """
        summary = plan.get("summary", {})
        dispatches = plan.get("dispatches", [])
        terminal = plan.get("terminal", [])

        # Count terminal states
        state_counts = {}
        for t in terminal:
            state = t.get("state", "unknown")
            state_counts[state] = state_counts.get(state, 0) + 1

        # Dispatch latency: time between release and dispatch
        latencies = []
        for d in dispatches:
            dispatch_time = datetime.fromisoformat(d["time"].replace("Z", "+00:00"))
            # Occurrence ID is "job-id@release-time"
            occ = d.get("occurrence", "")
            if "@" in occ:
                release_time_str = occ.split("@")[1]
                release_time = datetime.fromisoformat(release_time_str.replace("Z", "+00:00"))
                latency_sec = (dispatch_time - release_time).total_seconds()
                if latency_sec >= 0:
                    latencies.append(latency_sec)

        metrics = {
            "scheduling_dispatch_count": float(summary.get("dispatch_count", 0)),
            "scheduling_succeeded": float(summary.get("succeeded", 0)),
            "scheduling_failed": float(summary.get("failed", 0)),
            "scheduling_missed": float(summary.get("missed", 0)),
            "scheduling_blocked": float(summary.get("blocked", 0)),
            "scheduling_coalesced": float(summary.get("coalesced", 0)),
            "scheduling_unfinished": float(summary.get("unfinished", 0)),
            "scheduling_success_rate": (
                summary.get("succeeded", 0) / (summary.get("dispatch_count", 1))
                if summary.get("dispatch_count", 0) > 0
                else 0.0
            ),
        }

        if latencies:
            metrics["scheduling_dispatch_latency_mean_sec"] = sum(latencies) / len(latencies)
            metrics["scheduling_dispatch_latency_max_sec"] = max(latencies)
            metrics["scheduling_dispatch_latency_p95_sec"] = sorted(latencies)[
                int(0.95 * len(latencies))
            ]

        return metrics

    @staticmethod
    def dispatch_quality_score(plan: dict[str, Any]) -> float:
        """
        Score dispatch plan quality: 1.0 if all jobs succeeded, 0.0 if any failed or missed.

        Args:
            plan: Output from FleetReconciler.reconcile()

        Returns:
            float in [0.0, 1.0]
        """
        summary = plan.get("summary", {})

        total = (
            summary.get("succeeded", 0)
            + summary.get("failed", 0)
            + summary.get("missed", 0)
            + summary.get("blocked", 0)
        )

        if total == 0:
            return 1.0  # No jobs = success

        # Only count "succeeded" as full quality
        succeeded = summary.get("succeeded", 0)
        return min(1.0, succeeded / total)


def track_dispatch_plan(run_id: str, plan: dict[str, Any], window_start: str, window_end: str):
    """
    Log a dispatch plan to MLflow.

    Args:
        run_id: Unique run identifier
        plan: Output from FleetReconciler.reconcile()
        window_start: Window start timestamp (for context)
        window_end: Window end timestamp (for context)
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("MLflow not available, skipping scheduling metrics")
        return

    metrics = SchedulingMetrics.extract_metrics(plan)
    quality = SchedulingMetrics.dispatch_quality_score(plan)

    with mlflow.start_run(run_name=f"scheduling-{run_id}"):
        # Log parameters
        mlflow.log_param("window_start", window_start)
        mlflow.log_param("window_end", window_end)

        # Log metrics
        mlflow.log_metric("scheduling_quality_score", quality)
        mlflow.log_metrics(metrics)

        # Log plan summary
        summary = plan.get("summary", {})
        mlflow.log_dict(summary, "dispatch_summary.json")

        # Set tags
        mlflow.set_tag("scheduling_component", "fleet_reconciler")

        logger.info(f"Logged scheduling plan {run_id} to MLflow: quality={quality:.3f}")
