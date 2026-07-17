"""Fleet scheduler: wraps dynamo reconciliation for HALO integration."""

import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import sys

logger = logging.getLogger(__name__)


class FleetReconciler:
    """Wrapper around the dynamo fleet timer reconciler."""

    def __init__(self, reconcile_script_path: Optional[str] = None):
        """
        Initialize the reconciler.

        Args:
            reconcile_script_path: Path to reconcile.py. If None, uses bundled version.
        """
        if reconcile_script_path is None:
            # Use the bundled dynamo reconciler
            base = Path(__file__).parent.parent
            reconcile_script_path = str(
                base / "dynamo-8b56404-systems-infrastructure-and-operations/task/solution/reconcile.py"
            )

        self.reconcile_script = Path(reconcile_script_path)
        if not self.reconcile_script.exists():
            raise FileNotFoundError(f"Reconcile script not found: {reconcile_script_path}")

    def reconcile(self, fleet_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the fleet reconciler on a config.

        Args:
            fleet_config: Fleet configuration dict with workers, jobs, window, failed_attempts

        Returns:
            dict with dispatches, terminal, summary
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_path = tmpdir / "input.json"
            output_path = tmpdir / "output.json"

            # Write input
            with open(input_path, "w") as f:
                json.dump(fleet_config, f)

            # Run reconciler
            try:
                result = subprocess.run(
                    [sys.executable, str(self.reconcile_script), str(input_path), str(output_path)],
                    capture_output=True,
                    timeout=60,
                    text=True,
                )

                if result.returncode != 0:
                    logger.error(f"Reconciler failed: {result.stderr}")
                    raise RuntimeError(f"Reconciler error: {result.stderr}")

                # Read output
                with open(output_path) as f:
                    output = json.load(f)

                logger.info(
                    f"Reconciliation complete: {output['summary']['dispatch_count']} dispatches, "
                    f"succeeded={output['summary']['succeeded']}, "
                    f"failed={output['summary']['failed']}"
                )

                return output

            except subprocess.TimeoutExpired:
                logger.error("Reconciler timeout (60s)")
                raise RuntimeError("Reconciliation timeout")


def parse_fleet_config(
    workers: list,
    jobs: list,
    window_start: str,
    window_end: str,
    failed_attempts: Optional[list] = None,
) -> Dict[str, Any]:
    """
    Build a fleet config dict for the reconciler.

    Args:
        workers: List of worker dicts (id, cpu, memory, labels, blackouts)
        jobs: List of job dicts (id, timezone, weekdays, times, duration_sec, cpu, memory, labels, priority, etc.)
        window_start: RFC 3339 UTC timestamp (e.g., "2025-01-01T00:00:00Z")
        window_end: RFC 3339 UTC timestamp
        failed_attempts: Optional list of {occurrence, attempt} to simulate failures

    Returns:
        dict ready to pass to FleetReconciler.reconcile()
    """
    return {
        "window": {"start": window_start, "end": window_end},
        "workers": workers,
        "jobs": jobs,
        "failed_attempts": failed_attempts or [],
    }
