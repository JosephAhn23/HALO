"""Fleet scheduling integration tests."""

import pytest
import json
from pathlib import Path
from src.scheduling import FleetReconciler, parse_fleet_config
from src.scheduling.metrics import SchedulingMetrics


class TestFleetReconciler:
    """Test fleet reconciliation."""

    @pytest.fixture
    def reconciler(self):
        """Initialize reconciler."""
        return FleetReconciler()

    @pytest.fixture
    def sample_config(self):
        """Load sample fleet config."""
        dynamo_dir = Path(__file__).parent.parent / "dynamo-8b56404-systems-infrastructure-and-operations"
        fleet_path = dynamo_dir / "task/environment/data/fleet.json"

        if not fleet_path.exists():
            pytest.skip("Sample fleet config not found")

        with open(fleet_path) as f:
            return json.load(f)

    def test_reconcile_basic(self, reconciler, sample_config):
        """Basic reconciliation with sample config."""
        result = reconciler.reconcile(sample_config)

        assert "dispatches" in result
        assert "terminal" in result
        assert "summary" in result

        summary = result["summary"]
        assert "dispatch_count" in summary
        assert "succeeded" in summary
        assert "failed" in summary

    def test_reconcile_output_structure(self, reconciler, sample_config):
        """Verify output structure matches spec."""
        result = reconciler.reconcile(sample_config)

        # Check dispatch structure
        for dispatch in result["dispatches"]:
            assert "time" in dispatch
            assert "occurrence" in dispatch
            assert "attempt" in dispatch
            assert "worker" in dispatch
            assert "fence" in dispatch

        # Check terminal structure
        for term in result["terminal"]:
            assert "time" in term
            assert "occurrence" in term
            assert "state" in term
            assert "attempts" in term

    def test_parse_fleet_config(self):
        """Test fleet config builder."""
        workers = [
            {"id": "w1", "cpu": 4, "memory": 8, "labels": ["compute"], "blackouts": []}
        ]
        jobs = [
            {
                "id": "job1",
                "timezone": "UTC",
                "weekdays": [0, 1, 2, 3, 4],
                "times": ["09:00"],
                "fold_policy": "first",
                "gap_policy": "skip",
                "duration_sec": 3600,
                "cpu": 2,
                "memory": 4,
                "labels": ["compute"],
                "priority": 100,
                "max_lateness_sec": 0,
                "mutex": None,
                "dependencies": [],
                "max_attempts": 3,
                "retry_delay_sec": 60,
                "coalesce": False,
            }
        ]

        config = parse_fleet_config(
            workers=workers,
            jobs=jobs,
            window_start="2025-01-01T00:00:00Z",
            window_end="2025-01-08T00:00:00Z",
        )

        assert config["window"]["start"] == "2025-01-01T00:00:00Z"
        assert config["window"]["end"] == "2025-01-08T00:00:00Z"
        assert len(config["workers"]) == 1
        assert len(config["jobs"]) == 1


class TestSchedulingMetrics:
    """Test metric extraction."""

    def test_extract_metrics(self):
        """Extract metrics from a plan."""
        plan = {
            "dispatches": [
                {
                    "time": "2025-01-01T09:00:00Z",
                    "occurrence": "job1@2025-01-01T09:00:00Z",
                    "attempt": 1,
                    "worker": "w1",
                    "fence": 1,
                }
            ],
            "terminal": [
                {
                    "time": "2025-01-01T10:00:00Z",
                    "occurrence": "job1@2025-01-01T09:00:00Z",
                    "state": "succeeded",
                    "attempts": 1,
                }
            ],
            "summary": {
                "dispatch_count": 1,
                "succeeded": 1,
                "failed": 0,
                "missed": 0,
                "blocked": 0,
                "coalesced": 0,
                "unfinished": 0,
            },
        }

        metrics = SchedulingMetrics.extract_metrics(plan)

        assert metrics["scheduling_dispatch_count"] == 1.0
        assert metrics["scheduling_succeeded"] == 1.0
        assert metrics["scheduling_failed"] == 0.0
        assert "scheduling_success_rate" in metrics

    def test_quality_score_all_succeeded(self):
        """Quality score should be 1.0 when all succeed."""
        plan = {
            "summary": {
                "dispatch_count": 10,
                "succeeded": 10,
                "failed": 0,
                "missed": 0,
                "blocked": 0,
                "coalesced": 0,
                "unfinished": 0,
            }
        }

        quality = SchedulingMetrics.dispatch_quality_score(plan)
        assert quality == 1.0

    def test_quality_score_partial_failure(self):
        """Quality score should decrease with failures."""
        plan = {
            "summary": {
                "dispatch_count": 10,
                "succeeded": 5,
                "failed": 3,
                "missed": 2,
                "blocked": 0,
                "coalesced": 0,
                "unfinished": 0,
            }
        }

        quality = SchedulingMetrics.dispatch_quality_score(plan)
        assert 0 < quality < 1.0
        assert quality == 5.0 / 10.0

    def test_quality_score_empty_plan(self):
        """Quality score should be 1.0 for empty plan."""
        plan = {"summary": {}}
        quality = SchedulingMetrics.dispatch_quality_score(plan)
        assert quality == 1.0


class TestSchedulingEdgeCases:
    """Test edge cases."""

    def test_empty_workers(self):
        """Should handle empty worker list gracefully."""
        config = parse_fleet_config(
            workers=[],
            jobs=[],
            window_start="2025-01-01T00:00:00Z",
            window_end="2025-01-02T00:00:00Z",
        )

        assert len(config["workers"]) == 0
        assert len(config["jobs"]) == 0

    def test_timezone_aware_scheduling(self):
        """Jobs in different timezones should be handled."""
        jobs = [
            {
                "id": "job_est",
                "timezone": "America/New_York",
                "weekdays": [0],
                "times": ["09:00"],
                "fold_policy": "first",
                "gap_policy": "skip",
                "duration_sec": 3600,
                "cpu": 1,
                "memory": 1,
                "labels": [],
                "priority": 1,
                "max_lateness_sec": 0,
                "mutex": None,
                "dependencies": [],
                "max_attempts": 1,
                "retry_delay_sec": 0,
                "coalesce": False,
            },
            {
                "id": "job_utc",
                "timezone": "UTC",
                "weekdays": [0],
                "times": ["14:00"],
                "fold_policy": "first",
                "gap_policy": "skip",
                "duration_sec": 3600,
                "cpu": 1,
                "memory": 1,
                "labels": [],
                "priority": 1,
                "max_lateness_sec": 0,
                "mutex": None,
                "dependencies": [],
                "max_attempts": 1,
                "retry_delay_sec": 0,
                "coalesce": False,
            },
        ]

        config = parse_fleet_config(
            workers=[{"id": "w1", "cpu": 8, "memory": 16, "labels": [], "blackouts": []}],
            jobs=jobs,
            window_start="2025-01-01T00:00:00Z",
            window_end="2025-01-08T00:00:00Z",
        )

        assert len(config["jobs"]) == 2
