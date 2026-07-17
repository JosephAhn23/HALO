"""Physics simulation integration tests."""

import pytest
import torch
import numpy as np
from src.simulation import batch_simulate, QualityGates, track_to_mlflow
from src.inference.cuda_dispatch.dispatcher import get_hardware_info


class TestBatchSimulate:
    """Test batched physics simulator."""

    def test_batch_simulate_basic(self):
        """Basic simulation with 10 particles."""
        result = batch_simulate(
            batch_size=5,
            num_steps=100,
            params={
                "q": 1.0, "m": 0.938, "c": 299.792458, "anomaly": 1.793,
                "edm_eta": 1e-3, "L": 0.5, "B0_initial": 1.0, "dtau": 1e-3,
                "sensitivity_eps": 1e-6, "shape_sensitivity_eps": 1e-6,
            },
        )

        assert "trajectories" in result
        assert "sensitivity_W" in result
        assert "diagnostics" in result
        assert "quality_score" in result

        trajectories = result["trajectories"]
        assert trajectories.shape == (5, 100, 7)  # (batch, steps, 7-state)

    def test_batch_simulate_single_particle(self):
        """Simulate single particle."""
        result = batch_simulate(
            batch_size=1,
            num_steps=50,
        )

        assert result["trajectories"].shape == (1, 50, 7)
        assert result["quality_score"] >= 0.0
        assert result["quality_score"] <= 1.0

    def test_batch_simulate_with_initial_conditions(self):
        """Simulate with custom initial conditions."""
        # Columns: x, y, z, vx, vy, vz, spin_z (spin seeds along z; batch_simulate
        # normalizes the derived 3D spin vector internally).
        ic = torch.randn(3, 7) * 0.1

        result = batch_simulate(
            batch_size=3,
            num_steps=50,
            initial_conditions=ic,
        )

        assert result["trajectories"].shape == (3, 50, 7)

    def test_batch_simulate_output_shapes(self):
        """Check all output shapes."""
        result = batch_simulate(
            batch_size=2,
            num_steps=100,
        )

        traj = result["trajectories"]
        W = result["sensitivity_W"]
        diag = result["diagnostics"]

        assert traj.shape == (2, 100, 7)
        assert W.shape == (2, 100, 44)

        assert "energy_drift_percent" in diag
        assert "spin_norm_drift_percent" in diag
        assert "gates_passed" in diag


class TestQualityGates:
    """Test quality validation."""

    def test_gates_pass(self):
        """Gates pass with good diagnostics."""
        diagnostics = {
            "energy_drift_percent": 0.001,
            "spin_norm_drift_percent": 0.0001,
            "mu_max_rel_dev": 0.001,
        }

        gates_passed, reason, score = QualityGates.validate(diagnostics)
        assert gates_passed
        assert score == 1.0
        assert "All gates passed" in reason

    def test_gates_fail_energy(self):
        """Gates fail on high energy drift."""
        diagnostics = {
            "energy_drift_percent": 1.0,  # Too high
            "spin_norm_drift_percent": 0.0001,
            "mu_max_rel_dev": 0.001,
        }

        gates_passed, reason, score = QualityGates.validate(diagnostics)
        assert not gates_passed
        assert score == 0.0
        assert "Energy drift" in reason

    def test_gates_fail_spin_norm(self):
        """Gates fail on high spin norm drift."""
        diagnostics = {
            "energy_drift_percent": 0.001,
            "spin_norm_drift_percent": 0.01,  # Too high
            "mu_max_rel_dev": 0.001,
        }

        gates_passed, reason, score = QualityGates.validate(diagnostics)
        assert not gates_passed
        assert score == 0.0
        assert "Spin norm drift" in reason

    def test_gates_fail_mu_deviation(self):
        """Gates fail on high magnetic moment deviation."""
        diagnostics = {
            "energy_drift_percent": 0.001,
            "spin_norm_drift_percent": 0.0001,
            "mu_max_rel_dev": 0.05,  # Too high
        }

        gates_passed, reason, score = QualityGates.validate(diagnostics)
        assert not gates_passed
        assert score == 0.0
        assert "Magnetic moment" in reason


class TestHardwareDispatch:
    """Test hardware detection and dispatch."""

    def test_get_hardware_info(self):
        """Should return hardware configuration."""
        hw = get_hardware_info()

        assert isinstance(hw, dict)
        assert "cuda_available" in hw
        assert "device_count" in hw
        assert "rocm_available" in hw

        if hw["cuda_available"]:
            assert "device_name" in hw
            assert "device_capability" in hw

    def test_hardware_info_types(self):
        """Check hardware info types."""
        hw = get_hardware_info()

        assert isinstance(hw["cuda_available"], bool)
        assert isinstance(hw["device_count"], int)
        assert isinstance(hw["rocm_available"], bool)


class TestSimulationStability:
    """Test physics simulation stability."""

    def test_spin_norm_conservation(self):
        """Spin norm should be approximately conserved."""
        result = batch_simulate(
            batch_size=1,
            num_steps=200,
        )

        spin_drift = result["diagnostics"]["spin_norm_drift_percent"]
        # Should be small for good integration
        assert spin_drift < 1.0  # < 1% drift

    def test_trajectory_continuity(self):
        """Trajectories should be continuous (no NaN/Inf)."""
        result = batch_simulate(
            batch_size=2,
            num_steps=100,
        )

        traj = result["trajectories"]
        assert not torch.isnan(traj).any()
        assert not torch.isinf(traj).any()

    def test_sensitivity_tensor_validity(self):
        """Sensitivity tensors should not have NaN/Inf."""
        result = batch_simulate(
            batch_size=1,
            num_steps=50,
        )

        W = result["sensitivity_W"]
        assert not torch.isnan(W).any()
        assert not torch.isinf(W).any()


class TestMLFlowLogging:
    """Test MLFlow integration (mock)."""

    def test_track_to_mlflow_no_error(self):
        """Should not error when MLFlow unavailable."""
        result = {
            "batch_size": 10,
            "num_steps": 100,
            "quality_score": 0.95,
            "diagnostics": {
                "energy_drift_percent": 0.001,
                "spin_norm_drift_percent": 0.0001,
                "mu_max_rel_dev": 0.001,
            },
        }

        hw = get_hardware_info()

        # Should not raise, even if MLFlow not available
        try:
            track_to_mlflow("test-run-1", result, hw, 5.0)
        except ImportError:
            # Expected if MLFlow not installed
            pass


class TestEdgeCases:
    """Test edge cases."""

    def test_zero_edm_eta(self):
        """Should work with zero EDM coupling."""
        result = batch_simulate(
            batch_size=1,
            num_steps=50,
            params={
                "q": 1.0, "m": 0.938, "c": 299.792458, "anomaly": 1.793,
                "edm_eta": 0.0, "L": 0.5, "B0_initial": 1.0, "dtau": 1e-3,
                "sensitivity_eps": 1e-6, "shape_sensitivity_eps": 1e-6,
            },
        )

        assert result["quality_score"] >= 0.0

    def test_large_batch_size(self):
        """Should handle larger batch sizes."""
        result = batch_simulate(
            batch_size=50,
            num_steps=20,
        )

        assert result["trajectories"].shape == (50, 20, 7)

    def test_many_steps(self):
        """Should handle more steps."""
        result = batch_simulate(
            batch_size=2,
            num_steps=500,
        )

        assert result["trajectories"].shape == (2, 500, 7)
