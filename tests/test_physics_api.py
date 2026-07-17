"""Physics simulation API endpoint tests."""

import pytest
from api.main import PhysicsSimulationRequest, PhysicsSimulationResponse


class TestPhysicsAPI:
    """Test /api/simulate endpoint."""

    def test_physics_simulation_request_validation(self):
        """PhysicsSimulationRequest should validate batch_size and num_steps."""
        # Valid request
        req = PhysicsSimulationRequest(batch_size=10, num_steps=1000)
        assert req.batch_size == 10
        assert req.num_steps == 1000

    def test_physics_simulation_request_defaults(self):
        """PhysicsSimulationRequest should have sensible defaults."""
        req = PhysicsSimulationRequest()
        assert req.batch_size == 10
        assert req.num_steps == 10000
        assert req.particle_mass == 0.938
        assert req.edm_eta == 1e-3

    def test_physics_simulation_request_bounds(self):
        """Should enforce min/max bounds on batch_size and num_steps."""
        # batch_size must be >= 1
        with pytest.raises(ValueError):
            PhysicsSimulationRequest(batch_size=0)

        # batch_size must be <= 1000
        with pytest.raises(ValueError):
            PhysicsSimulationRequest(batch_size=2000)

        # num_steps must be >= 100
        with pytest.raises(ValueError):
            PhysicsSimulationRequest(num_steps=50)

        # num_steps must be <= 100000
        with pytest.raises(ValueError):
            PhysicsSimulationRequest(num_steps=200000)

    def test_physics_simulation_response_fields(self):
        """PhysicsSimulationResponse should have all required fields."""
        resp = PhysicsSimulationResponse(
            run_id="test-run-1",
            batch_size=10,
            num_steps=1000,
            quality_score=0.95,
            diagnostics={"gates_passed": True},
            hardware_used="cuda",
            wall_clock_time_sec=5.2,
            estimated_cost_usd=0.003,
            gates_passed=True,
            status="ok",
        )

        assert resp.run_id == "test-run-1"
        assert resp.quality_score == 0.95
        assert resp.status == "ok"
        assert resp.hardware_used == "cuda"
