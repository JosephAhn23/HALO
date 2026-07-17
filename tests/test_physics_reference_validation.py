"""Validate the batched physics integrator against the authoritative single-particle
reference implementation in dynamo-4fd5425-scientific-computing-and-domain-science.

That reference (task/solution/simulate.py) is the ground truth this project's
batched engine (simulation/batched_integrator.py) is derived from: the 44-state
RK4 system (orbit + spin + first/second-order variational sensitivities) must
match it element-for-element, just executed as tensor ops across a particle
batch instead of scalar/numpy ops for one particle at a time.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from src.simulation import batch_simulate

DYNAMO_DIR = Path(__file__).parent.parent / "dynamo-4fd5425-scientific-computing-and-domain-science"
REFERENCE_SCRIPT = DYNAMO_DIR / "task/solution/simulate.py"
DATA_DIR = DYNAMO_DIR / "task/environment/data"
PARAMS_PATH = DATA_DIR / "params.json"
B0_SCHEDULE_PATH = DATA_DIR / "b0_schedule.json"
ALPHA_SCHEDULE_PATH = DATA_DIR / "alpha_schedule.json"

pytestmark = pytest.mark.skipif(
    not REFERENCE_SCRIPT.exists(), reason="dynamo-4fd5425 reference folder not present"
)

PHASE_VECTOR_KEYS = (
    "spin_final",
    "spin_eta_sensitivity_final",
    "spin_eta_curvature_final",
    "state_logB0_sensitivity_final",
    "state_logB0_curvature_final",
    "state_logL_sensitivity_final",
    "state_logB0_logL_mixed_final",
)

# Reduced from the reference task's 30000 to keep this test fast. Any implementation
# mismatch (wrong sign, missing term, wrong RK4 indexing) shows up well within a few
# hundred steps since both sides run the identical truncated system.
REDUCED_STEPS = 300


def _load_reference_module():
    spec = importlib.util.spec_from_file_location("dynamo_reference_simulate", REFERENCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _close(a, b, rtol=5e-3, atol=5e-3):
    return abs(a - b) <= atol + rtol * abs(b)


@pytest.fixture(scope="module")
def ref_module():
    return _load_reference_module()


@pytest.fixture(scope="module")
def sample_data():
    with open(PARAMS_PATH) as f:
        params = json.load(f)
    with open(B0_SCHEDULE_PATH) as f:
        b0_schedule = json.load(f)
    with open(ALPHA_SCHEDULE_PATH) as f:
        alpha_schedule = json.load(f)
    return params, b0_schedule, alpha_schedule


@pytest.fixture(scope="module")
def reference_and_batched(ref_module, sample_data):
    """Run both implementations on the sample params with a reduced step count."""
    params, b0_schedule, alpha_schedule = sample_data

    ref_params = dict(params)
    ref_params["n_steps"] = REDUCED_STEPS
    ref_params["b0_schedule"] = b0_schedule
    ref_params["alpha_schedule"] = alpha_schedule
    ref = ref_module.integrate(ref_params, str(PARAMS_PATH))

    initial_conditions = torch.tensor(
        [params["x0"] + params["v0"] + params["spin0"]], dtype=torch.float64
    )

    ours = batch_simulate(
        batch_size=1,
        B_field_schedule=b0_schedule,
        alpha_schedule=alpha_schedule,
        initial_conditions=initial_conditions,
        params=params,
        num_steps=REDUCED_STEPS,
        device="cpu",
    )

    return ref, ours


class TestBatchedSchedulesMatchReference:
    """Unit-level check: our vectorized schedule evaluators vs. the reference's scalar ones."""

    def test_b0_schedule_matches(self, ref_module, sample_data):
        from src.simulation.batched_integrator import BatchedB0Schedule

        params, b0_schedule, _ = sample_data
        t_nodes = b0_schedule["t"]
        rates = b0_schedule["dlnB0_dt"]
        B0_initial = params["B0_initial"]

        B0_of_t, dB0dt = ref_module.make_rate_b0_funs(t_nodes, rates, B0_initial)
        batched = BatchedB0Schedule(t_nodes, rates, B0_initial, device="cpu", dtype=torch.float64)

        query_points = [-100.0, 0.0, 400.0, 800.0, 2000.0, 4000.0, 5000.0]
        t_tensor = torch.tensor(query_points, dtype=torch.float64)
        B0_batch, dB0dt_batch = batched(t_tensor)

        for i, t in enumerate(query_points):
            assert _close(B0_batch[i].item(), B0_of_t(t), rtol=1e-9, atol=1e-9)
            assert _close(dB0dt_batch[i].item(), dB0dt(t), rtol=1e-9, atol=1e-9)

    def test_alpha_schedule_matches(self, ref_module, sample_data):
        from src.simulation.batched_integrator import BatchedAlphaSchedule

        _, _, alpha_schedule = sample_data
        t_nodes = alpha_schedule["t"]
        alpha_vals = alpha_schedule["alpha"]

        alpha_of_t, dalphadt = ref_module.make_lin_funs(t_nodes, alpha_vals)
        batched = BatchedAlphaSchedule(t_nodes, alpha_vals, device="cpu", dtype=torch.float64)

        query_points = [-200.0, 0.0, 250.0, 500.0, 3000.0, 4000.0, 4500.0]
        t_tensor = torch.tensor(query_points, dtype=torch.float64)
        alpha_batch, dalpha_batch = batched(t_tensor)

        for i, t in enumerate(query_points):
            assert _close(alpha_batch[i].item(), alpha_of_t(t), rtol=1e-9, atol=1e-9)
            assert _close(dalpha_batch[i].item(), dalphadt(t), rtol=1e-9, atol=1e-9)


class TestBatchedIntegratorMatchesReference:
    """Full 44-state integration: batched engine vs. reference, same truncated run."""

    def test_final_phase_vectors_match_reference(self, reference_and_batched):
        ref, ours = reference_and_batched
        final_state = ours["sensitivity_W"][0, -1, :]

        got = {
            "spin_final": final_state[7:10].tolist(),
            "spin_eta_sensitivity_final": final_state[10:13].tolist(),
            "spin_eta_curvature_final": final_state[13:16].tolist(),
            "state_logB0_sensitivity_final": final_state[16:23].tolist(),
            "state_logB0_curvature_final": final_state[23:30].tolist(),
            "state_logL_sensitivity_final": final_state[30:37].tolist(),
            "state_logB0_logL_mixed_final": final_state[37:44].tolist(),
        }

        for key in PHASE_VECTOR_KEYS:
            for got_val, expected_val in zip(got[key], ref[key]):
                assert _close(got_val, expected_val), f"{key} mismatch: {got_val} vs {expected_val}"

    def test_orbit_matches_reference(self, reference_and_batched, sample_data):
        params, _, _ = sample_data
        ref, ours = reference_and_batched
        final_state = ours["sensitivity_W"][0, -1, :]
        c = float(params["c"])

        position = final_state[1:4].tolist()
        u = final_state[4:7]
        g = torch.sqrt(1.0 + (u * u).sum() / (c * c))
        velocity = (u / g).tolist()

        for got_val, expected_val in zip(position, ref["position"]):
            assert _close(got_val, expected_val)
        for got_val, expected_val in zip(velocity, ref["velocity"]):
            assert _close(got_val, expected_val)

    def test_quality_gates_pass_on_reference_sample(self, reference_and_batched):
        """The batched engine's own quality gates should pass on this well-behaved sample."""
        _, ours = reference_and_batched
        assert ours["diagnostics"]["mu_max_rel_dev"] < 5.0e-3
        assert ours["quality_score"] == 1.0
