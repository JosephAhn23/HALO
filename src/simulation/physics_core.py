"""Physics simulator: RK4 integrator + Thomas-BMT + EDM + variational sensitivities.

Batching runs the entire particle batch through each RK4 stage as a single set
of torch tensor ops (see batched_integrator.py) instead of looping over
particles in Python -- this is what lets cuda-morph route the simulation to a
single GPU dispatch for N particles at once.

Adapted from: https://github.com/handshake-project-dynamo/dynamo-4fd5425-scientific-computing-and-domain-science/pull/1
Original author: Joseph Ahn
Correctness is validated against that reference implementation in
tests/test_physics_reference_validation.py.
"""

import json
import logging

import numpy as np
import torch

from .batched_integrator import integrate_batch

logger = logging.getLogger(__name__)


class FieldSchedules:
    """Load B0 and alpha field schedules from params (shared helper for both the
    scalar reference math and the batched tensor integrator)."""

    @staticmethod
    def load_b0_rate_schedule(params, params_path=""):
        """Load B0 schedule: d(ln B0)/dt rates."""
        p = params
        if "b0_schedule" in p:
            sch = p["b0_schedule"]
        elif "b0_schedule_file" in p:
            with open(p["b0_schedule_file"]) as f:
                sch = json.load(f)
        else:
            # Default constant schedule
            sch = {"t": [0.0], "dlnB0_dt": [0.0]}

        t = np.asarray(sch["t"], dtype=float)
        rates = np.asarray(sch["dlnB0_dt"], dtype=float)
        order = np.argsort(t)
        return t[order], rates[order], float(p.get("B0_initial", 1.0))

    @staticmethod
    def load_alpha_schedule(params, params_path=""):
        """Load alpha schedule: mirror field quadrupole correction."""
        p = params
        if "alpha_schedule" in p:
            sch = p["alpha_schedule"]
        elif "alpha_schedule_file" in p:
            with open(p["alpha_schedule_file"]) as f:
                sch = json.load(f)
        else:
            sch = {"t": [0.0], "alpha": [0.0]}

        t = np.asarray(sch["t"], dtype=float)
        alpha = np.asarray(sch["alpha"], dtype=float)
        order = np.argsort(t)
        return t[order], alpha[order]


def _default_initial_conditions(batch_size: int, dtype=torch.float64):
    """Particle 0 is the canonical default; the rest get small seeded perturbations.

    Generated as batched tensor ops (one randn call for the whole batch) rather
    than a per-particle Python loop, consistent with the tensor-batched design.
    """
    x0 = torch.zeros(batch_size, 3, dtype=dtype)
    v0 = torch.zeros(batch_size, 3, dtype=dtype)
    v0[:, 2] = 0.1
    spin0 = torch.zeros(batch_size, 3, dtype=dtype)
    spin0[:, 2] = 1.0

    if batch_size > 1:
        gen = torch.Generator().manual_seed(0)
        x0[1:] = torch.randn(batch_size - 1, 3, generator=gen, dtype=dtype) * 0.01
        v0[1:] += torch.randn(batch_size - 1, 3, generator=gen, dtype=dtype) * 0.005
        spin0[1:] += torch.randn(batch_size - 1, 3, generator=gen, dtype=dtype) * 0.001

    spin0 = spin0 / spin0.norm(dim=-1, keepdim=True)
    return x0, v0, spin0


def _initial_conditions_from_tensor(initial_conditions: torch.Tensor, dtype=torch.float64):
    """Unpack a (batch, 7) tensor: [x,y,z,vx,vy,vz,spin_z].

    Only a single spin scalar is carried per particle (spin assumed to seed
    along the z-axis with that polarization) -- callers that need a fully
    general initial spin direction should pass the (batch, 9) form
    [x,y,z,vx,vy,vz,sx,sy,sz] instead.
    """
    ic = initial_conditions.to(dtype=dtype)
    x0 = ic[:, 0:3]
    v0 = ic[:, 3:6]

    if ic.shape[1] >= 9:
        spin0 = ic[:, 6:9]
    elif ic.shape[1] == 7:
        spin0 = torch.zeros_like(x0)
        spin0[:, 2] = ic[:, 6]
    else:
        spin0 = torch.zeros_like(x0)
        spin0[:, 2] = 1.0

    spin0 = spin0 / spin0.norm(dim=-1, keepdim=True)
    return x0, v0, spin0


def batch_simulate(
    batch_size: int,
    B_field_schedule: dict | None = None,
    alpha_schedule: dict | None = None,
    initial_conditions: torch.Tensor | None = None,
    params: dict | None = None,
    num_steps: int = 10000,
    device: str | None = None,
) -> dict:
    """
    Batched relativistic spin transport simulation.

    Every RK4 stage runs the whole batch as one tensor operation (see
    batched_integrator.integrate_batch) -- N particles are not simulated as N
    sequential loops.

    Args:
        batch_size: Number of particles
        B_field_schedule: B0 rate schedule (embedded in params or dict)
        alpha_schedule: Mirror field schedule
        initial_conditions: (batch, 7) initial state [x, y, z, vx, vy, vz, spin_z]
        params: Physics parameters dict
        num_steps: Integration steps
        device: 'cuda' or 'cpu'

    Returns:
        {
            "trajectories": (batch, num_steps, 7),
            "sensitivity_W": (batch, num_steps, 44),
            "diagnostics": {
                "mu_max_rel_dev": float,
                "spin_norm_drift_percent": float,
                "gates_passed": bool
            },
            "quality_score": float (0-1),
            "success": bool
        }
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if params is None:
        # Default parameters for proton in EDM storage ring
        params = {
            "q": 1.0,
            "m": 0.938,  # proton mass GeV/c^2
            "c": 299.792458,  # speed of light units
            "anomaly": 1.793,  # proton g-2
            "edm_eta": 1e-3,
            "L": 0.5,  # mirror field length
            "B0_initial": 1.0,  # Tesla
            "dtau": 1e-3,  # proper time step
            "n_steps": num_steps,
            "sensitivity_eps": 1e-6,
            "shape_sensitivity_eps": 1e-6,
            "diagnostic_time": 0.0,
        }
    params = dict(params)
    params["n_steps"] = num_steps

    if B_field_schedule:
        params["b0_schedule"] = B_field_schedule
    if alpha_schedule:
        params["alpha_schedule"] = alpha_schedule

    logger.info(f"Starting batch simulation: batch_size={batch_size}, device={device}")

    if initial_conditions is not None:
        x0, v0, spin0 = _initial_conditions_from_tensor(initial_conditions)
    else:
        x0, v0, spin0 = _default_initial_conditions(batch_size)

    result = integrate_batch(x0, v0, spin0, params, device=device)

    mu_max_rel_dev = result["mu_max_rel_dev"]
    spin_norm_drift = result["spin_norm_drift"]

    mu_gate = mu_max_rel_dev < 0.01  # < 1% deviation, per particle
    spin_gate = spin_norm_drift < 0.001  # < 0.1% drift, per particle
    gates_passed_per_particle = mu_gate & spin_gate

    gates_passed = bool(gates_passed_per_particle.all().item())
    quality_score = float(gates_passed_per_particle.float().mean().item())

    out = {
        "trajectories": result["trajectories"].float(),
        "sensitivity_W": result["sensitivity_W"].float(),
        "diagnostics": {
            # Magnetic-moment adiabatic invariance is this system's analogue of an
            # energy conservation check; expressed in percent to match QualityGates'
            # and spin_norm_drift_percent's units.
            "energy_drift_percent": float(100.0 * mu_max_rel_dev.mean().item()),
            "mu_max_rel_dev": float(mu_max_rel_dev.mean().item()),
            "spin_norm_drift_percent": float(100.0 * spin_norm_drift.mean().item()),
            "gates_passed": gates_passed,
        },
        "quality_score": quality_score,
        "success": gates_passed,
    }

    logger.info(
        f"Simulation complete: quality_score={quality_score:.4f}, gates_passed={gates_passed}"
    )

    return out
