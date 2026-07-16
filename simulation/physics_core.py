"""Physics simulator: RK4 integrator + Thomas-BMT + EDM + variational sensitivities.

Adapted from: https://github.com/handshake-project-dynamo/dynamo-4fd5425-scientific-computing-and-domain-science/pull/1
Original author: Joseph Ahn
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FieldSchedules:
    """Load and interpolate B0 and alpha field schedules."""

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

    @staticmethod
    def make_lin_funs(t_nodes, y_nodes):
        """Linear interpolation for schedules."""
        t_nodes = np.asarray(t_nodes, dtype=float)
        y_nodes = np.asarray(y_nodes, dtype=float)
        slopes = np.diff(y_nodes) / np.diff(t_nodes)

        def y_of_t(t):
            t = float(t)
            if t <= t_nodes[0]:
                return float(y_nodes[0] + slopes[0] * (t - t_nodes[0]))
            if t >= t_nodes[-1]:
                return float(y_nodes[-1] + slopes[-1] * (t - t_nodes[-1]))
            i = int(np.searchsorted(t_nodes, t, side="right") - 1)
            i = max(0, min(i, len(slopes) - 1))
            return float(y_nodes[i] + slopes[i] * (t - t_nodes[i]))

        def dydt(t):
            t = float(t)
            if t <= t_nodes[0]:
                return float(slopes[0])
            if t >= t_nodes[-1]:
                return float(slopes[-1])
            i = int(np.searchsorted(t_nodes, t, side="right") - 1)
            i = max(0, min(i, len(slopes) - 1))
            return float(slopes[i])

        return y_of_t, dydt

    @staticmethod
    def make_rate_b0_funs(t_nodes, rates, B0_initial):
        """Create B0(t) and dB0/dt functions from rate schedule."""
        ln0 = float(np.log(B0_initial))

        def dln_dt(t):
            t = float(t)
            if t <= t_nodes[0]:
                return float(rates[0])
            if t >= t_nodes[-1]:
                return float(rates[-1])
            i = int(np.searchsorted(t_nodes, t, side="right") - 1)
            i = max(0, min(i, len(rates) - 1))
            return float(rates[i])

        def lnB_of_t(t):
            t = float(t)
            if t <= t_nodes[0]:
                return ln0 + float(rates[0]) * (t - t_nodes[0])
            acc = ln0
            for i in range(len(t_nodes) - 1):
                t0, t1 = t_nodes[i], t_nodes[i + 1]
                if t <= t1:
                    return acc + float(rates[i]) * (t - t0)
                acc += float(rates[i]) * (t1 - t0)
            return acc + float(rates[-1]) * (t - t_nodes[-1])

        def B0_of_t(t):
            return float(np.exp(lnB_of_t(t)))

        def dB0dt(t):
            return float(B0_of_t(t) * dln_dt(t))

        return B0_of_t, dB0dt


class ThomasBMTSimulator:
    """Thomas-BMT equation + EDM coupling + variational sensitivities."""

    @staticmethod
    def fields(t, x, p, B0_of_t, dB0dt, alpha_of_t, dalphadt):
        """Compute E and B fields at position x and time t."""
        L = float(p["L"])
        b0 = B0_of_t(t)
        alpha = alpha_of_t(t)

        z = x[2]
        u2 = (z / L) ** 2
        ex = np.exp(-u2)
        f = 1.0 + alpha * (1.0 - ex)
        dfdz = alpha * (2.0 * z / (L * L)) * ex

        B = np.array(
            [-0.5 * b0 * x[0] * dfdz, -0.5 * b0 * x[1] * dfdz, b0 * f], dtype=float
        )
        dBzdt = dB0dt(t) * f + b0 * (1.0 - ex) * dalphadt(t)
        E = 0.5 * dBzdt * np.array([x[1], -x[0], 0.0], dtype=float)

        return E, B

    @staticmethod
    def gamma(v, c):
        """Lorentz gamma factor."""
        return 1.0 / np.sqrt(max(1e-14, 1.0 - float(np.dot(v, v)) / (c * c)))

    @staticmethod
    def spin_omega(t, x, v, p, B0_of_t, dB0dt, alpha_of_t, dalphadt):
        """Thomas-BMT spin precession frequency."""
        q = float(p["q"])
        m = float(p["m"])
        c = float(p["c"])
        a = float(p["anomaly"])
        eta = float(p.get("edm_eta", 0.0))

        g = ThomasBMTSimulator.gamma(v, c)
        E, B = ThomasBMTSimulator.fields(t, x, p, B0_of_t, dB0dt, alpha_of_t, dalphadt)

        return -(q / m) * (
            (a + 1.0 / g) * B
            - a * g / (g + 1.0) * float(np.dot(v, B)) * v / (c * c)
            - (a + 1.0 / (g + 1.0)) * np.cross(v, E) / (c * c)
            + 0.5 * eta * (E / c + np.cross(v, B) / c - g / (g + 1.0) * float(np.dot(v, E)) * v / (c * c * c))
        )

    @staticmethod
    def spin_omega_eta(t, x, v, p, B0_of_t, dB0dt, alpha_of_t, dalphadt):
        """EDM-specific spin sensitivity."""
        q = float(p["q"])
        m = float(p["m"])
        c = float(p["c"])

        g = ThomasBMTSimulator.gamma(v, c)
        E, B = ThomasBMTSimulator.fields(t, x, p, B0_of_t, dB0dt, alpha_of_t, dalphadt)

        return -(q / (2.0 * m)) * (
            E / c + np.cross(v, B) / c - g / (g + 1.0) * float(np.dot(v, E)) * v / (c * c * c)
        )


class DiagnosticGates:
    """Quality gates: energy + spin-norm conservation checks."""

    @staticmethod
    def check_magnetic_moment(mus, mus0):
        """Check magnetic moment adiabatic invariance."""
        mu_mean = float(np.mean(mus))
        mu_rms_rel = float(np.sqrt(np.mean((mus - mu_mean) ** 2)) / abs(mus0))
        mu_max_rel_dev = float(np.max(np.abs(mus - mus0)) / abs(mus0))

        gates_passed = mu_max_rel_dev < 0.01  # < 1% deviation
        return mu_rms_rel, mu_max_rel_dev, gates_passed

    @staticmethod
    def check_spin_norm(spin_final, spin_norm0):
        """Check spin normalization conservation."""
        spin_norm_final = float(np.linalg.norm(spin_final))
        spin_norm_drift = float(abs(spin_norm_final - spin_norm0) / spin_norm0)
        gates_passed = spin_norm_drift < 0.001  # < 0.1% drift
        return spin_norm_drift, gates_passed


def batch_simulate(
    batch_size: int,
    B_field_schedule: Optional[Dict] = None,
    alpha_schedule: Optional[Dict] = None,
    initial_conditions: Optional[torch.Tensor] = None,
    params: Optional[Dict] = None,
    num_steps: int = 10000,
    device: Optional[str] = None,
) -> Dict:
    """
    Batched relativistic spin transport simulation.

    Args:
        batch_size: Number of particles
        B_field_schedule: B0 rate schedule (embedded in params or dict)
        alpha_schedule: Mirror field schedule
        initial_conditions: (batch, 7) initial state [x, y, z, vx, vy, vz, spin_state]
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
            "b0_schedule": {"t": [0.0], "dlnB0_dt": [0.0]},
            "alpha_schedule": {"t": [0.0], "alpha": [0.0]},
        }

    if B_field_schedule:
        params["b0_schedule"] = B_field_schedule
    if alpha_schedule:
        params["alpha_schedule"] = alpha_schedule

    logger.info(f"Starting batch simulation: batch_size={batch_size}, device={device}")

    # Load schedules
    sched = FieldSchedules()
    t_nodes_b0, rates_b0, B0_init = sched.load_b0_rate_schedule(params)
    B0_of_t, dB0dt = sched.make_rate_b0_funs(t_nodes_b0, rates_b0, B0_init)
    t_nodes_alpha, alpha_vals = sched.load_alpha_schedule(params)
    alpha_of_t, dalphadt = sched.make_lin_funs(t_nodes_alpha, alpha_vals)

    # Initialize trajectories (batch, num_steps, 7)
    trajectories = []
    sensitivities = []
    qualities = []

    # For simplicity, run sequential (can be parallelized with vmap)
    for batch_idx in range(batch_size):
        if batch_idx == 0:
            # Use provided initial condition or defaults
            if initial_conditions is not None:
                ic = initial_conditions[batch_idx].cpu().numpy()
                x0 = ic[0:3]
                v0 = ic[3:6]
                spin0 = ic[6:9] if ic.shape[0] > 6 else np.array([0, 0, 1.0])
            else:
                x0 = np.array([0.0, 0.0, 0.0])
                v0 = np.array([0.0, 0.0, 0.1])
                spin0 = np.array([0.0, 0.0, 1.0])
        else:
            # Vary initial conditions slightly
            if initial_conditions is not None:
                ic = initial_conditions[batch_idx].cpu().numpy()
                x0 = ic[0:3]
                v0 = ic[3:6]
                spin0 = ic[6:9] if ic.shape[0] > 6 else np.array([0, 0, 1.0])
            else:
                rng = np.random.RandomState(seed=batch_idx)
                x0 = rng.randn(3) * 0.01
                v0 = np.array([0.0, 0.0, 0.1]) + rng.randn(3) * 0.005
                spin0 = np.array([0.0, 0.0, 1.0]) + rng.randn(3) * 0.001
                spin0 /= np.linalg.norm(spin0)

        # Run single trajectory
        traj, W, quality = _integrate_single(
            x0, v0, spin0, params, B0_of_t, dB0dt, alpha_of_t, dalphadt
        )

        trajectories.append(traj)
        sensitivities.append(W)
        qualities.append(quality)

        if (batch_idx + 1) % max(1, batch_size // 10) == 0:
            logger.info(f"  Completed {batch_idx + 1}/{batch_size}")

    # Stack results
    trajectories_tensor = torch.from_numpy(np.stack(trajectories)).float().to(device)
    W_tensor = torch.from_numpy(np.stack(sensitivities)).float().to(device)

    # Average quality metrics
    avg_quality = np.mean([q["quality_score"] for q in qualities])
    gates_passed = all(q["gates_passed"] for q in qualities)

    result = {
        "trajectories": trajectories_tensor,
        "sensitivity_W": W_tensor,
        "diagnostics": {
            "mu_max_rel_dev": np.mean([q["mu_max_rel_dev"] for q in qualities]),
            "spin_norm_drift_percent": 100.0 * np.mean([q["spin_norm_drift"] for q in qualities]),
            "gates_passed": gates_passed,
        },
        "quality_score": avg_quality,
        "success": gates_passed,
    }

    logger.info(
        f"Simulation complete: quality_score={avg_quality:.4f}, "
        f"gates_passed={gates_passed}"
    )

    return result


def _integrate_single(x0, v0, spin0, params, B0_of_t, dB0dt, alpha_of_t, dalphadt):
    """Integrate single trajectory using Joseph Ahn's RK4 method."""
    q = float(params["q"])
    m = float(params["m"])
    c = float(params["c"])
    dtau = float(params.get("dtau", 1e-3))
    n_steps = int(params.get("n_steps", 10000))
    sensitivity_eps = float(params.get("sensitivity_eps", 1e-6))
    shape_sensitivity_eps = float(params.get("shape_sensitivity_eps", 1e-6))

    sim = ThomasBMTSimulator()

    # Initialize state
    t = 0.0
    x = np.array(x0, dtype=float)
    v = np.array(v0, dtype=float)
    spin = np.array(spin0, dtype=float)
    spin /= np.linalg.norm(spin)  # Normalize

    # Proper velocity
    g0 = sim.gamma(v, c)
    u = g0 * v

    # Initial diagnostics
    mu0 = _magnetic_moment(m, x, v, t, params, B0_of_t, dB0dt, alpha_of_t, dalphadt)
    spin_norm0 = float(np.linalg.norm(spin))
    mus = np.empty(n_steps + 1, dtype=float)
    mus[0] = mu0

    # Trajectory storage
    traj = np.zeros((n_steps, 7), dtype=float)
    W = np.zeros((n_steps, 44), dtype=float)  # Full state (44 dimensions in Joseph's code)

    def translation_rhs(state, field_scale=1.0, length_scale=1.0):
        tlab = state[0]
        pos = state[1:4]
        proper_v = state[4:7]
        g = float(np.sqrt(1.0 + float(np.dot(proper_v, proper_v)) / (c * c)))
        E, B = sim.fields(tlab, pos, params, B0_of_t, dB0dt, alpha_of_t, dalphadt)
        return np.concatenate(([g], proper_v, (q / m) * (g * E + np.cross(proper_v, B))))

    def rhs(y):
        """Full RHS with sensitivities (44-dim state)."""
        tlab = y[0]
        pos = y[1:4]
        proper_v = y[4:7]
        spin_vec = y[7:10]
        spin_eta = y[10:13]
        spin_eta2 = y[13:16]
        orbit_b0 = y[16:23]
        orbit_b02 = y[23:30]
        orbit_L = y[30:37]
        orbit_b0L = y[37:44]

        g = float(np.sqrt(1.0 + float(np.dot(proper_v, proper_v)) / (c * c)))
        vel = proper_v / g

        E, B = sim.fields(tlab, pos, params, B0_of_t, dB0dt, alpha_of_t, dalphadt)
        omega = sim.spin_omega(tlab, pos, vel, params, B0_of_t, dB0dt, alpha_of_t, dalphadt)
        omega_eta = sim.spin_omega_eta(tlab, pos, vel, params, B0_of_t, dB0dt, alpha_of_t, dalphadt)

        orbit = y[:7]
        h = sensitivity_eps
        plus = orbit + h * orbit_b0 + 0.5 * h * h * orbit_b02
        minus = orbit - h * orbit_b0 + 0.5 * h * h * orbit_b02

        f0 = translation_rhs(orbit)
        fp = translation_rhs(plus, float(np.exp(h)))
        fm = translation_rhs(minus, float(np.exp(-h)))

        k = shape_sensitivity_eps
        frp = translation_rhs(orbit + k * orbit_L, 1.0, float(np.exp(k)))
        frm = translation_rhs(orbit - k * orbit_L, 1.0, float(np.exp(-k)))

        corners = {}
        for sb in (-1.0, 1.0):
            for sl in (-1.0, 1.0):
                corner = orbit + sb * h * orbit_b0 + sl * k * orbit_L + sb * sl * h * k * orbit_b0L + 0.5 * h * h * orbit_b02
                corners[sb, sl] = translation_rhs(corner, float(np.exp(sb * h)), float(np.exp(sl * k)))

        return np.concatenate((
            f0,
            g * np.cross(omega, spin_vec),
            g * (np.cross(omega, spin_eta) + np.cross(omega_eta, spin_vec)),
            g * (np.cross(omega, spin_eta2) + 2.0 * np.cross(omega_eta, spin_eta)),
            (fp - fm) / (2.0 * h),
            (fp - 2.0 * f0 + fm) / (h * h),
            (frp - frm) / (2.0 * k),
            (corners[1.0, 1.0] - corners[1.0, -1.0] - corners[-1.0, 1.0] + corners[-1.0, -1.0]) / (4.0 * h * k),
        ))

    # RK4 integration
    y = np.concatenate(([t], x, u, spin, np.zeros(34, dtype=float)))

    for i in range(n_steps):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * dtau * k1)
        k3 = rhs(y + 0.5 * dtau * k2)
        k4 = rhs(y + dtau * k3)

        y = y + (dtau / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        t = float(y[0])
        x = y[1:4]
        u = y[4:7]
        g = float(np.sqrt(1.0 + float(np.dot(u, u)) / (c * c)))
        v = u / g
        spin = y[7:10]

        mus[i + 1] = _magnetic_moment(m, x, v, t, params, B0_of_t, dB0dt, alpha_of_t, dalphadt)
        traj[i] = np.concatenate([x, v, spin])
        W[i] = y[7:]  # Store sensitivity state

    # Quality checks
    mu_rms_rel, mu_max_rel_dev, mu_gate = DiagnosticGates.check_magnetic_moment(mus[len(mus) // 2:], mus[0])
    spin_norm_drift, spin_gate = DiagnosticGates.check_spin_norm(spin, spin_norm0)

    gates_passed = mu_gate and spin_gate
    quality_score = 1.0 if gates_passed else 0.0

    return (
        traj,
        W,
        {
            "mu_max_rel_dev": mu_max_rel_dev,
            "spin_norm_drift": spin_norm_drift,
            "gates_passed": gates_passed,
            "quality_score": quality_score,
        },
    )


def _magnetic_moment(m, x, v, t, p, B0_of_t, dB0dt, alpha_of_t, dalphadt):
    """Compute magnetic moment (adiabatic invariant)."""
    c = float(p["c"])
    sim = ThomasBMTSimulator()

    E, B = sim.fields(t, x, p, B0_of_t, dB0dt, alpha_of_t, dalphadt)
    Bmag = float(np.linalg.norm(B))
    if Bmag < 1e-10:
        return 0.0

    bhat = B / Bmag
    g = sim.gamma(v, c)
    v_par = float(np.dot(v, bhat))
    v_perp2 = float(np.dot(v, v) - v_par * v_par)

    return 0.5 * m * (g * g) * v_perp2 / Bmag
