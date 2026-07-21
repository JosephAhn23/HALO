"""True tensor-batched RK4 integrator for the 44-state relativistic spin-transport ODE.

Every RK4 stage evaluates the full ODE for the entire particle batch as a single
set of torch tensor ops (leading batch dimension) instead of looping over
particles in Python. Only the time axis is sequential, which is inherent to any
explicit integrator -- the batch axis is what gets parallelized, on CPU or GPU.

The physics (fields, Thomas-BMT precession, variational sensitivities) mirrors
the single-particle reference implementation at
dynamo-4fd5425-scientific-computing-and-domain-science/task/solution/simulate.py
element-for-element. See tests/test_physics_reference_validation.py for a
diagnostic-level comparison against that reference.
"""

import math

import torch


class BatchedB0Schedule:
    """Vectorized B0(t), dB0/dt for a batch of (possibly distinct) query times.

    Mirrors make_rate_b0_funs from the reference implementation, but accepts a
    (batch,) tensor of query times and returns (batch,) tensors so it can be
    called once per RK4 sub-stage for the whole batch at once.
    """

    def __init__(self, t_nodes, rates, B0_initial: float, device, dtype):
        t_nodes = list(t_nodes)
        rates = list(rates)
        if len(t_nodes) == 1:
            # Pad a single-node (constant-rate) schedule so segment math below
            # (which needs at least one interval) still degenerates correctly
            # to a constant rate extrapolated in both directions.
            t_nodes = [t_nodes[0], t_nodes[0] + 1.0]
            rates = [rates[0], rates[0]]

        self.t_nodes = torch.tensor(t_nodes, device=device, dtype=dtype)
        self.rates = torch.tensor(rates, device=device, dtype=dtype)
        self.ln0 = math.log(B0_initial)

        seg_len = self.t_nodes[1:] - self.t_nodes[:-1]
        seg_contrib = self.rates[:-1] * seg_len
        zero = torch.zeros(1, device=device, dtype=dtype)
        self.cum_at_node = torch.cat([zero, torch.cumsum(seg_contrib, dim=0)])
        self.n_rates = self.rates.shape[0]

    def __call__(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.searchsorted(self.t_nodes, t.contiguous(), right=True) - 1
        idx = idx.clamp(0, self.n_rates - 1)
        t0 = self.t_nodes[idx]
        rate = self.rates[idx]
        cum = self.cum_at_node[idx]
        lnB = self.ln0 + cum + rate * (t - t0)
        B0 = torch.exp(lnB)
        dB0dt = B0 * rate
        return B0, dB0dt


class BatchedAlphaSchedule:
    """Vectorized piecewise-linear alpha(t), dalpha/dt for a batch of query times."""

    def __init__(self, t_nodes, alpha_vals, device, dtype):
        t_nodes = list(t_nodes)
        alpha_vals = list(alpha_vals)
        if len(t_nodes) == 1:
            t_nodes = [t_nodes[0], t_nodes[0] + 1.0]
            alpha_vals = [alpha_vals[0], alpha_vals[0]]

        self.t_nodes = torch.tensor(t_nodes, device=device, dtype=dtype)
        self.alpha_vals = torch.tensor(alpha_vals, device=device, dtype=dtype)

        seg_len = self.t_nodes[1:] - self.t_nodes[:-1]
        self.slopes = (self.alpha_vals[1:] - self.alpha_vals[:-1]) / seg_len
        self.n_slopes = self.slopes.shape[0]

    def __call__(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.searchsorted(self.t_nodes, t.contiguous(), right=True) - 1
        idx = idx.clamp(0, self.n_slopes - 1)
        t0 = self.t_nodes[idx]
        y0 = self.alpha_vals[idx]
        slope = self.slopes[idx]
        alpha = y0 + slope * (t - t0)
        return alpha, slope


def gamma_batched(v: torch.Tensor, c: float) -> torch.Tensor:
    """Lorentz gamma factor. v: (..., 3) -> (...,)."""
    beta2 = (v * v).sum(-1) / (c * c)
    return 1.0 / torch.sqrt(torch.clamp(1.0 - beta2, min=1e-14))


def fields_batched(
    t: torch.Tensor,
    x: torch.Tensor,
    L_base: float,
    b0_schedule: BatchedB0Schedule,
    alpha_schedule: BatchedAlphaSchedule,
    field_scale: float = 1.0,
    length_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Axisymmetric divergence-free B field + Faraday-induced E field. x: (..., 3)."""
    b0, dB0dt_val = b0_schedule(t)
    alpha, dalphadt_val = alpha_schedule(t)

    L = L_base * length_scale
    z = x[..., 2]
    u2 = (z / L) ** 2
    ex = torch.exp(-u2)
    f = 1.0 + alpha * (1.0 - ex)
    dfdz = alpha * (2.0 * z / (L * L)) * ex

    Bx = -0.5 * b0 * x[..., 0] * dfdz
    By = -0.5 * b0 * x[..., 1] * dfdz
    Bz = b0 * f
    B = torch.stack([Bx, By, Bz], dim=-1) * field_scale

    dBzdt = dB0dt_val * f + b0 * (1.0 - ex) * dalphadt_val
    Ex = 0.5 * dBzdt * x[..., 1]
    Ey = -0.5 * dBzdt * x[..., 0]
    Ez = torch.zeros_like(Ex)
    E = torch.stack([Ex, Ey, Ez], dim=-1) * field_scale

    return E, B


def spin_omega_batched(t, x, v, q, m, c, a, eta, L_base, b0_schedule, alpha_schedule):
    """Thomas-BMT spin precession frequency, batched. Returns (..., 3)."""
    g = gamma_batched(v, c)
    E, B = fields_batched(t, x, L_base, b0_schedule, alpha_schedule)
    vB = (v * B).sum(-1)
    vE = (v * E).sum(-1)

    term1 = (a + 1.0 / g).unsqueeze(-1) * B
    term2 = (a * g / (g + 1.0) * vB / (c * c)).unsqueeze(-1) * v
    term3 = (a + 1.0 / (g + 1.0)).unsqueeze(-1) * torch.cross(v, E, dim=-1) / (c * c)
    term4 = (
        0.5
        * eta
        * (E / c + torch.cross(v, B, dim=-1) / c - (g / (g + 1.0) * vE / (c**3)).unsqueeze(-1) * v)
    )
    return -(q / m) * (term1 - term2 - term3 + term4)


def spin_omega_eta_batched(t, x, v, q, m, c, L_base, b0_schedule, alpha_schedule):
    """EDM-specific spin sensitivity, batched. Returns (..., 3)."""
    g = gamma_batched(v, c)
    E, B = fields_batched(t, x, L_base, b0_schedule, alpha_schedule)
    vE = (v * E).sum(-1)
    return -(q / (2.0 * m)) * (
        E / c + torch.cross(v, B, dim=-1) / c - (g / (g + 1.0) * vE / (c**3)).unsqueeze(-1) * v
    )


def translation_rhs_batched(
    state, q, m, c, L_base, b0_schedule, alpha_schedule, field_scale=1.0, length_scale=1.0
):
    """RHS of the 7-state orbit (tlab, x, y, z, ux, uy, uz). state: (..., 7)."""
    pos = state[..., 1:4]
    proper_v = state[..., 4:7]
    tlab = state[..., 0]

    g = torch.sqrt(1.0 + (proper_v * proper_v).sum(-1) / (c * c))
    E, B = fields_batched(tlab, pos, L_base, b0_schedule, alpha_schedule, field_scale, length_scale)
    dv = (q / m) * (g.unsqueeze(-1) * E + torch.cross(proper_v, B, dim=-1))
    return torch.cat([g.unsqueeze(-1), proper_v, dv], dim=-1)


def rhs_batched(
    y, q, m, c, a, eta, L_base, sensitivity_eps, shape_eps, b0_schedule, alpha_schedule
):
    """Full 44-state RHS: orbit + spin + first/second-order variational sensitivities.

    y: (batch, 44) -> (batch, 44). Column layout matches the reference exactly:
    [0]      tlab
    [1:4]    position
    [4:7]    proper velocity
    [7:10]   spin
    [10:13]  d(spin)/d(eta)
    [13:16]  d^2(spin)/d(eta)^2
    [16:23]  d(orbit)/d(ln B0)
    [23:30]  d^2(orbit)/d(ln B0)^2
    [30:37]  d(orbit)/d(ln L)
    [37:44]  d^2(orbit)/d(ln B0)d(ln L)
    """
    tlab = y[..., 0]
    pos = y[..., 1:4]
    proper_v = y[..., 4:7]
    spin_vec = y[..., 7:10]
    spin_eta = y[..., 10:13]
    spin_eta2 = y[..., 13:16]
    orbit_b0 = y[..., 16:23]
    orbit_b02 = y[..., 23:30]
    orbit_L = y[..., 30:37]
    orbit_b0L = y[..., 37:44]

    g = torch.sqrt(1.0 + (proper_v * proper_v).sum(-1) / (c * c))
    vel = proper_v / g.unsqueeze(-1)

    omega = spin_omega_batched(tlab, pos, vel, q, m, c, a, eta, L_base, b0_schedule, alpha_schedule)
    omega_eta = spin_omega_eta_batched(tlab, pos, vel, q, m, c, L_base, b0_schedule, alpha_schedule)

    orbit = y[..., :7]
    h = sensitivity_eps
    plus = orbit + h * orbit_b0 + 0.5 * h * h * orbit_b02
    minus = orbit - h * orbit_b0 + 0.5 * h * h * orbit_b02

    f0 = translation_rhs_batched(orbit, q, m, c, L_base, b0_schedule, alpha_schedule)
    fp = translation_rhs_batched(
        plus, q, m, c, L_base, b0_schedule, alpha_schedule, field_scale=math.exp(h)
    )
    fm = translation_rhs_batched(
        minus, q, m, c, L_base, b0_schedule, alpha_schedule, field_scale=math.exp(-h)
    )

    k = shape_eps
    frp = translation_rhs_batched(
        orbit + k * orbit_L, q, m, c, L_base, b0_schedule, alpha_schedule, length_scale=math.exp(k)
    )
    frm = translation_rhs_batched(
        orbit - k * orbit_L, q, m, c, L_base, b0_schedule, alpha_schedule, length_scale=math.exp(-k)
    )

    corners = {}
    for sb in (-1.0, 1.0):
        for sl in (-1.0, 1.0):
            corner = (
                orbit
                + sb * h * orbit_b0
                + sl * k * orbit_L
                + sb * sl * h * k * orbit_b0L
                + 0.5 * h * h * orbit_b02
            )
            corners[sb, sl] = translation_rhs_batched(
                corner,
                q,
                m,
                c,
                L_base,
                b0_schedule,
                alpha_schedule,
                field_scale=math.exp(sb * h),
                length_scale=math.exp(sl * k),
            )

    d_spin = g.unsqueeze(-1) * torch.cross(omega, spin_vec, dim=-1)
    d_spin_eta = g.unsqueeze(-1) * (
        torch.cross(omega, spin_eta, dim=-1) + torch.cross(omega_eta, spin_vec, dim=-1)
    )
    d_spin_eta2 = g.unsqueeze(-1) * (
        torch.cross(omega, spin_eta2, dim=-1) + 2.0 * torch.cross(omega_eta, spin_eta, dim=-1)
    )

    d_orbit_b0 = (fp - fm) / (2.0 * h)
    d_orbit_b02 = (fp - 2.0 * f0 + fm) / (h * h)
    d_orbit_L = (frp - frm) / (2.0 * k)
    d_orbit_b0L = (
        corners[1.0, 1.0] - corners[1.0, -1.0] - corners[-1.0, 1.0] + corners[-1.0, -1.0]
    ) / (4.0 * h * k)

    return torch.cat(
        [f0, d_spin, d_spin_eta, d_spin_eta2, d_orbit_b0, d_orbit_b02, d_orbit_L, d_orbit_b0L],
        dim=-1,
    )


def magnetic_moment_batched(m, x, v, t, L_base, b0_schedule, alpha_schedule, c):
    """Adiabatic invariant mu = 0.5 * m * gamma^2 * v_perp^2 / |B|. Returns (batch,)."""
    E, B = fields_batched(t, x, L_base, b0_schedule, alpha_schedule)
    Bmag = B.norm(dim=-1).clamp(min=1e-10)
    bhat = B / Bmag.unsqueeze(-1)
    g = gamma_batched(v, c)
    v_par = (v * bhat).sum(-1)
    v_perp2 = (v * v).sum(-1) - v_par * v_par
    return 0.5 * m * (g * g) * v_perp2 / Bmag


def integrate_batch(
    x0: torch.Tensor,
    v0: torch.Tensor,
    spin0: torch.Tensor,
    params: dict,
    device,
    dtype=torch.float64,
) -> dict[str, torch.Tensor]:
    """Integrate a whole batch of trajectories with one RK4 tensor op per stage.

    Args:
        x0, v0, spin0: (batch, 3) initial position, velocity, spin
        params: physics parameters (q, m, c, anomaly, edm_eta, L, dtau, n_steps, ...)
        device, dtype: torch device/dtype for the integration

    Returns:
        dict with "trajectories" (batch, n_steps, 7) = [x,y,z,vx,vy,vz,mu],
        "sensitivity_W" (batch, n_steps, 44) = full state history,
        "diagnostics" per-particle tensors, and final state.
    """
    batch = x0.shape[0]
    q = float(params["q"])
    m = float(params["m"])
    c = float(params["c"])
    a = float(params["anomaly"])
    eta = float(params.get("edm_eta", 0.0))
    L_base = float(params["L"])
    dtau = float(params.get("dtau", 1e-3))
    n_steps = int(params.get("n_steps", 10000))
    sensitivity_eps = float(params.get("sensitivity_eps", 1e-6))
    shape_eps = float(params.get("shape_sensitivity_eps", 1e-6))

    from .physics_core import FieldSchedules

    t_nodes_b0, rates_b0, B0_init = FieldSchedules.load_b0_rate_schedule(params)
    t_nodes_alpha, alpha_vals = FieldSchedules.load_alpha_schedule(params)
    b0_schedule = BatchedB0Schedule(t_nodes_b0, rates_b0, B0_init, device, dtype)
    alpha_schedule = BatchedAlphaSchedule(t_nodes_alpha, alpha_vals, device, dtype)

    x0 = x0.to(device=device, dtype=dtype)
    v0 = v0.to(device=device, dtype=dtype)
    spin0 = spin0.to(device=device, dtype=dtype)
    spin0 = spin0 / spin0.norm(dim=-1, keepdim=True)
    spin_norm0 = spin0.norm(dim=-1)

    g0 = gamma_batched(v0, c)
    u0 = g0.unsqueeze(-1) * v0
    t0 = torch.zeros(batch, device=device, dtype=dtype)

    y = torch.cat(
        [t0.unsqueeze(-1), x0, u0, spin0, torch.zeros(batch, 34, device=device, dtype=dtype)],
        dim=-1,
    )

    trajectories = torch.zeros(batch, n_steps, 7, device=device, dtype=dtype)
    sensitivities = torch.zeros(batch, n_steps, 44, device=device, dtype=dtype)
    mus = torch.zeros(batch, n_steps + 1, device=device, dtype=dtype)
    mus[:, 0] = magnetic_moment_batched(m, x0, v0, t0, L_base, b0_schedule, alpha_schedule, c)

    def _rhs(state):
        return rhs_batched(
            state, q, m, c, a, eta, L_base, sensitivity_eps, shape_eps, b0_schedule, alpha_schedule
        )

    for i in range(n_steps):
        k1 = _rhs(y)
        k2 = _rhs(y + 0.5 * dtau * k1)
        k3 = _rhs(y + 0.5 * dtau * k2)
        k4 = _rhs(y + dtau * k3)
        y = y + (dtau / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        tlab = y[..., 0]
        pos = y[..., 1:4]
        u = y[..., 4:7]
        g = torch.sqrt(1.0 + (u * u).sum(-1) / (c * c))
        v = u / g.unsqueeze(-1)

        mu_i = magnetic_moment_batched(m, pos, v, tlab, L_base, b0_schedule, alpha_schedule, c)
        mus[:, i + 1] = mu_i

        trajectories[:, i, :] = torch.cat([pos, v, mu_i.unsqueeze(-1)], dim=-1)
        sensitivities[:, i, :] = y

    half = n_steps // 2
    mu_half = mus[:, half:]
    mu_mean = mu_half.mean(dim=-1)
    mu0_abs = mus[:, 0].abs().clamp(min=1e-300)
    mu_rms_rel = torch.sqrt(((mu_half - mu_mean.unsqueeze(-1)) ** 2).mean(dim=-1)) / mu0_abs
    mu_max_rel_dev = (mus - mus[:, 0:1]).abs().max(dim=-1).values / mu0_abs

    spin_final = y[..., 7:10]
    spin_norm_final = spin_final.norm(dim=-1)
    spin_norm_drift = (spin_norm_final - spin_norm0).abs() / spin_norm0

    return {
        "trajectories": trajectories,
        "sensitivity_W": sensitivities,
        "final_state": y,
        "mu_rms_rel": mu_rms_rel,
        "mu_max_rel_dev": mu_max_rel_dev,
        "spin_norm_drift": spin_norm_drift,
    }
