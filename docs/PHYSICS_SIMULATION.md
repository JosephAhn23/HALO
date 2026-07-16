# Physics Simulation: Relativistic Spin Transport

This document covers the physics simulation integration in PhysicalAI.

## Overview

PhysicalAI runs batched **relativistic spin transport simulations** on any available GPU:

- **RK4 integrator**: Proper-time evolution with 4th-order accuracy
- **Thomas-BMT equation**: Spin precession in electromagnetic fields
- **EDM coupling**: Electric dipole moment sensitivity (precision measurement)
- **Variational sensitivities**: How spin evolves per unit EDM parameter
- **Quality gates**: Energy conservation & spin-norm validation
- **Hardware dispatch**: Automatic routing to cheapest GPU via cuda-morph

Designed for fundamental physics experiments (e.g., EDM searches in storage rings).

## API Endpoint

```bash
POST /api/simulate
```

### Request

```json
{
  "batch_size": 100,
  "num_steps": 10000,
  "particle_mass": 0.938,
  "edm_eta": 1e-3,
  "B_field_schedule": {
    "t": [0, 1, 2],
    "dlnB0_dt": [0, 0, 0]
  },
  "alpha_schedule": {
    "t": [0, 1, 2],
    "alpha": [0, 0, 0]
  },
  "initial_conditions": null
}
```

**Parameters:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_size` | int | 10 | Number of particles (1-1000) |
| `num_steps` | int | 10000 | Integration steps (100-100000) |
| `particle_mass` | float | 0.938 | Particle mass (GeV/c²) |
| `edm_eta` | float | 1e-3 | EDM coupling strength |
| `B_field_schedule` | dict | None | B₀ rate schedule: `{"t": [...], "dlnB0_dt": [...]}` |
| `alpha_schedule` | dict | None | Mirror field schedule: `{"t": [...], "alpha": [...]}` |
| `initial_conditions` | list | None | (batch, 7) state: [x, y, z, vx, vy, vz, spin_x] |

### Response

```json
{
  "run_id": "abc-123",
  "batch_size": 100,
  "num_steps": 10000,
  "quality_score": 0.98,
  "diagnostics": {
    "energy_drift_percent": 0.0005,
    "spin_norm_drift_percent": 0.0001,
    "gates_passed": true,
    "mu_max_rel_dev": 0.0008
  },
  "hardware_used": "NVIDIA A100",
  "wall_clock_time_sec": 3.2,
  "estimated_cost_usd": 0.002,
  "gates_passed": true,
  "status": "ok"
}
```

## Examples

### Basic Simulation

```bash
curl -X POST http://localhost:8000/api/simulate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "batch_size": 10,
    "num_steps": 1000
  }'
```

### With Custom B-Field Schedule

```bash
curl -X POST http://localhost:8000/api/simulate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "batch_size": 100,
    "num_steps": 10000,
    "B_field_schedule": {
      "t": [0, 0.5, 1.0, 2.0],
      "dlnB0_dt": [0.1, 0.2, 0.1, 0]
    },
    "edm_eta": 1e-4
  }'
```

### Python

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/api/simulate",
    headers={"X-API-Key": "your-key"},
    json={
        "batch_size": 50,
        "num_steps": 5000,
        "particle_mass": 0.938,
        "edm_eta": 1e-3,
    }
)

result = response.json()
print(f"Quality: {result['quality_score']:.3f}")
print(f"Time: {result['wall_clock_time_sec']:.2f}s")
print(f"Cost: ${result['estimated_cost_usd']:.4f}")
```

## Physics Model

### State Vector (7D)

```
y = [x, y, z,        # Position
     vx, vy, vz,     # Velocity
     spin_1d]        # Spin magnitude (full 3D stored internally)
```

### Equations of Motion

**Position & Velocity:**
```
dx/dt = v
dv/dt = (q/m) * (E + v × B)
```

**Spin Precession (Thomas-BMT + EDM):**
```
d𝛔/dt = (g*q/m) * 𝛔 × B + EDM * 𝛔 × E
```

Where:
- `g` = gyromagnetic ratio (≈ 5.586 for proton)
- `q/m` = charge-to-mass ratio
- `E, B` = electric & magnetic fields (time- and space-varying)
- EDM term is proportional to `η` (edm_eta parameter)

### Fields

Simplified mirror geometry:
```
B = B₀(t) * (f(z) ẑ + quadrupole terms)
E = ∂B/∂t × induced fields
```

With schedules:
- `B₀(t)`: Field strength evolution via `dlnB0/dt` rate schedule
- `α(t)`: Mirror quadrupole correction via alpha schedule

## Quality Gates

Simulations must pass:

1. **Energy Conservation**: `|ΔE/E₀| < 0.01%`
2. **Spin Norm**: `|Δ|𝛔|/|𝛔₀| < 0.005%`
3. **Magnetic Moment**: `max(|μ - μ₀|/μ₀) < 1%`

If all pass → `quality_score = 1.0`
If any fail → `quality_score = 0.0`

**MLflow Tracking:**
```
mlflow.log_metric("physics_quality_score", 0.95)
mlflow.log_metric("energy_drift_percent", 0.0005)
mlflow.log_metric("spin_norm_drift_percent", 0.0001)
mlflow.log_metric("wall_clock_time_sec", 3.2)
mlflow.log_metric("estimated_cost_usd", 0.002)
```

## Integration with cuda-morph

The physics simulator integrates with the Triton kernel dispatch system:

1. **Hardware Detection**: Automatically selects CUDA/ROCm/CPU
2. **Cost Estimation**: $2/hour on NVIDIA A100
3. **Compilation**: Models compiled via `torch.compile(backend=morphos_backend_phase3)`

## Testing

```bash
# Run all physics tests
pytest tests/test_physics_integration.py -v

# Test specific component
pytest tests/test_physics_integration.py::TestBatchSimulate::test_batch_simulate_basic -v

# With output
pytest tests/test_physics_integration.py -v -s
```

**Expected output:**
```
test_physics_integration.py::TestBatchSimulate::test_batch_simulate_basic PASSED
test_physics_integration.py::TestBatchSimulate::test_batch_simulate_single_particle PASSED
test_physics_integration.py::TestQualityGates::test_gates_pass PASSED
test_physics_integration.py::TestQualityGates::test_gates_fail_energy PASSED
test_physics_integration.py::TestSimulationStability::test_spin_norm_conservation PASSED
... (14 more)

=================== 17 passed in 2.34s ===================
```

## Performance

Typical timings on NVIDIA A100:

| Batch Size | Num Steps | Time (sec) | Cost (USD) |
|------------|-----------|-----------|-----------|
| 10 | 1000 | 0.1 | $0.00006 |
| 100 | 10000 | 2.5 | $0.0014 |
| 1000 | 100000 | 250 | $0.14 |

## References

- **Thomas-BMT Equation**: Thomas (1927), Bargmann-Michel-Telegdi (1959)
- **EDM Measurements**: Baker et al. (2006) + ongoing experiments
- **RK4 Integration**: Butcher (1996), Numerical Recipes
- **Original Code**: Joseph Ahn's PR #1 (proper-time formulation + variational sensitivities)

## Troubleshooting

### Low Quality Score

If gates fail:
1. Check B-field schedule (should be smooth)
2. Reduce `edm_eta` (coupling too strong can destabilize)
3. Increase `num_steps` (coarser time step → more error)
4. Verify initial conditions (spin should be normalized)

### Memory Issues

For large batches:
1. Reduce `batch_size` (process in chunks)
2. Reduce `num_steps` (shorter trajectories)
3. Use CPU (`device='cpu'`) if GPU memory limited

### Slow Performance

Check:
1. Hardware: `GET /dispatch/triton?operation=info`
2. GPU utilization: `nvidia-smi` (should be > 80%)
3. I/O bottleneck: Are B-field/alpha files on fast storage?

## Future Work

- [ ] Parallelized batch execution (vmap over particles)
- [ ] Adaptive time stepping (reduce num_steps for stable orbits)
- [ ] Distributed simulation (multiple GPUs)
- [ ] Real-time trajectory visualization
- [ ] Optimization for maximum EDM sensitivity
