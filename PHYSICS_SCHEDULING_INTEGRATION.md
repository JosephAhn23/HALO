# Physics + Fleet Scheduling Integration

This document explains what was built in this session, integrating two Dynamo task
submissions ([PR #1: gyro-adiabatic Faraday simulation](https://github.com/handshake-project-dynamo/dynamo-4fd5425-scientific-computing-and-domain-science/pull/1),
[PR #2: fleet timer reconciler](https://github.com/handshake-project-dynamo/dynamo-8b56404-systems-infrastructure-and-operations/pull/2))
into the HALO application, and what got fixed along the way.

## 1. Physics simulation layer

**Goal:** batched relativistic spin-transport simulation, callable via Python API and
HTTP, with quality gates and observability wired in.

**What already existed** (from an earlier pass): `simulation/physics_core.py`,
`simulation/quality_gates.py`, the `/api/simulate` FastAPI endpoint, MLflow logging.
Added in this session:

- `storage/physics_delta.py` — Delta Lake ingestion for simulation results (feature
  store: run_id, quality_score, hardware, cost, per-run diagnostics)
- `mlops/ragas_tracker.py::PhysicsEvaluator` — trajectory stability scoring +
  MLflow logging for physics runs, alongside the existing RAG-focused RAGAS tracker
- `inference/cuda_dispatch/dispatcher.py::dispatch_physics()` — routes physics
  simulations through `torch.compile(backend=morphos_backend_phase3)`, mirroring the
  existing LLM dispatch path
- `tests/test_physics_api.py` — request/response model validation for the endpoint
- README section documenting the physics pipeline end-to-end

### The batching problem, and the fix

The original `batch_simulate()` accepted a `batch_size` argument but ran a **Python
`for` loop calling a numpy single-particle integrator once per particle** — not
batching, just N sequential simulations dressed up as one call. It also had two
shape bugs that would have raised `ValueError` at runtime the first time anyone
actually ran it with `torch` installed (the `trajectories`/`sensitivity_W` arrays
were declared at one width and then assigned arrays of a different width).

`simulation/batched_integrator.py` (new) replaces this with a genuine tensor-batched
RK4 integrator: every stage of every RK4 step evaluates the full 44-state ODE —
orbit, Thomas-BMT spin precession, EDM coupling, and all first/second-order
variational sensitivities (∂/∂lnB0, ∂/∂lnL, ∂/∂η, and the mixed second derivatives)
— as **one set of torch tensor ops across the whole particle batch**. N particles
share one dispatch instead of N sequential loops. The only inherently sequential
part is the time axis (true of any explicit integrator, batched or not).

This included vectorizing the B0(t)/α(t) field-schedule interpolation itself
(`BatchedB0Schedule`, `BatchedAlphaSchedule`), since each particle's own lab time
can drift slightly from the others depending on its velocity history — schedule
lookups can't be done once per step for the whole batch, they need a
batch-of-query-times evaluation.

### How correctness was verified — not just claimed

The rewrite's ground truth is your actual accepted PR
(`dynamo-4fd5425-scientific-computing-and-domain-science`, `task/solution/simulate.py`).
Before treating that file as authoritative, I:

1. Discovered no `gh` CLI was available and unauthenticated `WebFetch` 404'd on the
   private repo (private repos 404 rather than 403 to unauthenticated requests, by
   design — no way to distinguish "doesn't exist" from "no access").
2. Downloaded and installed `gh` v2.96.0 directly from GitHub's release artifacts
   (no Homebrew available), since you asked to actually install it.
3. Found `gh` was already authenticated as you (`JosephAhn23`, `repo` scope, saved
   in the macOS keychain from a prior session).
4. Pulled the real PR metadata, commit list, and file list via `gh pr view`.
5. Fetched the actual file content from the PR's `submission` branch via
   `gh api .../contents/task/solution/simulate.py?ref=submission` and diffed it
   byte-for-byte against the local copy — **exit code 0, identical**. Same check
   for `reconcile.py` against PR #2.

`tests/test_physics_reference_validation.py` (new) then:

- Compares `BatchedB0Schedule`/`BatchedAlphaSchedule` against the reference's scalar
  `make_rate_b0_funs`/`make_lin_funs` at several query points (interior, boundary,
  extrapolated) — matches to 1e-9.
- Runs both the reference `integrate()` and the new batched engine on the same
  sample params (`task/environment/data/params.json`, real multi-node B0/alpha
  schedules), reduced to 300 steps for test speed, and compares the full 44-state
  final vector — matches within the reference task's own 0.5% componentwise
  tolerance (`PHASE_VECTOR_KEYS`: spin, spin-eta sensitivity/curvature, logB0
  sensitivity/curvature, logL sensitivity, mixed sensitivity).

I installed `torch`/`numpy`/`pytest` in this sandbox (none were present) specifically
to run these tests rather than leave the claim unverified. **31/31 tests pass**,
including a live run of `batch_simulate(batch_size=8, num_steps=200)` completing in
0.8s with `quality_score=1.0` and drift metrics at machine precision (~1e-13).

### Bugs fixed incidentally

- `diagnostics["energy_drift_percent"]` was never populated by `batch_simulate()`,
  even though `QualityGates.validate()` and `track_to_mlflow()` both read it — so
  that gate silently always saw `0.0` and always "passed". Now populated from the
  magnetic-moment adiabatic-invariance deviation (this system's real conservation
  check), in the same percent-units convention as `spin_norm_drift_percent`.
- Two existing tests called `batch_simulate(particle_mass=..., edm_eta=...)` —
  kwargs that never existed on the function's actual signature (it takes a `params`
  dict). Fixed to build the params dict the function has always expected.
- A test line (`ic[:, 6] /= torch.norm(ic[:, 6], dim=1, keepdim=True)`) called
  `torch.norm(..., dim=1)` on a 1-D tensor slice, which is invalid and would have
  raised `IndexError`. Removed — the normalization already happens internally in
  `batch_simulate`.

## 2. Fleet scheduling layer

**Goal:** wire the deterministic fleet timer reconciler (DST-aware job scheduling)
into HALO as a callable service, not just a standalone task submission.

`scheduling/reconciler.py::FleetReconciler` wraps your actual
`dynamo-8b56404-systems-infrastructure-and-operations/task/solution/reconcile.py`
(confirmed byte-identical to the PR's `submission` branch the same way as above) via
subprocess — this is deliberately **not** reimplemented, since the reconciler's
correctness depends on exact event-ordering and tie-breaking semantics that are easy
to subtly break in a rewrite, and it doesn't need tensor batching (it's inherently
sequential event processing, not numerical integration over a particle ensemble).

Added:

- `scheduling/metrics.py::SchedulingMetrics` — extracts dispatch latency (mean/max/p95),
  success rate, and a quality score from a reconciliation result
- `POST /api/schedule` — takes workers/jobs/window, returns the dispatch plan +
  MLflow-logged metrics
- `tests/test_scheduling_integration.py` — 9 tests covering the sample fleet config,
  output-shape validation, metric extraction, and timezone-aware job scheduling

Verified live against your real sample data
(`task/environment/data/fleet.json`) through the HALO wrapper: 6 dispatches, 5
succeeded, 1 missed, 0 failed, 0 blocked — real output from your real algorithm, not
a mock.

## 3. Getting this pushed

The local `HALO-main` directory had been `git init`'d fresh partway through this
session rather than cloned, so it had no shared history with the real
`JosephAhn23/HALO` remote — pushing directly was rejected (`fetch first`). Since a
force-push would have destroyed real project history already on GitHub (governance
work, the cuda-morph merge, RAG ablation work, etc.), instead:

1. Fetched `origin/main` and confirmed my first local commit's diff against it was
   **purely additive** (no modified/deleted pre-existing files) — safe to rebuild on.
2. Checked out a new branch from `origin/main`.
3. Replayed each of the 3 local commits' **exact file states** (not diffs, to avoid
   fighting cherry-pick's add/add conflicts against an unrelated-history root commit)
   onto that branch, in order, with matching commit messages.
4. Diffed the rebuilt branch against the original work-in-progress branch — empty,
   confirming no content was lost in the replay.
5. Re-ran all 31 tests on the rebuilt branch — still green.
6. Pushed as a fast-forward: `f661fe8..b7aa4c3` on `main`. No force-push, no lost
   history.

## Commits (in order, on `main`)

- `15daf5a` — Add physics simulation layer to PhysicalAI (Delta Lake, RAGAS physics
  eval, hardware dispatch, API tests, README)
- `777f95c` — Integrate fleet timer reconciler into HALO (`scheduling/` module,
  `/api/schedule`, tests)
- `b7aa4c3` — Rewrite physics simulator as genuine tensor-batched RK4 integrator
  (the batching fix + reference validation + bug fixes above)
