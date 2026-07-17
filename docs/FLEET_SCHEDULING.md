# Fleet Scheduling: Deterministic Timer Reconciliation

Deterministic job scheduler for worker fleets with timezone-aware calendars, resource
constraints, and failure recovery. Wraps a reference reconciler implementation
(`dynamo-8b56404-systems-infrastructure-and-operations/task/solution/reconcile.py`) via
`src/scheduling/reconciler.py`.

## Python API

```python
from src.scheduling import FleetReconciler, parse_fleet_config

fleet_config = parse_fleet_config(
    workers=[
        {"id": "w1", "cpu": 8, "memory": 16, "labels": ["compute"], "blackouts": []},
        {"id": "w2", "cpu": 4, "memory": 8, "labels": ["io"], "blackouts": []},
    ],
    jobs=[
        {
            "id": "job_daily",
            "timezone": "America/New_York",           # Timezone-aware scheduling
            "weekdays": [0, 1, 2, 3, 4],              # Monday-Friday
            "times": ["09:00", "17:00"],              # Twice daily
            "fold_policy": "first",                   # DST fall: pick earlier UTC
            "gap_policy": "shift",                    # DST spring: adjust nonexistent times
            "duration_sec": 3600,
            "cpu": 2,
            "memory": 4,
            "labels": ["compute"],
            "priority": 100,
            "max_lateness_sec": 3600,                 # Can start up to 1hr late
            "mutex": None,
            "dependencies": ["upstream_job"],         # Requires upstream to complete
            "max_attempts": 3,
            "retry_delay_sec": 60,                    # Exponential: 60s, 120s, 240s
            "coalesce": True,                         # Skip older runs when new one releases
        }
    ],
    window_start="2025-01-01T00:00:00Z",
    window_end="2025-01-31T23:59:59Z",
    failed_attempts=[],                               # Simulate failures: [{occurrence, attempt}]
)

reconciler = FleetReconciler()
plan = reconciler.reconcile(fleet_config)
# Returns: {dispatches, terminal, summary}
```

## HTTP API

```bash
curl -X POST http://localhost:8000/api/schedule \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{
    "workers": [...],
    "jobs": [...],
    "window_start": "2025-01-01T00:00:00Z",
    "window_end": "2025-01-31T23:59:59Z"
  }'
# Returns: {run_id, dispatch_count, quality_score, summary, dispatches, terminal, status}
```

## Pipeline

1. **Calendar Expansion** — Convert job local times to UTC, handling IANA timezone rules
2. **DST Handling** — Ambiguous fall transitions (pick first/second/both), nonexistent spring gaps (skip/shift)
3. **Release Materialization** — Generate occurrence IDs (job@UTC_release_time)
4. **Dependency Fixation** — Lock dependency targets at release time
5. **Event-Driven Replay** — Process completions → retries → releases → deadlines in order
6. **Batch Selection** — At each event, lexicographically optimize:
   - Sum of job priorities
   - Sum of waiting time (how long jobs have waited)
   - Number of dispatches
7. **Resource Allocation** — Respect CPU, memory, labels, blackouts, mutexes
8. **Terminal Transitions** — Record succeeded/failed/missed/blocked/coalesced states
9. **Fence Counting** — Monotonic counter per job for idempotency/replay safety

## Output Schema

**Dispatch record:**
```json
{
  "time": "2025-01-01T09:15:00Z",
  "occurrence": "job_daily@2025-01-01T14:00:00Z",
  "attempt": 1,
  "worker": "w1",
  "fence": 5
}
```

**Terminal record:**
```json
{
  "time": "2025-01-01T10:15:00Z",
  "occurrence": "job_daily@2025-01-01T14:00:00Z",
  "state": "succeeded|failed|missed|blocked|coalesced",
  "attempts": 1
}
```

**Summary:**
```json
{
  "dispatch_count": 42,
  "succeeded": 40,
  "failed": 1,
  "missed": 0,
  "blocked": 1,
  "coalesced": 0,
  "unfinished": 0
}
```

## Metrics & Observability

Tracked in MLflow via `src/scheduling/metrics.py`:
- `scheduling_quality_score` — 1.0 if all jobs succeeded, 0.0 if any failed/missed
- `scheduling_dispatch_count` — Total dispatches
- `scheduling_success_rate` — Succeeded / Dispatched
- `scheduling_dispatch_latency_mean_sec` — Avg time from release to dispatch
- `scheduling_dispatch_latency_p95_sec` — 95th percentile latency

## Tests

```bash
pytest tests/test_scheduling_integration.py -v
```
