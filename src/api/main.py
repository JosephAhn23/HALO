"""
FastAPI gateway - realtime + batch inference endpoints.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from typing import Any

import torch
from fastapi import BackgroundTasks, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field, field_validator

from src.agents.orchestrator import run_pipeline
from src.api.batch import enqueue_batch_job
from src.api.websocket_streaming import router as websocket_router
from src.inference.cuda_dispatch.dispatcher import dispatch_model, get_hardware_info
from src.scheduling import FleetReconciler, parse_fleet_config
from src.scheduling.metrics import SchedulingMetrics, track_dispatch_plan
from src.simulation import QualityGates, batch_simulate, track_to_mlflow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CORS — restrict to an explicit allowlist in production.
# Set CORS_ORIGINS="https://app.example.com,https://admin.example.com" to
# override. Falls back to localhost-only for local development.
# ---------------------------------------------------------------------------
_raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080")
_allow_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# ---------------------------------------------------------------------------
# Optional API-key authentication.
# Set API_KEY env var to require a bearer key on all mutating endpoints.
# When unset, auth is skipped (dev mode).
# ---------------------------------------------------------------------------
_API_KEY = os.getenv("API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _require_api_key(key: str | None = Security(_api_key_header)) -> None:
    if _API_KEY and key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _constitutional_api_gate(text: str, endpoint: str) -> None:
    """Mitchell-style + behavioral pre-check before LLM-backed paths (opt-in)."""
    if os.getenv("ENABLE_API_CONSTITUTIONAL_GATE", "").lower() not in ("1", "true", "yes"):
        return
    from src.safety.constitutional_filter import (
        audit_gateway_decision,
        constitutional_filter_query,
    )

    r = constitutional_filter_query(text)
    audit_gateway_decision(
        endpoint=endpoint,
        allowed=r.allowed,
        reason=r.reason,
        extra={"text_chars": len(text)},
    )
    if not r.allowed:
        raise HTTPException(status_code=400, detail=r.to_http_detail())


app = FastAPI(title="HALO")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

app.include_router(websocket_router)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    stream: bool = False
    session_id: str | None = Field(
        default=None,
        max_length=256,
        description="Optional long-running research session for ResearchLog injection.",
    )


class BatchRequest(BaseModel):
    queries: list[str] = Field(..., min_length=1, max_length=100)


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4096)
    top_k: int = Field(default=5, ge=1, le=100)
    rerank: bool = True


class IngestRequest(BaseModel):
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_size: int = 512

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty or whitespace-only")
        return v


class TritonDispatchRequest(BaseModel):
    """Request to compile + dispatch inference through Triton kernels."""

    use_triton: bool = Field(default=True, description="Use Triton backend")
    operation: str = Field(
        default="info", description="'info' to get hardware, 'dispatch' to compile"
    )


class TritonDispatchResponse(BaseModel):
    """Response with hardware info and dispatch status."""

    hardware: dict[str, Any]
    status: str
    message: str


class PhysicsSimulationRequest(BaseModel):
    """Request for batched physics simulation."""

    batch_size: int = Field(default=10, ge=1, le=1000)
    num_steps: int = Field(default=10000, ge=100, le=100000)
    initial_conditions: list | None = Field(default=None, description="(batch, 7) initial states")
    B_field_schedule: dict | None = Field(default=None, description="B0 rate schedule")
    alpha_schedule: dict | None = Field(default=None, description="Mirror field schedule")
    particle_mass: float = Field(default=0.938, description="GeV/c^2")
    edm_eta: float = Field(default=1e-3, description="EDM coupling strength")


class PhysicsSimulationResponse(BaseModel):
    """Response from physics simulation."""

    run_id: str
    batch_size: int
    num_steps: int
    quality_score: float
    diagnostics: dict[str, Any]
    hardware_used: str
    wall_clock_time_sec: float
    estimated_cost_usd: float
    gates_passed: bool
    status: str


class SchedulingRequest(BaseModel):
    """Request for fleet scheduling reconciliation."""

    workers: list = Field(
        ..., description="Worker definitions with id, cpu, memory, labels, blackouts"
    )
    jobs: list = Field(
        ..., description="Job definitions with timezone, weekdays, times, dependencies, etc."
    )
    window_start: str = Field(
        ..., description="RFC 3339 UTC window start (e.g., 2025-01-01T00:00:00Z)"
    )
    window_end: str = Field(..., description="RFC 3339 UTC window end")
    failed_attempts: list | None = Field(default=None, description="Simulated failures for testing")


class SchedulingResponse(BaseModel):
    """Response from fleet scheduling."""

    run_id: str
    dispatch_count: int
    quality_score: float
    summary: dict[str, Any]
    dispatches: list
    terminal: list
    status: str


@app.post("/retrieve", dependencies=[Security(_require_api_key)])
async def retrieve(request: RetrieveRequest):
    """Retrieve and optionally rerank documents — used by the MCP server."""
    _constitutional_api_gate(request.query, "/retrieve")
    from src.agents.orchestrator import get_pipeline

    pipeline = get_pipeline()
    chunks = await asyncio.to_thread(pipeline.retriever.retrieve, request.query)
    if request.rerank:
        chunks = await asyncio.to_thread(pipeline.reranker.rerank, request.query, chunks)
    return {"results": chunks[: request.top_k]}


_ingestion_pipeline = None
_ingestion_lock = threading.Lock()


def _get_ingestion_pipeline():
    global _ingestion_pipeline
    if _ingestion_pipeline is None:
        with _ingestion_lock:
            if _ingestion_pipeline is None:
                from src.ingestion.pipeline import IngestionPipeline

                _ingestion_pipeline = IngestionPipeline()
    return _ingestion_pipeline


@app.post("/ingest", dependencies=[Security(_require_api_key)])
async def ingest(request: IngestRequest):
    """Ingest a document into the FAISS index — used by the MCP server."""
    doc_id = str(uuid.uuid4())
    source = (request.metadata or {}).get("source", doc_id)
    pipeline = _get_ingestion_pipeline()
    await asyncio.to_thread(
        pipeline.ingest_documents,
        [{"id": doc_id, "text": request.content, "source": source}],
    )
    return {"status": "ingested", "doc_id": doc_id, "source": source}


@app.post("/query", dependencies=[Security(_require_api_key)])
async def query_realtime(request: QueryRequest):
    _constitutional_api_gate(request.query, "/query")
    result = await asyncio.to_thread(run_pipeline, request.query, request.session_id)
    if result.get("error"):
        raise HTTPException(status_code=502, detail=result["error"])
    resp = result["response"]
    out: dict[str, Any] = {
        "answer": resp["answer"],
        "sources": resp.get("sources") or [],
        "tokens_used": resp.get("tokens_used", 0),
    }
    if resp.get("behavioral_blocked") is not None:
        out["behavioral_blocked"] = resp["behavioral_blocked"]
    if resp.get("behavioral_reasons") is not None:
        out["behavioral_reasons"] = resp["behavioral_reasons"]
    if resp.get("grounding_confidence") is not None:
        out["grounding_confidence"] = resp["grounding_confidence"]
    if resp.get("source_attributions") is not None:
        out["source_attributions"] = resp["source_attributions"]
        out["speculative_sentence_count"] = resp.get("speculative_sentence_count", 0)
    if resp.get("constitutional_score") is not None:
        out["constitutional_score"] = resp["constitutional_score"]
        out["constitutional_passed"] = resp.get("constitutional_passed")
    if resp.get("quality_alert"):
        out["quality_alert"] = resp["quality_alert"]
    if resp.get("effective_faithfulness") is not None:
        out["effective_faithfulness"] = resp["effective_faithfulness"]
    if resp.get("answer_with_provenance"):
        out["answer_with_provenance"] = resp["answer_with_provenance"]
    if resp.get("adversarial_consensus") is not None:
        out["adversarial_consensus"] = resp["adversarial_consensus"]
    if resp.get("consensus_hitl") is not None:
        out["consensus_hitl"] = resp["consensus_hitl"]
    if resp.get("consensus_score") is not None:
        out["consensus_score"] = resp["consensus_score"]
    if resp.get("consensus_discrepancy") is not None:
        out["consensus_discrepancy"] = resp["consensus_discrepancy"]
    if resp.get("truth_committee") is not None:
        out["truth_committee"] = resp["truth_committee"]
    return out


@app.post("/batch", dependencies=[Security(_require_api_key)])
async def query_batch(request: BatchRequest, background_tasks: BackgroundTasks):
    for q in request.queries:
        _constitutional_api_gate(q, "/batch")
    job_id = str(uuid.uuid4())
    # BackgroundTasks runs sync callables in a thread pool automatically,
    # so enqueue_batch_job (which does blocking Redis + Celery I/O) is safe here.
    background_tasks.add_task(enqueue_batch_job, job_id, request.queries)
    return {"job_id": job_id, "status": "queued"}


@app.get("/batch/{job_id}", dependencies=[Security(_require_api_key)])
async def get_batch_status(job_id: str):
    from src.api.batch import get_job_status

    status = get_job_status(job_id)
    if status.get("error"):
        raise HTTPException(status_code=404, detail=status["error"])
    return status


@app.post("/dispatch/triton", dependencies=[Security(_require_api_key)])
async def dispatch_triton(request: TritonDispatchRequest):
    """
    Dispatch inference through Triton kernels (cuda-morph integration).

    - operation='info': Get available hardware
    - operation='dispatch': Compile a test model
    """
    try:
        hardware = get_hardware_info()

        if request.operation == "info":
            return TritonDispatchResponse(
                hardware=hardware, status="ok", message="Hardware info retrieved"
            )
        elif request.operation == "dispatch":
            if not request.use_triton:
                return TritonDispatchResponse(
                    hardware=hardware, status="ok", message="Triton dispatch disabled"
                )

            logger.info("Testing Triton backend compilation...")
            test_model = torch.nn.Linear(128, 64)
            dispatch_model(test_model, use_triton=True)

            return TritonDispatchResponse(
                hardware=hardware, status="ok", message="Model compiled with morphos_backend_phase3"
            )
        else:
            raise ValueError(f"Unknown operation: {request.operation}")

    except Exception as e:
        logger.error(f"Dispatch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simulate", dependencies=[Security(_require_api_key)])
async def simulate_physics(request: PhysicsSimulationRequest):
    """
    Batched relativistic spin transport simulation.

    Runs N particle trajectories in parallel using:
    - RK4 integrator (proper-time evolution)
    - Thomas-BMT spin transport with EDM coupling
    - Variational sensitivity tensors
    - Quality gates (energy/spin conservation)
    - Hardware dispatch via cuda-morph

    Returns trajectories + diagnostics + quality score + hardware info + estimated cost.
    """
    run_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        hardware_info = get_hardware_info()
        logger.info(
            f"Physics simulation {run_id}: batch_size={request.batch_size}, "
            f"hardware={hardware_info.get('device_name', 'cpu')}"
        )

        # Convert initial conditions to torch tensor
        initial_conditions = None
        if request.initial_conditions:
            initial_conditions = torch.tensor(request.initial_conditions, dtype=torch.float32)

        # Build params dict
        params = {
            "q": 1.0,
            "m": request.particle_mass,
            "c": 299.792458,
            "anomaly": 1.793,
            "edm_eta": request.edm_eta,
            "L": 0.5,
            "B0_initial": 1.0,
            "dtau": 1e-3,
            "n_steps": request.num_steps,
            "sensitivity_eps": 1e-6,
            "shape_sensitivity_eps": 1e-6,
            "diagnostic_time": 0.0,
        }

        # Run simulation
        result = await asyncio.to_thread(
            batch_simulate,
            request.batch_size,
            request.B_field_schedule,
            request.alpha_schedule,
            initial_conditions,
            params,
            request.num_steps,
        )

        wall_clock_time = time.time() - start_time

        # Quality validation
        gates_passed, reason, quality_score = QualityGates.validate(result.get("diagnostics", {}))

        # MLflow logging
        await asyncio.to_thread(
            track_to_mlflow,
            run_id,
            {**result, "batch_size": request.batch_size, "num_steps": request.num_steps},
            hardware_info,
            wall_clock_time,
        )

        estimated_cost = (wall_clock_time / 3600.0) * 2.0  # $2/hr for A100

        response = PhysicsSimulationResponse(
            run_id=run_id,
            batch_size=request.batch_size,
            num_steps=request.num_steps,
            quality_score=quality_score,
            diagnostics=result.get("diagnostics", {}),
            hardware_used=hardware_info.get("device_name", "cpu"),
            wall_clock_time_sec=wall_clock_time,
            estimated_cost_usd=estimated_cost,
            gates_passed=gates_passed,
            status="ok" if gates_passed else "quality_warning",
        )

        logger.info(
            f"Simulation {run_id} complete: quality={quality_score:.3f}, "
            f"time={wall_clock_time:.2f}s, cost=${estimated_cost:.2f}"
        )

        return response

    except Exception as e:
        logger.error(f"Simulation {run_id} failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/schedule", dependencies=[Security(_require_api_key)])
async def schedule_fleet(request: SchedulingRequest):
    """
    Deterministic fleet timer reconciliation.

    Converts job calendars (with timezone-aware schedules, DST handling, retry policies)
    and worker constraints into an exact dispatch plan with:
    - All releases expanded from local wall times to UTC
    - Dependency resolution (job A must complete before job B starts)
    - Resource contention resolution (CPU/memory allocation across workers)
    - Mutex exclusion (jobs that cannot run simultaneously)
    - Retry scheduling with exponential backoff
    - Deadline enforcement and missed job detection

    Input: fleet config with workers, jobs, scheduling window
    Output: dispatch plan with execution order, terminal states, summary metrics
    """
    run_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        logger.info(
            f"Fleet scheduling {run_id}: {len(request.jobs)} jobs, "
            f"{len(request.workers)} workers"
        )

        # Build fleet config
        fleet_config = parse_fleet_config(
            workers=request.workers,
            jobs=request.jobs,
            window_start=request.window_start,
            window_end=request.window_end,
            failed_attempts=request.failed_attempts,
        )

        # Run reconciliation
        reconciler = FleetReconciler()
        plan = await asyncio.to_thread(reconciler.reconcile, fleet_config)

        wall_clock_time = time.time() - start_time

        # Extract metrics
        quality_score = SchedulingMetrics.dispatch_quality_score(plan)
        summary = plan.get("summary", {})

        # Log to MLflow
        await asyncio.to_thread(
            track_dispatch_plan,
            run_id,
            plan,
            request.window_start,
            request.window_end,
        )

        response = SchedulingResponse(
            run_id=run_id,
            dispatch_count=summary.get("dispatch_count", 0),
            quality_score=quality_score,
            summary=summary,
            dispatches=plan.get("dispatches", []),
            terminal=plan.get("terminal", []),
            status="ok" if quality_score > 0 else "degraded",
        )

        logger.info(
            f"Scheduling {run_id} complete: quality={quality_score:.3f}, "
            f"time={wall_clock_time:.2f}s, dispatches={summary.get('dispatch_count', 0)}"
        )

        return response

    except Exception as e:
        logger.error(f"Scheduling {run_id} failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    checks: dict[str, str] = {"api": "ok"}

    try:
        from src.api.batch import redis_client

        redis_client.ping()
        checks["redis"] = "ok"
    except Exception:
        checks["redis"] = "unavailable"

    overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    return {"status": overall, "checks": checks}
