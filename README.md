<div align="center">

# HALO

LLMOps platform with batched physics simulation and deterministic fleet scheduling.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![AWS](https://img.shields.io/badge/AWS-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com)
[![Azure](https://img.shields.io/badge/Azure-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white)](https://azure.microsoft.com)

<br>

![Pipeline Architecture](assets/pipeline_flow.png)

</div>

---

## What It Does

An end-to-end RAG + LLMOps platform, plus two extra domains (batched physics simulation,
deterministic fleet scheduling) integrated into the same stack.

| Stage | Implementation |
|:---|:---|
| **Ingest** | Chunking, FAISS indexing, WARC parsing, Spark ML feature engineering |
| **Search** | Two-stage FAISS bi-encoder + cross-encoder reranking, sub-50ms locally |
| **Generate** | LangGraph multi-agent pipeline, token streaming, vLLM self-hosted inference |
| **Fine-tune** | QLoRA, RLHF/PPO, GRPO, Ray fault-tolerant distributed training |
| **Evaluate** | RAGAS metrics + MLflow, wired into CI as a merge-blocking quality gate |
| **Serve** | TensorRT-LLM, ONNX/Triton, NVIDIA NIM, MoE expert parallelism, custom CUDA kernels |
| **Secure / Govern** | Red-team suite, jailbreak classifier, model cards, bias eval, PII redaction, audit log |
| **Experiment / Causal** | A/B router, sequential testing, CUPED, Double ML, uplift modeling |
| **Physics** | Batched relativistic spin-transport simulator — see [docs/PHYSICS_SIMULATION.md](docs/PHYSICS_SIMULATION.md) |
| **Scheduling** | Deterministic fleet timer reconciler — see [docs/FLEET_SCHEDULING.md](docs/FLEET_SCHEDULING.md) |
| **Data** | Delta Lake medallion pipeline, feature store, MLflow model registry |
| **Deploy** | Docker Compose, Kubernetes, Terraform (AWS), Azure Bicep/Terraform |
| **Connect** | MCP server exposing retrieve/ingest/evaluate/benchmark for Claude Desktop / Cursor |

Full per-module breakdown: [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md).

---

## Performance

| Metric | Value |
|:---|:---|
| Vector search latency | `< 5 ms` (single-node FAISS) |
| Reranking latency | `~40 ms` (cross-encoder, CPU) |
| End-to-end p50 / p99 | `3,284 ms` / `6,238 ms` (GPT-4o-mini) |
| Cost-aware router savings | `80.5%` vs. always-`gpt-4o` |
| RAGAS faithfulness / relevancy | `0.847` / `0.823` |

Full methodology, caveats, and the retrieval ablation results (bi-encoder beat both
cross-encoder reranking and BM25+dense hybrid on `n=22` labeled queries): [RESULTS.md](RESULTS.md).

---

## Architecture Decisions

Why LangGraph over a simple chain, why FAISS over a managed vector DB, why QLoRA,
why Double ML, why circuit breakers, why Kafka *and* Redis Streams — see
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## Quick Start

```bash
git clone https://github.com/JosephAhn23/HALO
cd HALO
pip install -r requirements.txt
export OPENAI_API_KEY=your_key

# Start services + API
docker compose up -d
uvicorn src.api.main:app --reload

# Run quality evaluation
python -m src.mlops.ragas_tracker

# Physics simulation
python -c "from src.simulation import batch_simulate; print(batch_simulate(batch_size=10, num_steps=100)['quality_score'])"

# Fleet scheduling
python -c "from src.scheduling import FleetReconciler; print('ok')"

# Start MCP server (for Claude Desktop / Cursor)
python src/mcp_server/server.py
```

More recipes (fine-tuning, RLHF, benchmarks, tokenization, interpretability, sandbox,
CUDA kernels, Kafka): see the `Justfile` (`just --list`).

### Testing

```bash
pytest                                     # full suite (130+ tests)
pytest tests/test_physics_integration.py   # physics
pytest tests/test_scheduling_integration.py # scheduling
pytest tests/test_safety.py -v             # safety + red-team
```

### MCP Integration

```json
{
  "mcpServers": {
    "halo": {
      "command": "python",
      "args": ["src/mcp_server/server.py"]
    }
  }
}
```

---

## Project Structure

All application code lives under `src/`. See [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)
for the full per-directory breakdown.

```
src/      All application code (agents, api, inference, mlops, physics, scheduling, ...)
tests/    130+ tests
docs/     Detailed docs (physics, scheduling, architecture, project structure)
assets/   README/demo images
```
