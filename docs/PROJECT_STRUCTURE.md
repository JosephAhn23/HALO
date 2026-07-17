# Project Structure

All application code lives under `src/`. Root only holds `src/`, `tests/`, `docs/`,
`assets/`, and the two Dynamo task-submission repos (`dynamo-*`, each a separate git
repository embedded for reference — see their own READMEs).

```
src/
  agents/               LangGraph pipeline + multi-agent system
    multi_agent/        Supervisor, Research/Critic/Verifier agents, consensus,
                        routing, circuit breakers, HITL, OTel tracing
  api/                  FastAPI gateway, WebSocket streaming, Celery batch queue
  causal_inference/     DR-Learner/T-Learner CATE, DoWhy backdoor adjustment,
                        counterfactual what-if analysis for retrieval pipeline
  cicd/                 RAGAS regression gate (blocks CI on quality drop)
  compile/              torch.compile benchmarking, AoT export, graph break detection
  config/               Hydra structured configs (conf/) + provider factory
  context_engineering/  PromptCompressor (LLMLingua-style), DynamicFewShot (FAISS+MMR),
                        ContextWindowManager (priority eviction), ChainOfThoughtBuilder
  csrc/                 Custom CUDA kernels: fused attention, RMSNorm, top-k sampling
  cuda_ext/             Fused softmax+temperature, RoPE, top-p sampling kernels
  dataset_engineering/  DatasetVersion (DVC lineage), QualityChecker (schema/PII/drift),
                        SyntheticQAGenerator (4 question types), FeatureStore (topo sort)
  eval/                 Gradio evaluation UI
  experimentation/      A/B router, O'Brien-Fleming sequential testing, CUPED,
                        Double ML, power analysis, SRM detection, markdown reports
  experiments/          ABExperiment: Variant handlers, GuardrailConfig, Thompson Sampling,
                        bootstrap CI, MLflow integration
  finetune/             QLoRA, RLHF/PPO, GRPO, Ray fault-tolerant training, quantization
  governance/            Model cards, bias checks (SPD/EOD), SHA-256 audit log,
                        PII redaction, CI enforcement with GitHub Actions integration
  inference/            vLLM, llama.cpp, TRT-LLM, ONNX/Triton, NIM, MoE serving,
                        cuda_dispatch/ (Triton kernel dispatch, physics dispatch)
  infra/                Kubernetes, Terraform (AWS), Azure Bicep/Terraform, SageMaker,
                        Dockerfiles, docker-compose stacks
  ingestion/            Chunking, FAISS indexing, WARC parsing, Spark ML pipelines
  interpretability/     Attention visualization, linear probes, activation patching, CKA
  mcp_server/           MCP protocol server (6 tools, stdio transport)
  microservices/        ServiceRegistry, EventBus (Kafka/Redis), API gateway pattern
  mlops/                RAGAS tracking, MLflow integration, evaluation pipeline,
                        physics trajectory-stability evaluation
  monitoring/            Prometheus + Grafana stack, SLO alert rules, CloudWatch/Azure Monitor
  multimodal/           CLIP retrieval, Stable Diffusion RAG grounding, LLaVA VQA
  observability/        FastAPI Prometheus middleware, pre-built Grafana dashboard
  recsys/               Learn-to-rank (LightGBM), SHAP explainability, NDCG/MAP/MRR
  rl/                   RLHF pipeline (TRL), GRPO reasoning fine-tuning, Gym environments
  safety/               Adversarial tests, semantic safety, ML classifiers, behavioral classifiers
  sandbox/              Docker-based sandboxed code execution with static analysis
  scheduling/           Fleet timer reconciler: IANA timezone expansion, DST handling,
                        deterministic job dispatch, dependency resolution, resource allocation
  simulation/           Physics simulator: batched RK4 integrator, Thomas-BMT spin transport,
                        variational sensitivities, quality gates (energy/spin conservation)
  spark_ml/             Delta Lake medallion pipeline, feature store, MLflow model registry
  storage/              Delta Lake physics results ingestion, feature store queries
  streaming/            Kafka + Kinesis producers/consumers, drift detection, online embeddings
  tokenization/         BPE/WordPiece from scratch, SentencePiece, multilingual analysis

  # Smaller/reference content
  analysis/, benchmarks/, blog_posts/, case_studies/, cost_dashboard/,
  dashboard/, experiments/, interview_prep/, prompt_registry/,
  tiny_companion/, web/

tests/                  130+ tests: unit, integration, adversarial, multi-agent,
                        physics, scheduling
docs/                   This file, plus PHYSICS_SIMULATION.md, FLEET_SCHEDULING.md,
                        ARCHITECTURE.md, TRITON_INTEGRATION.md, operability_evidence.md
assets/                 README/demo images and GIFs
```

## Why `src/`

Everything that used to sit as ~50 individual top-level directories now lives under one
`src/` package. Import paths gained a `src.` prefix (e.g. `from src.governance import ...`),
and `pyproject.toml`'s `[tool.hatch.build.targets.wheel]` packages entry now just points
at `["src"]` instead of maintaining a manually-updated list (which had drifted out of sync
with the actual directories anyway).
