"""
Labeled corpus + query relevance judgments for the retrieval ablation
(benchmarks/run_retrieval_ablation.py).

No FAISS index has been ingested into this repo, so there's no real corpus
to evaluate retrieval quality against. This uses real text instead of
synthetic filler: documents are pulled verbatim from this repo's own
README.md ("What It Does", "Architecture Decisions", "Business Impact"
sections), and several documents deliberately cover overlapping topics from
different angles (e.g. "two-stage retrieval" appears in a feature-inventory
doc, an architecture-rationale doc, and a business-impact doc) -- that
overlap is what makes the bi-encoder-only vs. +reranker comparison
informative rather than trivial.

Relevance judgments (QUERY_RELEVANCE) were hand-labeled by reading each
document, not inferred from keyword overlap -- n=22 queries, n=38 documents.
Small-n by construction, same house style as RESULTS.md's other small-sample
evals.
"""

from __future__ import annotations

from typing import TypedDict


class Document(TypedDict):
    doc_id: str
    text: str


DOCUMENTS: list[Document] = [
    # --- "What It Does" table (README.md) ---
    {
        "doc_id": "wid_01",
        "text": "Ingest: chunking, embedding, FAISS indexing. HuggingFace Datasets and MinHash dedup. CommonCrawl WARC parsing from S3. Spark distributed ingestion and Spark ML feature engineering with TF-IDF, Word2Vec, and K-Means.",
    },
    {
        "doc_id": "wid_02",
        "text": "Search: two-stage retrieval, FAISS bi-encoder scan across four distributed shards, then cross-encoder reranking. Sub-50ms locally.",
    },
    {
        "doc_id": "wid_03",
        "text": "Generate: LangGraph multi-agent pipeline, a stateful graph with retriever, reranker, and synthesizer nodes, conditional routing, tool protocols. Token streaming over WebSocket. vLLM backend for self-hosted inference.",
    },
    {
        "doc_id": "wid_04",
        "text": "Fine-tune: QLoRA 4-bit NF4 quantization via PEFT, BitsAndBytes, and Accelerate. RLHF with PPO using a Bradley-Terry reward model and KL-penalised policy optimisation. GRPO reasoning fine-tuning. Ray Train for fault-tolerant distributed training.",
    },
    {
        "doc_id": "wid_05",
        "text": "Evaluate: RAGAS metrics (faithfulness, relevancy, precision, recall) with MLflow tracking. Wired into GitHub Actions CI as a quality gate that blocks merges on regression greater than 5%.",
    },
    {
        "doc_id": "wid_06",
        "text": "Serve: TensorRT-LLM engine builder supporting FP16, INT8, and FP8, an ONNX/Triton pipeline, an NVIDIA NIM adapter, MoE expert parallelism, and custom CUDA kernels for fused attention and RMSNorm.",
    },
    {
        "doc_id": "wid_07",
        "text": "Secure: rule-based prompt injection detection, embedding anomaly detection, an LLM-as-judge red-team suite covering nine attack categories, an ML jailbreak classifier, and behavioral classifiers for toxicity, intent, and topic.",
    },
    {
        "doc_id": "wid_08",
        "text": "Govern: model cards following the Mitchell et al. standard, bias evaluation with statistical parity and equal opportunity, PII redaction across ten pattern types, a SHA-256 cryptographic audit log, and CI enforcement with an exit code for GitHub Actions.",
    },
    {
        "doc_id": "wid_09",
        "text": "Experiment: a hash-based A/B router, O'Brien-Fleming sequential testing with no alpha inflation, CUPED variance reduction, Double ML for unbiased average treatment effect, a sample size calculator, SRM detection, and automated markdown reports.",
    },
    {
        "doc_id": "wid_10",
        "text": "Causal: uplift modeling with a T-Learner, DoWhy-style propensity score matching, Double ML cross-fitting, CUPED variance reduction, and a synthetic experiment simulator with confounders.",
    },
    {
        "doc_id": "wid_11",
        "text": "Data: a Delta Lake medallion pipeline with bronze, silver, and gold layers, a feature store with point-in-time correct joins, and an MLflow model registry with gated promotion and rollback.",
    },
    {
        "doc_id": "wid_12",
        "text": "Recommend: hybrid retrieval combined with LightGBM learn-to-rank, SHAP feature importance, MMR diversity reranking, and offline NDCG, MAP, and MRR evaluation.",
    },
    {
        "doc_id": "wid_13",
        "text": "Stream: a stateful stream processor for Kafka and Kinesis events, Page-Hinkley plus ADWIN drift detection, PSI distribution monitoring, and online embedding refresh.",
    },
    {
        "doc_id": "wid_14",
        "text": "Context: token budget allocation, query rewriting via HyDE, step-back prompting, and sub-query decomposition, retrieval compression, memory decay policy, and model routing by query complexity.",
    },
    {
        "doc_id": "wid_15",
        "text": "Multi-agent: a research, critic, and verifier agent loop with circuit breakers, exponential backoff retry, graceful degradation, human-in-the-loop checkpoints, and OpenTelemetry tracing.",
    },
    {
        "doc_id": "wid_16",
        "text": "Deploy: Docker Compose, Kubernetes manifests, Terraform for AWS covering VPC, EC2, ALB, and RDS, and Azure Container Apps with Bicep infrastructure as code.",
    },
    # --- "Architecture Decisions" section (README.md) ---
    {
        "doc_id": "arch_01",
        "text": "Why LangGraph instead of a simple chain? A chain runs top-to-bottom and stops. LangGraph is a directed graph where each node can inspect the full state, decide which node to call next, and recover from failures without restarting -- useful when retrieval returns nothing useful or a safety check fires mid-pipeline.",
    },
    {
        "doc_id": "arch_02",
        "text": "Why two-stage retrieval combining a bi-encoder and a cross-encoder? Bi-encoders like FAISS are fast but approximate, comparing embeddings independently and missing subtle relevance signals. Cross-encoders read the query and document together, catching nuance the bi-encoder misses. Running cross-encoding only on the top-50 FAISS results keeps end-to-end latency under 50ms while improving precision.",
    },
    {
        "doc_id": "arch_03",
        "text": "Why FAISS over a managed vector database? FAISS runs in-process: no network hop, no managed service cost, no vendor lock-in. The distributed shard design gives horizontal scale without changing the query interface. The trade-off is no real-time updates as cleanly as Pinecone or Weaviate; for periodic re-indexing that's acceptable.",
    },
    {
        "doc_id": "arch_04",
        "text": "Why QLoRA instead of full fine-tuning? Full fine-tuning an 8B model requires roughly 80GB of GPU memory. QLoRA compresses the frozen base model to 4-bit and trains only small LoRA adapter matrices injected into attention layers, under 1% of total parameters, dropping the hardware requirement from four A100s to a single consumer GPU.",
    },
    {
        "doc_id": "arch_05",
        "text": "Why Double ML for causal inference instead of a simple A/B test? A/B tests measure correlation. When users self-select into a feature, a naive comparison is confounded. Double ML residualises both the outcome and the treatment on observed covariates using k-fold cross-fitting, then regresses the residuals, giving an unbiased average treatment effect even with complex nonlinear confounders.",
    },
    {
        "doc_id": "arch_06",
        "text": "Why O'Brien-Fleming sequential testing instead of fixed-horizon? Checking p-values repeatedly inflates the false positive rate. O'Brien-Fleming alpha spending allocates the Type-I error budget across planned looks: conservative early with a high boundary, liberal late with a low boundary, keeping the overall false positive rate at alpha regardless of how many times you check.",
    },
    {
        "doc_id": "arch_07",
        "text": "Why circuit breakers in the multi-agent system? A slow or failing verifier agent would block the entire pipeline under naive retry. The circuit breaker opens after N consecutive failures, immediately returning a degraded response with a confidence penalty instead of waiting for timeouts. After a cooldown period, one probe call tests recovery.",
    },
    {
        "doc_id": "arch_08",
        "text": "Why both Kafka and Redis Streams for the event bus? Kafka is the right choice for production: durable, ordered, replayable, with consumer groups. Redis Streams is right for local development: zero infrastructure, the same API shape, instant startup. The EventBus class auto-detects which backend to use from environment variables.",
    },
    # --- "Business Impact" table (README.md) ---
    {
        "doc_id": "biz_01",
        "text": "The RAGAS CI gate solves the problem that regressions reach production silently and erode user trust, by catching quality drops before merge instead of after user complaints.",
    },
    {
        "doc_id": "biz_02",
        "text": "Two-stage retrieval solves the problem that embedding similarity alone misses 15-25% of relevant results; cross-encoder reranking recovers that precision without the cost of a full scan.",
    },
    {
        "doc_id": "biz_03",
        "text": "Context engineering solves the problem that long-context LLM calls cost 5-10x more than necessary, delivering a 35% token reduction via extractive compression with the same RAGAS scores.",
    },
    {
        "doc_id": "biz_04",
        "text": "O'Brien-Fleming sequential testing solves the problem that fixed-horizon tests waste compute on obvious winners and losers; early stopping cuts experiment duration by 30-50% without inflating the false positive rate.",
    },
    {
        "doc_id": "biz_05",
        "text": "CUPED variance reduction solves the problem that standard A/B tests need large samples for noisy metrics; pre-experiment covariate adjustment reduces the required sample size by 30-50%.",
    },
    {
        "doc_id": "biz_06",
        "text": "Double ML causal inference solves the problem that correlation metrics can't prove a new feature caused an improvement; it isolates the true treatment effect from selection bias with unbiased ATE estimation.",
    },
    {
        "doc_id": "biz_07",
        "text": "Uplift modeling solves the problem that deploying a feature to all users wastes resources if it only helps a subset, by targeting high-uplift users and skipping the rest.",
    },
    {
        "doc_id": "biz_08",
        "text": "SRM detection solves the problem that traffic split bugs silently invalidate experiment results, using a chi-squared test to catch assignment mechanism failures before decisions are made.",
    },
    {
        "doc_id": "biz_09",
        "text": "Streaming drift detection solves the problem that model quality degrades silently as data distribution shifts; Page-Hinkley and ADWIN detect drift within 50-100 events instead of days later in a batch job.",
    },
    {
        "doc_id": "biz_10",
        "text": "The feature store and Delta Lake pipeline solve the problem that training/serving skew causes silent accuracy loss; point-in-time correct feature joins eliminate leakage and Delta time-travel enables reproducibility.",
    },
    {
        "doc_id": "biz_11",
        "text": "PII redaction and the audit log solve the problem that prompt logging can leak user data and violate GDPR/CCPA, via automatic scrubbing before any log write and a SHA-256 hash chain for tamper-proof compliance records.",
    },
    {
        "doc_id": "biz_12",
        "text": "Governance CI enforcement solves the problem that fairness regressions ship silently; the build fails if the statistical parity difference or equal opportunity difference exceeds a threshold.",
    },
    {
        "doc_id": "biz_13",
        "text": "The multi-agent critic/verifier loop solves the problem that single-pass generation hallucinates on complex questions, reducing unsupported claims before returning an answer to the user.",
    },
    {
        "doc_id": "biz_14",
        "text": "QLoRA plus RLHF solve the problem that cloud LLM APIs cost around $0.005 per 1,000 tokens at scale; a self-hosted fine-tuned model runs at roughly $0.0001 per 1,000 tokens at 1,500 tokens/second on an A100.",
    },
]


class QueryRelevance(TypedDict):
    query: str
    relevant_doc_ids: list[str]


QUERY_RELEVANCE: list[QueryRelevance] = [
    {
        "query": "Why does this project use two-stage retrieval instead of raw embedding search?",
        "relevant_doc_ids": ["arch_02", "biz_02"],
    },
    {
        "query": "What's the hardware and cost tradeoff of QLoRA versus full fine-tuning?",
        "relevant_doc_ids": ["arch_04", "wid_04"],
    },
    {
        "query": "How does the system prevent a single slow agent from blocking the whole pipeline?",
        "relevant_doc_ids": ["arch_07"],
    },
    {
        "query": "Why was FAISS chosen over a managed vector database like Pinecone?",
        "relevant_doc_ids": ["arch_03"],
    },
    {
        "query": "How does the CI pipeline prevent quality regressions from reaching production?",
        "relevant_doc_ids": ["biz_01", "wid_05"],
    },
    {
        "query": "What technique reduces prompt token costs without hurting answer quality?",
        "relevant_doc_ids": ["biz_03", "wid_14"],
    },
    {
        "query": "How does the project detect data drift in production?",
        "relevant_doc_ids": ["biz_09", "wid_13"],
    },
    {
        "query": "What method estimates the true causal effect of a new feature, correcting for user self-selection?",
        "relevant_doc_ids": ["arch_05", "biz_06"],
    },
    {
        "query": "How does sequential testing avoid inflating false positive rates from repeated peeking at results?",
        "relevant_doc_ids": ["arch_06", "biz_04"],
    },
    {
        "query": "What variance reduction technique shrinks the sample size needed for an A/B test?",
        "relevant_doc_ids": ["biz_05"],
    },
    {
        "query": "How does the project decide which individual users should get a new feature based on treatment effect?",
        "relevant_doc_ids": ["biz_07", "wid_10"],
    },
    {
        "query": "How are traffic-split bugs in an experiment detected before a bad decision is made?",
        "relevant_doc_ids": ["biz_08"],
    },
    {
        "query": "How does the multi-agent system reduce hallucinated claims in complex answers?",
        "relevant_doc_ids": ["biz_13", "wid_15"],
    },
    {
        "query": "What prevents training/serving skew from silently degrading model accuracy?",
        "relevant_doc_ids": ["biz_10", "wid_11"],
    },
    {
        "query": "How is user data protected from leaking through prompt logs?",
        "relevant_doc_ids": ["biz_11", "wid_08"],
    },
    {
        "query": "What enforces fairness thresholds so a biased model can't merge?",
        "relevant_doc_ids": ["biz_12", "wid_08"],
    },
    {
        "query": "Why is LangGraph used instead of a simple linear chain?",
        "relevant_doc_ids": ["arch_01"],
    },
    {
        "query": "Why does the project run both Kafka and Redis Streams for its event bus?",
        "relevant_doc_ids": ["arch_08"],
    },
    {
        "query": "What is the estimated per-token cost difference between a cloud LLM API and a self-hosted fine-tuned model?",
        "relevant_doc_ids": ["biz_14"],
    },
    {
        "query": "How is raw web crawl data cleaned and deduplicated before ingestion?",
        "relevant_doc_ids": ["wid_01"],
    },
    {
        "query": "What serving stack is used for low-latency GPU inference?",
        "relevant_doc_ids": ["wid_06"],
    },
    {
        "query": "How does the recommendation system rank candidates after retrieval?",
        "relevant_doc_ids": ["wid_12"],
    },
]
