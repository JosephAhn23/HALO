"""
Hand-labeled seed dataset for the cost-aware complexity classifier.

Small-n by construction (n=126 labeled + n=18 unanswerable): this is a seed set
for a portfolio-scale demonstration, not a production training corpus. Labels
were assigned by hand using the rubric below, not scraped or LLM-generated, so
held-out accuracy reported against this set is a genuine (if narrow) signal —
see benchmarks/run_cost_router_eval.py for the accuracy this actually achieves.

Label rubric
------------
low    - single-fact lookup, definition, list, or yes/no; answerable in one
         sentence with no reasoning chain.
medium - requires synthesizing 2-3 facts, a short explanation, or a how-to.
high   - requires comparison, multi-step reasoning, tradeoff analysis,
         critique, or open-ended design.

UNANSWERABLE_QUERIES are realistic user queries that fall outside an
LLMOps/RAG/ML-infra knowledge base (the corpus this assistant is meant to
serve). They're used to exercise the retrieval-abstention path: a real corpus
would return low-similarity or empty hits for these, and the router should
decline rather than let the synthesizer hallucinate an answer.
"""

from __future__ import annotations

from typing import TypedDict


class LabeledQuery(TypedDict):
    query: str
    label: str


LOW: list[str] = [
    "What is retrieval-augmented generation?",
    "Define perplexity in the context of language models.",
    "What does LoRA stand for?",
    "List the stages of a typical RAG pipeline.",
    "What is a vector database?",
    "Is FAISS open source?",
    "What does QLoRA stand for?",
    "What is the default embedding dimension of all-MiniLM-L6-v2?",
    "Define hallucination in the context of LLMs.",
    "What is a reward model?",
    "What does RLHF stand for?",
    "Is Redis used for caching in this project?",
    "What is a cross-encoder?",
    "Define context window.",
    "What is chunk overlap?",
    "What does CI/CD stand for?",
    "Is MLflow used for experiment tracking?",
    "What is a bi-encoder?",
    "Define top-k in retrieval.",
    "What does RAGAS stand for?",
    "Is Celery used for background jobs in this project?",
    "What is faithfulness in RAGAS?",
    "Define answer relevancy.",
    "What port does the API run on by default?",
    "Is bitsandbytes used for quantization?",
    "What does PEFT stand for?",
    "What is a healthcheck in Docker Compose?",
    "Define context precision.",
    "What model does the synthesizer use by default?",
    "Is vLLM supported as a backend?",
    "What does TRL stand for?",
    "List three cloud providers this project deploys to.",
    "What is the default chunk size?",
    "Is Kubernetes used for deployment?",
    "What does API_KEY control in api/main.py?",
    "Define tokens_used in the query response.",
    "What is a shard in the distributed FAISS setup?",
    "Is Prometheus used for metrics?",
    "What does CORS_ORIGINS default to?",
    "What is the health endpoint's URL path?",
]

MEDIUM: list[str] = [
    "How does the reranker improve retrieval quality over raw FAISS search?",
    "Explain how context compression reduces prompt tokens without hurting faithfulness.",
    "How does the /batch endpoint queue work end to end?",
    "Walk me through what happens when a query hits /query.",
    "How does the constitutional gate decide to regenerate an answer?",
    "Explain how the behavioral classifier blocks a request before retrieval runs.",
    "How is the FAISS index built during ingestion?",
    "Summarize how attribution links answer sentences back to source chunks.",
    "How does the truth committee combine two providers' answers?",
    "Explain how Celery workers pick up jobs from the priority queue vs default queue.",
    "How does session_id get used for ResearchLog injection?",
    "Summarize how the RAGAS gate decides to block a CI merge.",
    "How does the API authenticate requests with X-API-Key?",
    "Explain how the distributed FAISS aggregator combines shard results.",
    "How does grounding confidence get computed from an answer and its chunks?",
    "Summarize the steps in the ingestion pipeline from raw text to indexed vectors.",
    "How does the policy enforcement agent decide which sentences to drop?",
    "Explain how the flower dashboard monitors Celery workers.",
    "How does the reranker's cross-encoder score get combined with the retrieval score?",
    "Summarize how the health endpoint checks Redis connectivity.",
    "How would I add a new field to the /query response?",
    "Explain how the adversarial consensus step differs from the truth committee gate.",
    "How does the API decide whether to skip the RAG pipeline entirely?",
    "Summarize what the dead-letter queue worker is for.",
    "How does chunk_size interact with chunk_overlap during ingestion?",
    "Explain how MLflow nested runs are used inside the pipeline.",
    "How does the /ingest endpoint assign a doc_id?",
    "Summarize the difference between retrieval_score and rerank_score.",
    "How does the aggregator route a search request to the right shard?",
    "Explain how speculative sentence detection works in attribution.",
    "How would a client poll for batch job completion?",
    "Summarize how CORS is configured for local development vs production.",
    "How does the pipeline behave when the constitutional classifier fails to load?",
    "Explain how vLLM sampling parameters affect generation length.",
    "How is the embedding model shared between ingestion and retrieval?",
    "Summarize how a 502 response differs from a 400 in the /query endpoint.",
    "How does the shard aggregator handle a shard that's down?",
    "Explain what happens if OPENAI_API_KEY is unset when the synthesizer starts.",
    "How would I run the RAGAS baseline eval locally?",
    "Summarize what fields the /health endpoint returns and what they mean.",
]

HIGH: list[str] = [
    "Compare QLoRA against full fine-tuning and LoRA — what are the memory and quality tradeoffs?",
    "What are the tradeoffs between the consensus quality gate and the adversarial consensus step, and when would you use one over the other?",
    "Design a strategy for detecting retrieval quality regressions before they reach production.",
    "Critique the current abstention behavior when FAISS returns no relevant chunks — what's the failure mode and how would you fix it?",
    "How would you architect a cost-aware router that balances latency, quality, and $/query across model tiers?",
    "Analyze the tradeoffs between self-hosted vLLM inference and OpenAI's API for this pipeline's cost profile.",
    "What are the implications of running the constitutional gate synchronously versus as an async post-hoc check?",
    "Debate whether the truth committee's HITL halt threshold should be static or adaptive to query difficulty.",
    "Explain why context compression might hurt faithfulness on some query types even when average metrics look flat.",
    "Design a blue/green rollout strategy for a newly fine-tuned adapter that minimizes risk of a quality regression.",
    "Compare the shard-based distributed FAISS design against a single large index — when does sharding actually pay off?",
    "What are the failure modes of relying on an LLM-as-judge (RAGAS) for quality gating, and how would you mitigate them?",
    "Analyze how query complexity classification could introduce systematic bias against certain phrasing styles.",
    "Critique the current retry/backoff strategy in the OpenAI client — what happens under sustained rate limiting?",
    "How would you evaluate whether the reranker is actually improving end-to-end answer quality versus just reordering noise?",
    "Design an experiment to measure whether adversarial consensus reduces hallucination rate without inflating latency unacceptably.",
    "What are the tradeoffs between reward-model-driven RLHF and constitutional-AI-style rule-based grading for this pipeline?",
    "Explain the reasoning behind choosing weighted-confidence consensus over majority-vote for the multi-agent supervisor.",
    "Compare the operational cost of running four FAISS shards versus one aggregator-less monolithic index at this data scale.",
    "How would you redesign the /batch endpoint so that partial failures in a 100-query batch don't require full re-submission?",
    "Critique whether the current health check meaningfully signals production readiness versus just process liveness.",
    "Analyze what would break if two API replicas behind a load balancer both try to write to the same local FAISS index concurrently.",
    "Design a test suite that would have caught the missing root Dockerfile and broken dependency pins before they shipped.",
    "What's the argument for keeping fine-tuning and RLHF as separate pipelines instead of a unified training loop?",
    "How would you decide, with real evidence, whether the multi-cloud deployment claim is actually load-bearing or just two disconnected templates?",
    "Compare synchronous consensus gating against an async review queue for HITL-flagged answers, in terms of user-facing latency.",
    "Explain the tradeoffs between hand-labeling a small classifier training set versus bootstrapping labels from an LLM.",
    "Design a rollback plan for a cost router that starts misrouting complex queries to the cheap tier in production.",
]

LABELED_QUERIES: list[LabeledQuery] = (
    [{"query": q, "label": "low"} for q in LOW]
    + [{"query": q, "label": "medium"} for q in MEDIUM]
    + [{"query": q, "label": "high"} for q in HIGH]
)

UNANSWERABLE_QUERIES: list[str] = [
    "What's a good substitute for buttermilk in a pancake recipe?",
    "Who won the World Cup in 2018?",
    "What's the best hiking trail near Denver?",
    "How do I remove a red wine stain from a carpet?",
    "What's the capital of Mongolia?",
    "Can you recommend a good sci-fi novel from the 1960s?",
    "What's the difference between a violin and a viola?",
    "How long should I marinate chicken before grilling it?",
    "What's a good name for a golden retriever puppy?",
    "Is it going to rain in Seattle this weekend?",
    "What's the offside rule in soccer?",
    "How do I train for a half marathon in 12 weeks?",
    "What's the best way to propagate a pothos plant?",
    "Who painted the Sistine Chapel ceiling?",
    "What's a fair price for a used 2018 Honda Civic?",
    "How do I get red wine out of a white shirt?",
    "What's the tallest mountain in South America?",
    "Can you suggest a birthday gift for a 10-year-old?",
]
