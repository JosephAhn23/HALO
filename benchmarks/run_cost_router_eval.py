"""
Before/after eval for the cost-aware router (agents/multi_agent/cost_router.py).

No FAISS index has been ingested into this repo yet, so this can't replay
queries through the real RetrieverAgent. Instead of faking a constant
"good"/"bad" retrieval score, this computes genuine cosine similarity between
each eval query and a small in-domain reference corpus (pulled verbatim from
this repo's own README "What It Does" table) using the same EmbeddingModel
the real retriever uses. That's a real measurement of whether the abstention
threshold actually separates in-domain from out-of-domain queries — it's just
measured against a hand-picked reference set instead of a real FAISS index.

Two things this script reports honestly rather than assumes:
  1. Classifier held-out accuracy (from ComplexityClassifier.evaluate()).
  2. Whether DEFAULT_ABSTAIN_THRESHOLD actually separates the labeled
     (in-domain) test queries from UNANSWERABLE_QUERIES (out-of-domain) on
     these real similarity scores — if it doesn't cleanly separate, that's
     reported as a finding, not hidden.

Cost numbers use the *real* mean prompt/completion token counts recorded in
RESULTS.md's "No Compression" row (n=1,000 queries) rather than an invented
number, multiplied by the pricing snapshot in cost_router.py. This is a cost
*model*, not a live-traffic measurement — see the eval-mode tradeoff this was
scoped against.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np

from agents.multi_agent.cost_classifier import train_test_split, train_default_classifier
from agents.multi_agent.cost_router import (
    DEFAULT_ABSTAIN_THRESHOLD,
    CostAwareRouter,
    estimate_cost_usd,
)
from agents.multi_agent.cost_router_data import LABELED_QUERIES, UNANSWERABLE_QUERIES

# RESULTS.md "Context Compression Impact" -> "No Compression" row, n=1,000.
# Real recorded tokens/query for this pipeline's baseline generation path.
BASELINE_PROMPT_TOKENS = 1420
BASELINE_COMPLETION_TOKENS = 188

NAIVE_MODEL = "gpt-4o"  # "always use the strongest model" -- today's implicit behavior

# In-domain reference corpus, verbatim from README.md's "What It Does" table
# (the actual domain this assistant is meant to answer questions about).
REFERENCE_CORPUS = [
    "Chunking, embedding, FAISS indexing. HuggingFace Datasets and MinHash dedup. "
    "CommonCrawl WARC parsing. Spark distributed ingestion and Spark ML feature engineering.",
    "Two-stage search: FAISS bi-encoder scan across distributed shards, then cross-encoder "
    "reranking.",
    "LangGraph multi-agent pipeline: stateful graph with retriever, reranker, synthesizer, "
    "conditional routing, tool protocols. Token streaming over WebSocket. vLLM backend for "
    "self-hosted inference.",
    "QLoRA 4-bit NF4 fine-tuning via PEFT, BitsAndBytes, and Accelerate. RLHF with PPO: "
    "Bradley-Terry reward model, KL-penalised policy optimisation.",
    "RAGAS evaluation: faithfulness, relevancy, precision, recall, with MLflow tracking. "
    "Wired into CI as a quality gate that blocks merges on regression.",
    "TensorRT-LLM engine builder, ONNX/Triton pipeline, NVIDIA NIM adapter, MoE expert "
    "parallelism, custom CUDA kernels for fused attention and RMSNorm.",
    "Rule-based prompt injection detection, embedding anomaly detection, LLM-as-judge "
    "red-team suite, ML jailbreak classifier, behavioral classifiers for toxicity and intent.",
    "Model cards, bias evaluation with statistical parity and equal opportunity, PII "
    "redaction, cryptographic audit log, CI enforcement for governance checks.",
    "Hash-based A/B router, sequential testing, CUPED variance reduction, Double ML for "
    "unbiased average treatment effect estimation, sample size calculator.",
    "Uplift modeling with T-Learner, propensity score matching, synthetic experiment "
    "simulator with confounders.",
    "Delta Lake medallion pipeline with bronze, silver, and gold layers, feature store with "
    "point-in-time correct joins, MLflow model registry with gated promotion.",
    "Hybrid retrieval with LightGBM learn-to-rank, SHAP feature importance, MMR diversity "
    "reranking, offline NDCG and MRR evaluation.",
    "Stateful stream processor for Kafka and Kinesis events, drift detection, distribution "
    "monitoring, online embedding refresh.",
    "Token budget allocation, query rewriting, retrieval compression, memory decay policy, "
    "model routing by query complexity.",
    "Research, critic, and verifier agent loop with circuit breakers, exponential backoff "
    "retry, graceful degradation, human-in-the-loop checkpoints.",
    "Docker Compose, Kubernetes manifests, Terraform for AWS, Azure Container Apps with "
    "Bicep infrastructure as code.",
]


def _cosine_top_scores(queries: List[str], reference_embeddings: np.ndarray, embedder) -> List[float]:
    query_embeddings = np.asarray(embedder.embed(queries))
    # Both sides are already L2-normalized by EmbeddingModel, so dot product
    # is cosine similarity -- same convention as retriever.py's FAISS index.
    sims = query_embeddings @ reference_embeddings.T
    return sims.max(axis=1).tolist()


def run_eval(seed: int = 13) -> Dict[str, Any]:
    print("=== Cost-Aware Router Eval ===\n")

    print("1. Training complexity classifier on hand-labeled seed set...")
    classifier, class_report = train_default_classifier(seed=seed)
    print(f"   held-out accuracy: {class_report.accuracy:.3f} (n={class_report.n_test})")
    print(f"   per-label accuracy: {class_report.per_label_accuracy}\n")

    _, held_out = train_test_split(LABELED_QUERIES, seed=seed)

    print("2. Computing real cosine similarity vs. in-domain reference corpus...")
    embedder = classifier.get_embedder()
    reference_embeddings = np.asarray(embedder.embed(REFERENCE_CORPUS))

    answerable_queries = [item["query"] for item in held_out]
    answerable_scores = _cosine_top_scores(answerable_queries, reference_embeddings, embedder)
    unanswerable_scores = _cosine_top_scores(UNANSWERABLE_QUERIES, reference_embeddings, embedder)

    print(f"   answerable held-out top-score:   min={min(answerable_scores):.3f} "
          f"max={max(answerable_scores):.3f} mean={np.mean(answerable_scores):.3f}")
    print(f"   unanswerable top-score:          min={min(unanswerable_scores):.3f} "
          f"max={max(unanswerable_scores):.3f} mean={np.mean(unanswerable_scores):.3f}")
    print(f"   abstain threshold: {DEFAULT_ABSTAIN_THRESHOLD}\n")

    router = CostAwareRouter(classifier=classifier)

    # --- Route every query in the eval set, tally cost + abstention ---
    per_query: List[Dict[str, Any]] = []
    naive_cost_total = 0.0
    router_cost_total = 0.0
    false_abstains = 0  # answerable query incorrectly abstained
    caught_unanswerable = 0  # unanswerable query correctly abstained

    for item, score in zip(held_out, answerable_scores):
        chunks = [{"text": "ref", "retrieval_score": score}]
        decision = router.route(item["query"], chunks)
        naive_cost = estimate_cost_usd(NAIVE_MODEL, BASELINE_PROMPT_TOKENS, BASELINE_COMPLETION_TOKENS)
        router_cost = 0.0 if decision.abstain else estimate_cost_usd(
            decision.model, BASELINE_PROMPT_TOKENS, BASELINE_COMPLETION_TOKENS
        )
        naive_cost_total += naive_cost
        router_cost_total += router_cost
        if decision.abstain:
            false_abstains += 1
        per_query.append({
            "query": item["query"],
            "gold_label": item["label"],
            "kind": "answerable",
            "similarity_score": round(score, 4),
            "decision": decision.to_dict(),
            "naive_cost_usd": naive_cost,
            "router_cost_usd": router_cost,
        })

    for query, score in zip(UNANSWERABLE_QUERIES, unanswerable_scores):
        chunks = [{"text": "ref", "retrieval_score": score}]
        decision = router.route(query, chunks)
        naive_cost = estimate_cost_usd(NAIVE_MODEL, BASELINE_PROMPT_TOKENS, BASELINE_COMPLETION_TOKENS)
        router_cost = 0.0 if decision.abstain else estimate_cost_usd(
            decision.model, BASELINE_PROMPT_TOKENS, BASELINE_COMPLETION_TOKENS
        )
        naive_cost_total += naive_cost
        router_cost_total += router_cost
        if decision.abstain:
            caught_unanswerable += 1
        per_query.append({
            "query": query,
            "gold_label": None,
            "kind": "unanswerable",
            "similarity_score": round(score, 4),
            "decision": decision.to_dict(),
            "naive_cost_usd": naive_cost,
            "router_cost_usd": router_cost,
        })

    n_eval = len(held_out) + len(UNANSWERABLE_QUERIES)
    savings_pct = (1 - router_cost_total / naive_cost_total) * 100 if naive_cost_total else 0.0
    abstain_recall = caught_unanswerable / len(UNANSWERABLE_QUERIES) if UNANSWERABLE_QUERIES else 0.0
    false_abstain_rate = false_abstains / len(held_out) if held_out else 0.0

    print("3. Results\n")
    print(f"   n eval queries:            {n_eval} ({len(held_out)} answerable, "
          f"{len(UNANSWERABLE_QUERIES)} unanswerable)")
    print(f"   naive baseline cost (always {NAIVE_MODEL}, no abstention): ${naive_cost_total:.4f}")
    print(f"   router-enabled cost:       ${router_cost_total:.4f}")
    print(f"   cost reduction:            {savings_pct:.1f}%")
    print(f"   abstention recall (caught unanswerable / total unanswerable): "
          f"{caught_unanswerable}/{len(UNANSWERABLE_QUERIES)} ({abstain_recall:.1%})")
    print(f"   false-abstain rate (answerable incorrectly declined): "
          f"{false_abstains}/{len(held_out)} ({false_abstain_rate:.1%})")

    results = {
        "classifier_held_out_accuracy": class_report.to_dict(),
        "abstain_threshold": DEFAULT_ABSTAIN_THRESHOLD,
        "similarity_scores": {
            "answerable": {
                "min": min(answerable_scores), "max": max(answerable_scores),
                "mean": float(np.mean(answerable_scores)),
            },
            "unanswerable": {
                "min": min(unanswerable_scores), "max": max(unanswerable_scores),
                "mean": float(np.mean(unanswerable_scores)),
            },
        },
        "n_eval": n_eval,
        "n_answerable": len(held_out),
        "n_unanswerable": len(UNANSWERABLE_QUERIES),
        "naive_cost_usd_total": naive_cost_total,
        "router_cost_usd_total": router_cost_total,
        "cost_reduction_pct": savings_pct,
        "abstention_recall": abstain_recall,
        "false_abstain_rate": false_abstain_rate,
        "per_query": per_query,
    }

    with open("benchmarks/cost_router_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nWrote benchmarks/cost_router_results.json")

    return results


if __name__ == "__main__":
    run_eval()
