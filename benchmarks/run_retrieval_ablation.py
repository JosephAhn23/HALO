"""
Ablation: bi-encoder-only retrieval vs. bi-encoder + cross-encoder reranking.

Uses the real production classes (BiEncoderEmbedder, CrossEncoderReranker
from agents/reranker.py), not mocks -- the whole point is to check whether
the two-stage design actually earns its latency cost, on real embeddings and
real cross-encoder scores. See benchmarks/retrieval_ablation_data.py for the
labeled corpus (n=38 documents) and query relevance judgments (n=22 queries,
hand-labeled).

No FAISS index exists in this repo yet, so this evaluates over the full
38-document corpus directly (brute-force cosine similarity) rather than
through RetrieverAgent's FAISS path -- the ranking math is identical to what
FAISS's inner-product search on normalized vectors would produce, just
without the ANN index structure, which doesn't matter at this corpus size.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List

import numpy as np

from agents.reranker import BiEncoderEmbedder, CrossEncoderReranker
from benchmarks.retrieval_ablation_data import DOCUMENTS, QUERY_RELEVANCE

K_VALUES = (3, 5)


def precision_at_k(ranked_doc_ids: List[str], relevant: set[str], k: int) -> float:
    top_k = ranked_doc_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for d in top_k if d in relevant) / len(top_k)


def recall_at_k(ranked_doc_ids: List[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = ranked_doc_ids[:k]
    return sum(1 for d in top_k if d in relevant) / len(relevant)


def run_ablation() -> Dict[str, Any]:
    print("=== Retrieval Ablation: bi-encoder-only vs. +cross-encoder ===\n")

    doc_ids = [d["doc_id"] for d in DOCUMENTS]
    doc_texts = [d["text"] for d in DOCUMENTS]

    print(f"1. Embedding {len(DOCUMENTS)} documents with BiEncoderEmbedder...")
    embedder = BiEncoderEmbedder()
    t0 = time.perf_counter()
    doc_embeddings = embedder.embed(doc_texts)
    embed_corpus_s = time.perf_counter() - t0

    print("2. Loading CrossEncoderReranker...")
    cross_encoder = CrossEncoderReranker()

    per_query: List[Dict[str, Any]] = []
    bi_only_latencies_ms: List[float] = []
    two_stage_latencies_ms: List[float] = []
    bi_only_scores = {k: [] for k in K_VALUES}
    bi_only_recalls = {k: [] for k in K_VALUES}
    two_stage_scores = {k: [] for k in K_VALUES}
    two_stage_recalls = {k: [] for k in K_VALUES}

    print(f"3. Running {len(QUERY_RELEVANCE)} queries through both configs...\n")
    for item in QUERY_RELEVANCE:
        query = item["query"]
        relevant = set(item["relevant_doc_ids"])

        # --- bi-encoder only ---
        t0 = time.perf_counter()
        query_emb = embedder.embed([query])
        sims = (query_emb @ doc_embeddings.T)[0]
        bi_only_latencies_ms.append((time.perf_counter() - t0) * 1000)
        bi_order = np.argsort(-sims)
        bi_ranked_ids = [doc_ids[i] for i in bi_order]

        # --- bi-encoder + cross-encoder rerank (rerank the full bi-encoder ranking) ---
        t0 = time.perf_counter()
        rerank_scores = cross_encoder.score_pairs(query, [doc_texts[i] for i in bi_order])
        two_stage_latencies_ms.append((time.perf_counter() - t0) * 1000 + bi_only_latencies_ms[-1])
        reranked_pairs = sorted(zip(bi_ranked_ids, rerank_scores), key=lambda x: x[1], reverse=True)
        two_stage_ranked_ids = [doc_id for doc_id, _ in reranked_pairs]

        row: Dict[str, Any] = {
            "query": query,
            "relevant_doc_ids": sorted(relevant),
            "bi_only_top5": bi_ranked_ids[:5],
            "two_stage_top5": two_stage_ranked_ids[:5],
        }
        for k in K_VALUES:
            p_bi = precision_at_k(bi_ranked_ids, relevant, k)
            r_bi = recall_at_k(bi_ranked_ids, relevant, k)
            p_two = precision_at_k(two_stage_ranked_ids, relevant, k)
            r_two = recall_at_k(two_stage_ranked_ids, relevant, k)
            bi_only_scores[k].append(p_bi)
            bi_only_recalls[k].append(r_bi)
            two_stage_scores[k].append(p_two)
            two_stage_recalls[k].append(r_two)
            row[f"precision_at_{k}"] = {"bi_only": p_bi, "two_stage": p_two}
            row[f"recall_at_{k}"] = {"bi_only": r_bi, "two_stage": r_two}
        per_query.append(row)

    summary: Dict[str, Any] = {"n_queries": len(QUERY_RELEVANCE), "n_documents": len(DOCUMENTS)}
    print("4. Results (macro-averaged over queries)\n")
    print(f"{'metric':<16}{'bi-encoder only':>18}{'+cross-encoder':>18}{'delta':>10}")
    for k in K_VALUES:
        p_bi_mean = float(np.mean(bi_only_scores[k]))
        p_two_mean = float(np.mean(two_stage_scores[k]))
        r_bi_mean = float(np.mean(bi_only_recalls[k]))
        r_two_mean = float(np.mean(two_stage_recalls[k]))
        print(f"P@{k:<14}{p_bi_mean:>18.3f}{p_two_mean:>18.3f}{p_two_mean - p_bi_mean:>+10.3f}")
        print(f"R@{k:<14}{r_bi_mean:>18.3f}{r_two_mean:>18.3f}{r_two_mean - r_bi_mean:>+10.3f}")
        summary[f"precision_at_{k}"] = {"bi_only": p_bi_mean, "two_stage": p_two_mean}
        summary[f"recall_at_{k}"] = {"bi_only": r_bi_mean, "two_stage": r_two_mean}

    bi_p50 = float(np.median(bi_only_latencies_ms))
    two_p50 = float(np.median(two_stage_latencies_ms))
    print(f"\nlatency p50 (ms):  bi-encoder only = {bi_p50:.2f}   +cross-encoder = {two_p50:.2f}")
    summary["latency_ms_p50"] = {"bi_only": bi_p50, "two_stage": two_p50}
    summary["embed_corpus_s"] = embed_corpus_s
    summary["per_query"] = per_query

    with open("benchmarks/retrieval_ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote benchmarks/retrieval_ablation_results.json")

    return summary


if __name__ == "__main__":
    run_ablation()
