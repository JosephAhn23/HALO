"""
Ablation: bi-encoder-only vs. bi-encoder + cross-encoder reranking vs.
BM25 + bi-encoder hybrid search fused with Reciprocal Rank Fusion (RRF).

Uses real production/library code, not mocks: BiEncoderEmbedder and
CrossEncoderReranker from agents/reranker.py, and rank_bm25.BM25Okapi (already
a core dependency per pyproject.toml -- no new dependency added for this).
The whole point is to check whether each retrieval strategy actually earns
its cost, on real embeddings and real scores, against a hand-labeled set. See
benchmarks/retrieval_ablation_data.py for the labeled corpus (n=38 documents)
and query relevance judgments (n=22 queries, hand-labeled).

No FAISS index exists in this repo yet, so this evaluates over the full
38-document corpus directly (brute-force cosine similarity for the dense
side) rather than through RetrieverAgent's FAISS path -- the ranking math is
identical to what FAISS's inner-product search on normalized vectors would
produce, just without the ANN index structure, which doesn't matter at this
corpus size.

RRF: rrf_score(doc) = sum over rankers r of 1/(k + rank_r(doc)), rank 1-indexed,
k=60 (the constant used in the original Cormack et al. 2009 RRF paper and the
de facto default in hybrid-search implementations).
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List

import numpy as np
from rank_bm25 import BM25Okapi

from agents.reranker import BiEncoderEmbedder, CrossEncoderReranker
from benchmarks.retrieval_ablation_data import DOCUMENTS, QUERY_RELEVANCE

K_VALUES = (3, 5)
RRF_K = 60
METHODS = ("bi_only", "two_stage", "hybrid_rrf")
METHOD_LABELS = {
    "bi_only": "bi-encoder only",
    "two_stage": "+cross-encoder",
    "hybrid_rrf": "BM25+dense RRF",
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


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


def reciprocal_rank_fusion(rankings: List[List[str]], k: int = RRF_K) -> List[str]:
    """Fuse N independent rankings of the same doc_ids into one, via RRF."""
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


def run_ablation() -> Dict[str, Any]:
    print("=== Retrieval Ablation: bi-encoder vs. +cross-encoder vs. BM25+dense RRF ===\n")

    doc_ids = [d["doc_id"] for d in DOCUMENTS]
    doc_texts = [d["text"] for d in DOCUMENTS]

    print(f"1. Embedding {len(DOCUMENTS)} documents with BiEncoderEmbedder...")
    embedder = BiEncoderEmbedder()
    t0 = time.perf_counter()
    doc_embeddings = embedder.embed(doc_texts)
    embed_corpus_s = time.perf_counter() - t0

    print("2. Loading CrossEncoderReranker...")
    cross_encoder = CrossEncoderReranker()

    print("3. Building BM25 index over the corpus...")
    tokenized_corpus = [_tokenize(t) for t in doc_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    per_query: List[Dict[str, Any]] = []
    latencies_ms: Dict[str, List[float]] = {m: [] for m in METHODS}
    scores: Dict[str, Dict[int, List[float]]] = {m: {k: [] for k in K_VALUES} for m in METHODS}
    recalls: Dict[str, Dict[int, List[float]]] = {m: {k: [] for k in K_VALUES} for m in METHODS}

    print(f"4. Running {len(QUERY_RELEVANCE)} queries through all three configs...\n")
    for item in QUERY_RELEVANCE:
        query = item["query"]
        relevant = set(item["relevant_doc_ids"])

        # --- bi-encoder only ---
        t0 = time.perf_counter()
        query_emb = embedder.embed([query])
        sims = (query_emb @ doc_embeddings.T)[0]
        bi_latency = (time.perf_counter() - t0) * 1000
        bi_order = np.argsort(-sims)
        bi_ranked_ids = [doc_ids[i] for i in bi_order]
        latencies_ms["bi_only"].append(bi_latency)

        # --- bi-encoder + cross-encoder rerank (rerank the full bi-encoder ranking) ---
        t0 = time.perf_counter()
        rerank_scores = cross_encoder.score_pairs(query, [doc_texts[i] for i in bi_order])
        two_stage_latency = (time.perf_counter() - t0) * 1000 + bi_latency
        reranked_pairs = sorted(zip(bi_ranked_ids, rerank_scores), key=lambda x: x[1], reverse=True)
        two_stage_ranked_ids = [doc_id for doc_id, _ in reranked_pairs]
        latencies_ms["two_stage"].append(two_stage_latency)

        # --- BM25 + dense hybrid, fused via RRF ---
        t0 = time.perf_counter()
        bm25_scores = bm25.get_scores(_tokenize(query))
        bm25_order = np.argsort(-bm25_scores)
        bm25_ranked_ids = [doc_ids[i] for i in bm25_order]
        hybrid_ranked_ids = reciprocal_rank_fusion([bi_ranked_ids, bm25_ranked_ids])
        hybrid_latency = (time.perf_counter() - t0) * 1000 + bi_latency
        latencies_ms["hybrid_rrf"].append(hybrid_latency)

        rankings = {"bi_only": bi_ranked_ids, "two_stage": two_stage_ranked_ids, "hybrid_rrf": hybrid_ranked_ids}

        row: Dict[str, Any] = {
            "query": query,
            "relevant_doc_ids": sorted(relevant),
            **{f"{m}_top5": rankings[m][:5] for m in METHODS},
        }
        for k in K_VALUES:
            for m in METHODS:
                p = precision_at_k(rankings[m], relevant, k)
                r = recall_at_k(rankings[m], relevant, k)
                scores[m][k].append(p)
                recalls[m][k].append(r)
                row.setdefault(f"precision_at_{k}", {})[m] = p
                row.setdefault(f"recall_at_{k}", {})[m] = r
        per_query.append(row)

    summary: Dict[str, Any] = {"n_queries": len(QUERY_RELEVANCE), "n_documents": len(DOCUMENTS)}
    print("5. Results (macro-averaged over queries)\n")
    header = f"{'metric':<8}" + "".join(f"{METHOD_LABELS[m]:>18}" for m in METHODS)
    print(header)
    for k in K_VALUES:
        p_row = {m: float(np.mean(scores[m][k])) for m in METHODS}
        r_row = {m: float(np.mean(recalls[m][k])) for m in METHODS}
        print(f"P@{k:<6}" + "".join(f"{p_row[m]:>18.3f}" for m in METHODS))
        print(f"R@{k:<6}" + "".join(f"{r_row[m]:>18.3f}" for m in METHODS))
        summary[f"precision_at_{k}"] = p_row
        summary[f"recall_at_{k}"] = r_row

    latency_p50 = {m: float(np.median(latencies_ms[m])) for m in METHODS}
    print("\nlatency p50 (ms):  " + "  ".join(f"{METHOD_LABELS[m]}={latency_p50[m]:.2f}" for m in METHODS))
    summary["latency_ms_p50"] = latency_p50
    summary["embed_corpus_s"] = embed_corpus_s
    summary["per_query"] = per_query

    with open("benchmarks/retrieval_ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nWrote benchmarks/retrieval_ablation_results.json")

    return summary


if __name__ == "__main__":
    run_ablation()
