"""
End-to-end RAG eval against a REAL ingested corpus (not hand-written contexts).

RESULTS.md admits the existing RAGAS baseline uses hand-written questions,
answers, and contexts — "standalone eval -- hand-written contexts from LLMOps
domain" (see run_ragas.py). No document had ever actually been ingested into
this repo's FAISS index, so no RAGAS score here had ever reflected the real
retriever/reranker/synthesizer chain.

This script closes that gap:
  1. Ingests this repo's own docs into a real FAISS index via IngestionPipeline
     (real chunking, real sentence-transformer embeddings, real FAISS write).
  2. Runs RetrieverAgent.retrieve() for real against that index — the contexts
     below are never hand-written, they're whatever the real retriever found.
  3. Reranks via the real cross-encoder in RerankerAgent.
  4. If OPENAI_API_KEY is set: calls the real SynthesizerAgent and then the
     real RAGASTracker (actual `ragas` package, LLM-judged) on the result.
     If not: skips the LLM-judged step honestly and reports retrieval-only
     metrics that need no LLM (did the right source doc make it into top-k).

Usage:
    python -m src.benchmarks.run_real_corpus_eval
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Use a dedicated index so this never collides with (or depends on) any other
# index a developer may already have under data/. Must be set before the
# first import of src.ingestion.pipeline, which reads these at module load.
os.environ.setdefault("INDEX_PATH", str(REPO_ROOT / "data" / "real_corpus_eval.index"))
os.environ.setdefault("META_PATH", str(REPO_ROOT / "data" / "real_corpus_eval.meta.json"))

sys.path.insert(0, str(REPO_ROOT))

from src.agents.reranker import RerankerAgent  # noqa: E402
from src.agents.retriever import RetrieverAgent  # noqa: E402
from src.agents.synthesizer import SynthesizerAgent  # noqa: E402
from src.ingestion.pipeline import IngestionPipeline  # noqa: E402

CORPUS_FILES = [
    "README.md",
    "RESULTS.md",
    "docs/ARCHITECTURE.md",
    "docs/PHYSICS_SIMULATION.md",
    "docs/FLEET_SCHEDULING.md",
    "docs/PROJECT_STRUCTURE.md",
]

# Real questions with real ground truth, drawn from the docs above — not
# invented LLMOps trivia. `expected_source` lets us check retrieval precision
# without needing an LLM judge.
QUESTIONS = [
    {
        "question": "Why does HALO use FAISS instead of a managed vector database?",
        "ground_truth": (
            "FAISS runs in-process, avoiding a network hop, managed service cost, "
            "or vendor lock-in. The 4-shard distributed design gives horizontal "
            "scale without changing the query interface; the trade-off is no "
            "real-time updates as clean as Pinecone or Weaviate."
        ),
        "expected_source": "docs/ARCHITECTURE.md",
    },
    {
        "question": (
            "What did HALO's retrieval ablation actually find when comparing "
            "two-stage cross-encoder reranking and BM25+dense hybrid against "
            "plain bi-encoder retrieval?"
        ),
        "ground_truth": (
            "On n=22 hand-labeled queries, bi-encoder-only retrieval won or tied "
            "on every precision/recall metric against both two-stage cross-encoder "
            "reranking and the BM25+dense RRF hybrid — neither alternative beat "
            "plain cosine similarity."
        ),
        "expected_source": "docs/ARCHITECTURE.md",
    },
    {
        "question": "Why does HALO use QLoRA instead of full fine-tuning?",
        "ground_truth": (
            "Full fine-tuning an 8B model needs roughly 80GB of GPU memory. QLoRA "
            "compresses the frozen base model to 4-bit and trains only LoRA "
            "adapter matrices (under 1% of parameters), dropping the hardware "
            "requirement from 4x A100s to a single consumer GPU."
        ),
        "expected_source": "docs/ARCHITECTURE.md",
    },
    {
        "question": "Why does HALO's multi-agent system use circuit breakers?",
        "ground_truth": (
            "A slow or failing verifier agent would block the whole pipeline under "
            "naive retry. The circuit breaker opens after N consecutive failures, "
            "immediately returning a degraded, confidence-penalized response instead "
            "of waiting for timeouts, then probes once after a cooldown to test recovery."
        ),
        "expected_source": "docs/ARCHITECTURE.md",
    },
    {
        "question": "What does HALO's fleet reconciler handle for job scheduling?",
        "ground_truth": (
            "A deterministic job scheduler for worker fleets with timezone-aware "
            "calendars, resource constraints, and failure recovery, wrapping a "
            "reference reconciler implementation via src/scheduling/reconciler.py."
        ),
        "expected_source": "docs/FLEET_SCHEDULING.md",
    },
    {
        "question": "What physics does PhysicalAI's batched simulation model?",
        "ground_truth": (
            "Batched relativistic spin transport simulations: an RK4 integrator for "
            "proper-time evolution, the Thomas-BMT equation for spin precession in "
            "electromagnetic fields, and EDM coupling for electric dipole moment "
            "sensitivity, aimed at fundamental physics experiments like EDM searches "
            "in storage rings."
        ),
        "expected_source": "docs/PHYSICS_SIMULATION.md",
    },
    {
        "question": "What lives in HALO's src/csrc and src/cuda_ext directories?",
        "ground_truth": (
            "src/csrc holds custom CUDA kernels (fused attention, RMSNorm, top-k "
            "sampling); src/cuda_ext holds fused softmax+temperature, RoPE, and "
            "top-p sampling kernels."
        ),
        "expected_source": "docs/PROJECT_STRUCTURE.md",
    },
    {
        "question": (
            "What held-out accuracy did HALO's cost-aware query-complexity "
            "classifier achieve, and on what sample size?"
        ),
        "ground_truth": (
            "86.4% held-out accuracy (n=22), from a hand-labeled seed set of "
            "n=126, using logistic regression on sentence embeddings."
        ),
        "expected_source": "RESULTS.md",
    },
]


def _load_corpus() -> list[dict[str, str]]:
    docs = []
    for rel_path in CORPUS_FILES:
        path = REPO_ROOT / rel_path
        docs.append(
            {
                "id": rel_path,
                "source": rel_path,
                "text": path.read_text(encoding="utf-8"),
            }
        )
    return docs


def main() -> dict:
    print(f"Ingesting {len(CORPUS_FILES)} real documents into a fresh FAISS index...")
    pipeline = IngestionPipeline()
    pipeline.ingest_documents(_load_corpus())
    print(f"Indexed {pipeline.index.ntotal} chunks -> {os.environ['INDEX_PATH']}\n")

    retriever = RetrieverAgent(top_k=5)
    reranker = RerankerAgent(top_k=3)
    synthesizer = SynthesizerAgent()

    have_llm = bool(os.getenv("OPENAI_API_KEY"))
    results = []
    hits = 0

    for item in QUESTIONS:
        retrieved = retriever.retrieve(item["question"])
        reranked = reranker.rerank(item["question"], retrieved)
        sources = [c["source"] for c in reranked]
        hit = item["expected_source"] in sources
        hits += int(hit)

        row = {
            "question": item["question"],
            "expected_source": item["expected_source"],
            "retrieved_sources": sources,
            "top1_score": reranked[0]["rerank_score"] if reranked else None,
            "expected_source_in_topk": hit,
        }

        if have_llm:
            synth = synthesizer.synthesize(item["question"], reranked)
            row["answer"] = synth["answer"]
            row["contexts"] = [c["text"] for c in reranked]
            row["ground_truth"] = item["ground_truth"]

        results.append(row)

    recall_at_k = hits / len(QUESTIONS)
    print("=" * 60)
    print("REAL RETRIEVAL — expected source in top-k reranked results")
    print("=" * 60)
    for row in results:
        mark = "HIT " if row["expected_source_in_topk"] else "MISS"
        print(f"  [{mark}] {row['question'][:64]}")
    print(f"\n  recall@k (source-level): {recall_at_k:.3f}  ({hits}/{len(QUESTIONS)})")

    output = {
        "mode": "full_ragas" if have_llm else "retrieval_only",
        "recall_at_k_source_level": recall_at_k,
        "n_questions": len(QUESTIONS),
        "results": results,
    }

    if have_llm:
        from src.mlops.ragas_tracker import RAGASTracker, build_eval_dataset

        print(
            "\nOPENAI_API_KEY set — running the real, LLM-judged RAGAS eval "
            "on these real contexts..."
        )
        ds = build_eval_dataset(
            [
                {
                    "question": r["question"],
                    "answer": r["answer"],
                    "contexts": r["contexts"],
                    "ground_truth": r["ground_truth"],
                }
                for r in results
            ]
        )
        tracker = RAGASTracker(
            experiment_name="real-corpus-eval",
            baseline_path=str(REPO_ROOT / "mlops" / "real_corpus_ragas_baseline.json"),
        )
        scores = tracker.evaluate(ds, run_name="real-corpus-eval")
        output["ragas_scores"] = scores
        print(f"\n  RAGAS scores (real retrieval, real generation): {scores}")
    else:
        print(
            "\nOPENAI_API_KEY not set — skipping LLM-judged RAGAS scoring "
            "(faithfulness/answer_relevancy need a real generated answer and "
            "an LLM judge). The recall@k above is real and needs no LLM."
        )

    out_path = REPO_ROOT / "src" / "benchmarks" / "real_corpus_eval_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nFull results written to {out_path}")
    return output


if __name__ == "__main__":
    main()
