"""
Cost-aware model-tier router + retrieval-abstention gate.

Two independent decisions live here, both made *before* the synthesizer is
called (so a decision to abstain costs zero generation tokens):

1. Abstain vs. answer — if retrieval didn't return anything with a
   plausible similarity score, don't let the synthesizer generate on top of
   irrelevant context. This is the gap flagged in the RAG audit: today
   ``RetrieverAgent`` can return an empty or low-relevance list and the
   pipeline still synthesizes an answer from whatever context it has.
2. Cheap vs. expensive model tier — route "low"/"medium" complexity queries
   (per ``ComplexityClassifier``) to a cheaper model, "high" complexity to a
   stronger one. Existing ``ComplexityRouter`` in ``routing.py`` already
   classifies complexity, but only uses it to pick *which agents run*
   (researcher/critic/verifier) — never model choice, so it has zero effect
   on cost. This router closes that gap.

Pricing note
------------
``MODEL_PRICING_USD_PER_1K`` is a snapshot of published OpenAI per-token list
prices at the time this was written, structured the same way as the cost
table in RESULTS.md. Treat it as a documented assumption, not a live-fetched
number — verify against https://openai.com/api/pricing/ before citing it
externally, and re-check it if this project is revisited later.

Abstention threshold note
--------------------------
``DEFAULT_ABSTAIN_THRESHOLD`` = 0.18, empirically set from
``benchmarks/run_cost_router_eval.py``: no documents have been ingested into
this repo's FAISS index yet, so real retrieval_score/rerank_score
distributions don't exist. As a proxy, that script computes real cosine
similarity between the eval queries and a small in-domain reference corpus
(pulled from README.md) using the same embedding model the retriever uses.
On that measurement, unanswerable queries topped out at 0.174 and 0.18 caught
all of them while only false-abstaining 2/22 answerable queries (vs. 14/22 at
the previous placeholder of 0.35 — see benchmarks/cost_router_results.json).
This is still a proxy measurement, not a measurement against this project's
actual corpus — re-run that script and re-tune once real documents are
ingested, since a real FAISS index's score distribution may differ from a
16-sentence reference corpus.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.agents.multi_agent.cost_classifier import ComplexityClassifier, get_default_classifier

DEFAULT_ABSTAIN_THRESHOLD = float(os.getenv("COST_ROUTER_ABSTAIN_THRESHOLD", "0.18"))

# $ / 1K tokens, (prompt, completion). Snapshot — see module docstring.
MODEL_PRICING_USD_PER_1K: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "gpt-4o": {"prompt": 0.0025, "completion": 0.01},
}

# Complexity tier -> which model handles it.
TIER_MODEL_MAP: Dict[str, str] = {
    "low": "gpt-4o-mini",
    "medium": "gpt-4o-mini",
    "high": "gpt-4o",
}

ABSTAIN_ANSWER = (
    "I don't have enough relevant context to answer this confidently. "
    "This query appears to be outside the indexed knowledge base, or "
    "retrieval didn't find anything sufficiently related — rather than "
    "guess, I'm declining to answer."
)


@dataclass
class RouterDecision:
    abstain: bool
    complexity_tier: Optional[str]
    model: Optional[str]
    retrieval_top_score: Optional[float]
    n_retrieved: int
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "abstain": self.abstain,
            "complexity_tier": self.complexity_tier,
            "model": self.model,
            "retrieval_top_score": self.retrieval_top_score,
            "n_retrieved": self.n_retrieved,
            "reasoning": self.reasoning,
        }


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = MODEL_PRICING_USD_PER_1K.get(model)
    if rates is None:
        return 0.0
    return (prompt_tokens / 1000.0) * rates["prompt"] + (completion_tokens / 1000.0) * rates["completion"]


class CostAwareRouter:
    def __init__(
        self,
        classifier: Optional[ComplexityClassifier] = None,
        abstain_threshold: float = DEFAULT_ABSTAIN_THRESHOLD,
        score_keys: tuple[str, ...] = ("rerank_score", "retrieval_score"),
    ) -> None:
        self._classifier = classifier
        self.abstain_threshold = abstain_threshold
        # rerank_score (sigmoid-normalized cross-encoder logit) is preferred when
        # present — it's the more precise Stage-2 signal (see reranker.py) — and
        # retrieval_score (cosine similarity) is the fallback when reranking
        # didn't run (e.g. pipeline error before the rerank node).
        self.score_keys = score_keys

    def _get_classifier(self) -> ComplexityClassifier:
        if self._classifier is None:
            self._classifier = get_default_classifier()
        return self._classifier

    def _top_score(self, chunks: List[Dict[str, Any]]) -> Optional[float]:
        for key in self.score_keys:
            scores = [c[key] for c in chunks if c.get(key) is not None]
            if scores:
                return max(scores)
        return None

    def route(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> RouterDecision:
        n = len(retrieved_chunks)
        top_score = self._top_score(retrieved_chunks)

        if n == 0 or top_score is None:
            return RouterDecision(
                abstain=True,
                complexity_tier=None,
                model=None,
                retrieval_top_score=top_score,
                n_retrieved=n,
                reasoning="No chunks retrieved.",
            )
        if top_score < self.abstain_threshold:
            return RouterDecision(
                abstain=True,
                complexity_tier=None,
                model=None,
                retrieval_top_score=top_score,
                n_retrieved=n,
                reasoning=(
                    f"Top retrieval score {top_score:.3f} below abstain threshold "
                    f"{self.abstain_threshold:.3f}."
                ),
            )

        tier = self._get_classifier().predict(query)
        model = TIER_MODEL_MAP.get(tier, "gpt-4o-mini")
        return RouterDecision(
            abstain=False,
            complexity_tier=tier,
            model=model,
            retrieval_top_score=top_score,
            n_retrieved=n,
            reasoning=f"Classified '{tier}' complexity -> routed to {model}.",
        )
