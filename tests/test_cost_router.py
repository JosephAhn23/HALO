"""
Unit tests for CostAwareRouter, isolated from the real classifier (which is
covered separately in test_cost_classifier.py) via a fake stand-in — these
tests are about routing/abstention logic and pricing math, not classifier
accuracy, so they should stay fast and not load the embedding model.
"""

from unittest.mock import MagicMock

from src.agents.multi_agent.cost_router import (
    MODEL_PRICING_USD_PER_1K,
    TIER_MODEL_MAP,
    CostAwareRouter,
    estimate_cost_usd,
)


def _fake_classifier(prediction: str) -> MagicMock:
    clf = MagicMock()
    clf.predict.return_value = prediction
    return clf


def test_abstains_when_no_chunks_retrieved() -> None:
    router = CostAwareRouter(classifier=_fake_classifier("low"))
    decision = router.route("some query", [])
    assert decision.abstain is True
    assert decision.model is None
    assert decision.n_retrieved == 0


def test_abstains_when_top_score_below_threshold() -> None:
    router = CostAwareRouter(classifier=_fake_classifier("low"), abstain_threshold=0.35)
    decision = router.route("some query", [{"text": "x", "retrieval_score": 0.1}])
    assert decision.abstain is True
    assert decision.retrieval_top_score == 0.1


def test_does_not_abstain_when_top_score_meets_threshold() -> None:
    router = CostAwareRouter(classifier=_fake_classifier("low"), abstain_threshold=0.35)
    decision = router.route("some query", [{"text": "x", "retrieval_score": 0.5}])
    assert decision.abstain is False
    assert decision.model == TIER_MODEL_MAP["low"]


def test_prefers_rerank_score_over_retrieval_score() -> None:
    """rerank_score is the more precise Stage-2 signal; a low retrieval_score
    shouldn't cause an abstain if reranking already boosted relevance."""
    router = CostAwareRouter(classifier=_fake_classifier("low"), abstain_threshold=0.35)
    chunks = [{"text": "x", "retrieval_score": 0.1, "rerank_score": 0.9}]
    decision = router.route("some query", chunks)
    assert decision.abstain is False
    assert decision.retrieval_top_score == 0.9


def test_routes_each_tier_to_its_mapped_model() -> None:
    for tier, expected_model in TIER_MODEL_MAP.items():
        router = CostAwareRouter(classifier=_fake_classifier(tier), abstain_threshold=0.0)
        decision = router.route("q", [{"text": "x", "retrieval_score": 0.9}])
        assert decision.complexity_tier == tier
        assert decision.model == expected_model


def test_classifier_only_called_when_not_abstaining() -> None:
    clf = _fake_classifier("high")
    router = CostAwareRouter(classifier=clf, abstain_threshold=0.35)
    router.route("q", [])
    clf.predict.assert_not_called()


def test_estimate_cost_usd_matches_pricing_table_arithmetic() -> None:
    rates = MODEL_PRICING_USD_PER_1K["gpt-4o-mini"]
    cost = estimate_cost_usd("gpt-4o-mini", prompt_tokens=2000, completion_tokens=500)
    expected = (2000 / 1000.0) * rates["prompt"] + (500 / 1000.0) * rates["completion"]
    assert abs(cost - expected) < 1e-12


def test_estimate_cost_usd_unknown_model_is_zero() -> None:
    assert estimate_cost_usd("not-a-real-model", 1000, 1000) == 0.0


def test_gpt4o_more_expensive_than_gpt4o_mini_for_same_tokens() -> None:
    cheap = estimate_cost_usd("gpt-4o-mini", 1000, 1000)
    expensive = estimate_cost_usd("gpt-4o", 1000, 1000)
    assert expensive > cheap
