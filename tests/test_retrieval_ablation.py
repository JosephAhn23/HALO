from src.benchmarks.run_retrieval_ablation import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank_fusion,
)


def test_precision_at_k_all_relevant() -> None:
    assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0


def test_precision_at_k_none_relevant() -> None:
    assert precision_at_k(["a", "b", "c"], {"x", "y"}, 3) == 0.0


def test_precision_at_k_partial() -> None:
    assert precision_at_k(["a", "b", "c"], {"a"}, 3) == 1 / 3


def test_precision_at_k_only_considers_top_k() -> None:
    assert precision_at_k(["a", "x", "x", "x", "b"], {"a", "b"}, 1) == 1.0


def test_recall_at_k_finds_all() -> None:
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0


def test_recall_at_k_misses_some() -> None:
    assert recall_at_k(["a", "x", "x"], {"a", "b"}, 3) == 0.5


def test_recall_at_k_empty_relevant_set_is_zero() -> None:
    assert recall_at_k(["a", "b"], set(), 3) == 0.0


def test_recall_at_k_respects_k_cutoff() -> None:
    assert recall_at_k(["x", "x", "a"], {"a"}, 2) == 0.0
    assert recall_at_k(["x", "x", "a"], {"a"}, 3) == 1.0


def test_rrf_boosts_doc_ranked_high_in_both_lists() -> None:
    ranking_a = ["a", "b", "c"]
    ranking_b = ["a", "c", "b"]
    fused = reciprocal_rank_fusion([ranking_a, ranking_b])
    assert fused[0] == "a"  # rank 1 in both -- should win outright


def test_rrf_ranks_doc_present_in_both_over_doc_present_in_one() -> None:
    # "a" is rank 5 in both rankers (RRF score 2/65 ~= 0.0308, the only doc
    # shared between the two rankings); "z" is rank 1 in one ranker only
    # (RRF score 1/61 ~= 0.0164) -- appearing in both, even at a middling
    # rank, should still beat appearing at #1 in just one.
    ranking_a = ["p1", "p2", "p3", "p4", "a"]
    ranking_b = ["z", "q2", "q3", "q4", "a"]
    fused = reciprocal_rank_fusion([ranking_a, ranking_b])
    assert fused[0] == "a"


def test_rrf_includes_every_doc_seen_in_any_ranking() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["c"]])
    assert set(fused) == {"a", "b", "c"}
