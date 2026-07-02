from benchmarks.run_retrieval_ablation import precision_at_k, recall_at_k


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
