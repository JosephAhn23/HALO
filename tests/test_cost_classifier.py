"""
Integration test for the real (non-mocked) complexity classifier.

Unlike test_cost_router.py, this deliberately does NOT mock the embedder or
sklearn — the whole point of this module is to report genuine held-out
accuracy, so the test has to actually train and evaluate it. The MiniLM
embedder is small enough that this stays fast.
"""

from src.agents.multi_agent.cost_classifier import (
    LABELS,
    ComplexityClassifier,
    train_default_classifier,
    train_test_split,
)
from src.agents.multi_agent.cost_router_data import LABELED_QUERIES


def test_train_test_split_is_stratified_and_disjoint() -> None:
    train, test = train_test_split(LABELED_QUERIES, test_size=0.2, seed=13)
    train_queries = {item["query"] for item in train}
    test_queries = {item["query"] for item in test}
    assert not (train_queries & test_queries)
    assert len(train) + len(test) == len(LABELED_QUERIES)
    for label in LABELS:
        assert any(item["label"] == label for item in test)


def test_classifier_beats_random_baseline_on_held_out_set() -> None:
    """Real accuracy, not asserted to be perfect — just meaningfully above
    the 1/3 random baseline for three balanced-ish classes."""
    clf, report = train_default_classifier(seed=13)
    assert report.n_test > 0
    assert report.accuracy > 0.5, (
        f"held-out accuracy {report.accuracy:.3f} did not clear the sanity "
        f"threshold; confusion={report.confusion}"
    )


def test_classifier_predicts_a_valid_label() -> None:
    clf, _ = train_default_classifier(seed=13)
    assert clf.predict("What is a vector database?") in LABELS
    assert (
        clf.predict("Compare QLoRA against full fine-tuning and analyze the tradeoffs.") in LABELS
    )


def test_predict_proba_sums_to_one() -> None:
    clf, _ = train_default_classifier(seed=13)
    probs = clf.predict_proba("What does LoRA stand for?")
    assert set(probs.keys()) == set(LABELS)
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_predict_before_fit_raises() -> None:
    clf = ComplexityClassifier()
    try:
        clf.predict("anything")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
