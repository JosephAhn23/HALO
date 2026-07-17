"""
Query-complexity classifier for the cost-aware router.

Trains a multinomial logistic regression on top of the same sentence
embeddings used elsewhere in the retrieval stack (``ingestion.pipeline.
EmbeddingModel``), rather than reusing the keyword-heuristic
``classify_complexity`` in ``routing.py``. That heuristic only ever routes
*which agents run* (see ``ComplexityRouter``); it was never wired to model
selection or cost, and a fixed keyword list doesn't generalize to phrasing it
wasn't written for.

This is trained on a small hand-labeled seed set (n=126, see
``cost_router_data.py``) — held-out accuracy is reported honestly by
``evaluate()`` rather than assumed. Treat it as a baseline, not a
production-grade classifier: scikit-learn is available transitively (pulled
in by ``umap-learn``, which is a core dependency) even though it isn't listed
directly in ``pyproject.toml``.
"""
from __future__ import annotations

import logging
import random
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.agents.multi_agent.cost_router_data import LabeledQuery

logger = logging.getLogger(__name__)

LABELS = ("low", "medium", "high")


def train_test_split(
    labeled_queries: List[LabeledQuery], test_size: float = 0.2, seed: int = 13
) -> Tuple[List[LabeledQuery], List[LabeledQuery]]:
    """Stratified split so each label is represented in both halves."""
    by_label: Dict[str, List[LabeledQuery]] = {label: [] for label in LABELS}
    for item in labeled_queries:
        by_label[item["label"]].append(item)

    rng = random.Random(seed)
    train: List[LabeledQuery] = []
    test: List[LabeledQuery] = []
    for label, items in by_label.items():
        shuffled = items[:]
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_size))
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])
    return train, test


@dataclass
class ClassifierEvalReport:
    accuracy: float
    n_test: int
    per_label_accuracy: Dict[str, float] = field(default_factory=dict)
    confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "n_test": self.n_test,
            "per_label_accuracy": self.per_label_accuracy,
            "confusion": self.confusion,
        }


class ComplexityClassifier:
    """Embeds a query and predicts one of ``LABELS`` via logistic regression."""

    def __init__(self, embedder: Optional[Any] = None) -> None:
        self._embedder = embedder
        self._model = None

    def get_embedder(self):
        """Public accessor so callers (e.g. the eval harness) can embed
        additional text with the exact same model/instance as this classifier."""
        if self._embedder is None:
            from src.ingestion.pipeline import EmbeddingModel

            self._embedder = EmbeddingModel()
        return self._embedder

    def _embed(self, queries: List[str]) -> np.ndarray:
        return np.asarray(self.get_embedder().embed(queries))

    def fit(self, labeled_queries: List[LabeledQuery]) -> "ComplexityClassifier":
        from sklearn.linear_model import LogisticRegression

        X = self._embed([item["query"] for item in labeled_queries])
        y = [item["label"] for item in labeled_queries]
        self._model = LogisticRegression(max_iter=1000)
        self._model.fit(X, y)
        return self

    def predict(self, query: str) -> str:
        if self._model is None:
            raise RuntimeError("ComplexityClassifier must be fit() before predict()")
        X = self._embed([query])
        return str(self._model.predict(X)[0])

    def predict_proba(self, query: str) -> Dict[str, float]:
        if self._model is None:
            raise RuntimeError("ComplexityClassifier must be fit() before predict_proba()")
        X = self._embed([query])
        probs = self._model.predict_proba(X)[0]
        return {cls: float(p) for cls, p in zip(self._model.classes_, probs)}

    def evaluate(self, held_out: List[LabeledQuery]) -> ClassifierEvalReport:
        if self._model is None:
            raise RuntimeError("ComplexityClassifier must be fit() before evaluate()")
        confusion: Dict[str, Dict[str, int]] = {
            gold: {pred: 0 for pred in LABELS} for gold in LABELS
        }
        correct = 0
        per_label_total: Dict[str, int] = {label: 0 for label in LABELS}
        per_label_correct: Dict[str, int] = {label: 0 for label in LABELS}
        for item in held_out:
            pred = self.predict(item["query"])
            gold = item["label"]
            confusion[gold][pred] += 1
            per_label_total[gold] += 1
            if pred == gold:
                correct += 1
                per_label_correct[gold] += 1

        per_label_accuracy = {
            label: (per_label_correct[label] / per_label_total[label]) if per_label_total[label] else 0.0
            for label in LABELS
        }
        return ClassifierEvalReport(
            accuracy=correct / len(held_out) if held_out else 0.0,
            n_test=len(held_out),
            per_label_accuracy=per_label_accuracy,
            confusion=confusion,
        )


_default_classifier: Optional[ComplexityClassifier] = None
_default_eval_report: Optional[ClassifierEvalReport] = None
_classifier_lock = threading.Lock()


def train_default_classifier(
    seed: int = 13,
) -> Tuple[ComplexityClassifier, ClassifierEvalReport]:
    """Train on the hand-labeled seed set, report real held-out accuracy."""
    from src.agents.multi_agent.cost_router_data import LABELED_QUERIES

    train, test = train_test_split(LABELED_QUERIES, seed=seed)
    clf = ComplexityClassifier().fit(train)
    report = clf.evaluate(test)
    logger.info(
        "ComplexityClassifier trained: held-out accuracy=%.3f on n=%d (seed=%d)",
        report.accuracy,
        report.n_test,
        seed,
    )
    return clf, report


def get_default_classifier() -> ComplexityClassifier:
    """Thread-safe lazy singleton, mirrors ``orchestrator.get_pipeline()``."""
    global _default_classifier, _default_eval_report
    if _default_classifier is None:
        with _classifier_lock:
            if _default_classifier is None:
                _default_classifier, _default_eval_report = train_default_classifier()
    return _default_classifier


def get_default_eval_report() -> Optional[ClassifierEvalReport]:
    """None until get_default_classifier() has been called at least once."""
    return _default_eval_report
