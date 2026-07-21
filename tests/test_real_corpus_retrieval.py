"""
Real end-to-end retrieval test — no mocks, no `__new__` bypass.

tests/test_agents.py exercises RetrieverAgent/RerankerAgent by constructing
them with `RetrieverAgent.__new__(RetrieverAgent)` and hand-injecting a fake
FAISS index and metadata — useful for unit-testing the scoring/sorting logic,
but it never proves ingestion, embedding, or the real FAISS index actually
work together. This test does: it ingests real text through the real
IngestionPipeline, builds a real on-disk FAISS index, and confirms
RetrieverAgent's real constructor finds the right chunk for an unambiguous
query.
"""

from __future__ import annotations

import importlib
import os


def test_real_ingestion_and_retrieval_roundtrip(tmp_path, monkeypatch):
    index_path = tmp_path / "test.index"
    meta_path = tmp_path / "test.meta.json"
    monkeypatch.setenv("INDEX_PATH", str(index_path))
    monkeypatch.setenv("META_PATH", str(meta_path))

    # INDEX_PATH/META_PATH are read at import time, so force both modules to
    # re-read the env vars we just set rather than reusing an already-imported
    # (and possibly differently-configured) module from another test.
    import src.ingestion.pipeline as pipeline_module

    importlib.reload(pipeline_module)
    import src.agents.retriever as retriever_module

    importlib.reload(retriever_module)

    docs = [
        {
            "id": "doc_astronomy",
            "source": "astronomy.md",
            "text": (
                "Jupiter is the largest planet in the Solar System. It is a gas "
                "giant primarily composed of hydrogen and helium, and its Great "
                "Red Spot is a storm that has persisted for centuries."
            ),
        },
        {
            "id": "doc_cooking",
            "source": "cooking.md",
            "text": (
                "A classic French omelette is cooked over low heat while "
                "constantly stirring the eggs with a fork, then folded into a "
                "smooth, unbrowned roll just before it fully sets."
            ),
        },
    ]

    ingestion = pipeline_module.IngestionPipeline()
    ingestion.ingest_documents(docs)

    assert os.path.exists(index_path), "ingestion should write a real FAISS index to disk"
    assert os.path.exists(meta_path), "ingestion should write real chunk metadata to disk"
    assert ingestion.index.ntotal == 2

    retriever = retriever_module.RetrieverAgent(top_k=1)
    results = retriever.retrieve("What is the largest planet in the Solar System?")

    assert len(results) == 1
    assert results[0]["source"] == "astronomy.md"
    assert "retrieval_score" in results[0]
