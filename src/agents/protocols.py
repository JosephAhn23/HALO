"""
Agent protocols - testable interfaces for pipeline components.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str) -> list[dict[str, Any]]: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


@runtime_checkable
class Synthesizer(Protocol):
    def synthesize(self, query: str, context_chunks: list[dict[str, Any]]) -> dict[str, Any]: ...
