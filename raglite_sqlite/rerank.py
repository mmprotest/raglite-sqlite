from __future__ import annotations

from typing import Iterable, Protocol

from .typing import SearchResult


class Reranker(Protocol):
    def rerank(self, query: str, results: Iterable[SearchResult]) -> Iterable[SearchResult]:
        ...


class NoopReranker:
    def rerank(self, query: str, results: Iterable[SearchResult]) -> Iterable[SearchResult]:
        return results
