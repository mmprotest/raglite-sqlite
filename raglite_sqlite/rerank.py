from __future__ import annotations

from typing import Callable, Iterable, Protocol

from .typing import SearchResult


class Reranker(Protocol):
    def rerank(self, query: str, results: Iterable[SearchResult]) -> Iterable[SearchResult]:
        ...


class NoopReranker:
    def rerank(self, query: str, results: Iterable[SearchResult]) -> Iterable[SearchResult]:
        return results


RerankerFactory = Callable[..., Reranker]

_RERANKER_REGISTRY: dict[str, RerankerFactory] = {
    "none": lambda **_: NoopReranker(),
}


def register_reranker(name: str, factory: RerankerFactory) -> None:
    """Register a reranker factory."""

    _RERANKER_REGISTRY[name] = factory


def get_reranker(name: str, **options: object) -> Reranker:
    """Instantiate a reranker by name."""

    factory = _RERANKER_REGISTRY.get(name)
    if factory is None:
        raise KeyError(name)
    return factory(**options)


class CrossEncoderReranker:
    """Rerank using a sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", batch_size: int = 16) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "sentence-transformers is required for the cross-encoder reranker"
            ) from exc

        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size

    def rerank(self, query: str, results: Iterable[SearchResult]) -> Iterable[SearchResult]:
        result_list = list(results)
        if not result_list:
            return result_list
        pairs = [(query, item["text"]) for item in result_list]
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        for item, score in zip(result_list, scores):
            item["rerank_score"] = float(score)
        return sorted(result_list, key=lambda item: item.get("rerank_score", 0.0), reverse=True)


register_reranker("cross-encoder", lambda **opts: CrossEncoderReranker(**opts))
