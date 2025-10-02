from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class EmbeddingBackend(ABC):
    """Abstract embedding backend."""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str], model_name: str | None = None) -> Sequence[Sequence[float]]:
        """Return embeddings as float32 numpy array of shape (N, D)."""


class DummyBackend(EmbeddingBackend):
    """Deterministic embedding backend used for tests."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls = 0

    def embed_texts(self, texts: Sequence[str], model_name: str | None = None) -> Sequence[Sequence[float]]:
        self.calls += 1
        vectors: list[list[float]] = []
        for text in texts:
            h = abs(hash(text))
            vec = [(h >> i) & 0xFF for i in range(self.dim)]
            norm = sum(value * value for value in vec) ** 0.5
            if norm == 0:
                vec[0] = 1.0
                norm = 1.0
            vectors.append([value / norm for value in vec])
        return vectors
