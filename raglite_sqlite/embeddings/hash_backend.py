from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Sequence

from ..utils import normalize_text
from .base import EmbeddingBackend


class HashingBackend(EmbeddingBackend):
    """Lightweight hashing-based embedding backend."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self.model_name = f"hashing-{dim}"

    def embed_texts(self, texts: Sequence[str], model_name: str | None = None) -> Sequence[Sequence[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        tokens = [token for token in normalize_text(text).split() if token]
        if not tokens:
            return [0.0] * self.dim
        counts = Counter(tokens)
        vector = [0.0] * self.dim
        for token, weight in counts.items():
            index = self._bucket(token)
            vector[index] += float(weight)
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        return int.from_bytes(digest, "big") % self.dim
