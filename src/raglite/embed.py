"""Embedding utilities."""

from __future__ import annotations

import hashlib
import math
import struct
from array import array
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Sequence


@dataclass
class EmbeddingStore:
    model_name: str
    dimension: int

    def embed_many(self, texts: Sequence[str]) -> List[bytes]:  # pragma: no cover - overridden
        raise NotImplementedError


class DebugEmbeddingStore(EmbeddingStore):
    def __init__(self, dimension: int = 256):
        super().__init__(model_name="debug", dimension=dimension)

    def embed_many(self, texts: Sequence[str]) -> List[bytes]:
        vectors: List[bytes] = []
        for text in texts:
            vec = array("f", [0.0] * self.dimension)
            for token in text.split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                for i in range(0, len(digest), 4):
                    idx = (digest[i] + i) % self.dimension
                    value = struct.unpack("!f", digest[i : i + 4])[0]
                    vec[idx] += value
            norm = math.sqrt(sum(v * v for v in vec))
            if norm:
                for i in range(self.dimension):
                    vec[i] /= norm
            vectors.append(vec.tobytes())
        return vectors


class SentenceTransformerStore(EmbeddingStore):
    def __init__(self, model_name: str) -> None:
        self._model = _load_sentence_transformer(model_name)
        super().__init__(model_name=model_name, dimension=self._model.get_sentence_embedding_dimension())

    def embed_many(self, texts: Sequence[str]) -> List[bytes]:
        embeddings = self._model.encode(list(texts), convert_to_numpy=False, normalize_embeddings=True)
        return [array("f", vec).tobytes() for vec in embeddings]


@lru_cache(maxsize=4)
def get_embedding_store(model_name: str) -> EmbeddingStore:
    if model_name.lower() in {"debug", "hash"}:
        return DebugEmbeddingStore()
    return SentenceTransformerStore(model_name)


def embedding_from_bytes(blob: bytes) -> array:
    vec = array("f")
    vec.frombytes(blob)
    return vec


def _load_sentence_transformer(model_name: str):  # pragma: no cover - heavy load
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("sentence-transformers is required for embedding") from exc
    return SentenceTransformer(model_name)
