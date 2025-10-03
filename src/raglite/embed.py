"""Embedding utilities."""

from __future__ import annotations

import hashlib
import math
from array import array
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Sequence


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
                clean = token.strip().lower()
                if not clean:
                    continue
                digest = hashlib.sha256(clean.encode("utf-8")).digest()
                for i in range(0, len(digest), 4):
                    chunk = digest[i : i + 4]
                    if not chunk:
                        continue
                    idx = int.from_bytes(chunk, "big", signed=False) % self.dimension
                    sign = 1.0 if chunk[0] % 2 == 0 else -1.0
                    vec[idx] += sign
                for alias in _SYNONYMS.get(clean, ()):  # lightweight synonym boost
                    adigest = hashlib.sha256(alias.encode("utf-8")).digest()
                    for i in range(0, len(adigest), 4):
                        chunk = adigest[i : i + 4]
                        if not chunk:
                            continue
                        idx = int.from_bytes(chunk, "big", signed=False) % self.dimension
                        sign = 1.0 if chunk[0] % 2 == 0 else -1.0
                        vec[idx] += sign * 0.5
                for gram in _character_ngrams(clean):
                    gdigest = hashlib.sha256(f"char:{gram}".encode("utf-8")).digest()
                    for i in range(0, len(gdigest), 4):
                        chunk = gdigest[i : i + 4]
                        if not chunk:
                            continue
                        idx = int.from_bytes(chunk, "big", signed=False) % self.dimension
                        sign = 1.0 if chunk[0] % 2 == 0 else -1.0
                        vec[idx] += sign
            norm = math.sqrt(sum(v * v for v in vec))
            if norm:
                for i in range(self.dimension):
                    vec[i] /= norm
            vectors.append(vec.tobytes())
        return vectors


class SentenceTransformerStore(EmbeddingStore):
    def __init__(self, model_name: str) -> None:
        self._model = _load_sentence_transformer(model_name)
        super().__init__(
            model_name=model_name, dimension=self._model.get_sentence_embedding_dimension()
        )

    def embed_many(self, texts: Sequence[str]) -> List[bytes]:
        embeddings = self._model.encode(
            list(texts), convert_to_numpy=False, normalize_embeddings=True
        )
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


def _character_ngrams(text: str, n: int = 3) -> List[str]:
    if len(text) < n:
        return [text]
    return [text[i : i + n] for i in range(len(text) - n + 1)]


_SYNONYMS = {
    "sync": {"synchronization", "replication", "mirror"},
    "synchronization": {"sync", "replication"},
    "replicates": {"replication", "sync"},
    "replication": {"replicates", "sync"},
    "backup": {"backups", "archives"},
    "backups": {"backup", "archives"},
    "wal": {"write-ahead-log"},
    "latest": {"newest"},
    "newest": {"latest"},
    "wins": {"prevails"},
    "prevails": {"wins"},
}
