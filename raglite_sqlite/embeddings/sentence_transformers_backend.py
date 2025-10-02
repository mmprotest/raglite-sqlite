from __future__ import annotations

from typing import Sequence

from .base import EmbeddingBackend


class SentenceTransformersBackend(EmbeddingBackend):
    """Wrapper around sentence-transformers models."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_texts(self, texts: Sequence[str], model_name: str | None = None) -> Sequence[Sequence[float]]:
        model = self._load()
        embeddings = model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True, device=self.device)
        return embeddings.astype("float32").tolist()
