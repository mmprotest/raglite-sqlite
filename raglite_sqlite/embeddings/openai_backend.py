from __future__ import annotations

import os
from typing import Sequence

from .base import EmbeddingBackend


class OpenAIBackend(EmbeddingBackend):
    """Embedding backend using the OpenAI API."""

    def __init__(self, default_model: str = "text-embedding-3-small", batch_size: int = 64) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.default_model = default_model
        self.batch_size = batch_size
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)

    def embed_texts(self, texts: Sequence[str], model_name: str | None = None) -> Sequence[Sequence[float]]:
        model = model_name or self.default_model
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = self._client.embeddings.create(model=model, input=batch)
            for item in response.data:
                embeddings.append([float(value) for value in item.embedding])
        return embeddings
