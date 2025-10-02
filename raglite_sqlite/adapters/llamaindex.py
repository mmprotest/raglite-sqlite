from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..api import RagLite
from ..embeddings.base import EmbeddingBackend


class RagLiteNodeRetriever:
    """Minimal adapter compatible with LlamaIndex retriever interface."""

    def __init__(self, rag: RagLite, *, backend: Optional[EmbeddingBackend] = None, k: int = 4) -> None:
        self.rag = rag
        self.backend = backend
        self.k = k

    def retrieve(self, query: str, **kwargs: Any) -> List[Dict[str, Any]]:
        results = self.rag.search(query, k=self.k, embedding_backend=self.backend)
        nodes: List[Dict[str, Any]] = []
        for item in results:
            nodes.append(
                {
                    "text": item.get("text", ""),
                    "id": item.get("chunk_id"),
                    "metadata": {
                        "doc_id": item.get("doc_id"),
                        "section": item.get("section"),
                        "source_path": item.get("source_path"),
                        "tags": item.get("tags"),
                    },
                }
            )
        return nodes
