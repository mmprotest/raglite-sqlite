from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

try:
    from langchain.schema import Document
except Exception:  # pragma: no cover - optional dependency
    Document = None  # type: ignore[assignment]

from ..api import RagLite
from ..embeddings.base import EmbeddingBackend
from ..typing import SearchResult


class RagLiteRetriever:
    """Minimal LangChain retriever wrapper."""

    def __init__(
        self,
        rag: RagLite,
        *,
        backend: Optional[EmbeddingBackend] = None,
        k: int = 4,
    ) -> None:
        self.rag = rag
        self.backend = backend
        self.k = k

    def get_relevant_documents(self, query: str, **kwargs: Any) -> List[Any]:
        results = self.rag.search(query, k=self.k, embedding_backend=self.backend)
        if Document is None:
            raise RuntimeError("LangChain is not installed")
        docs: List[Any] = []
        for item in results:
            metadata: Dict[str, Any] = {
                "doc_id": item.get("doc_id"),
                "section": item.get("section"),
                "source_path": item.get("source_path"),
                "tags": item.get("tags"),
            }
            docs.append(Document(page_content=item.get("text", ""), metadata=metadata))
        return docs

    async def aget_relevant_documents(self, query: str, **kwargs: Any) -> List[Any]:
        return self.get_relevant_documents(query, **kwargs)
