from __future__ import annotations

from pathlib import Path
from typing import List

from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever

from raglite.api import RagliteAPI, RagliteConfig


class RagliteRetriever(BaseRetriever):
    def __init__(self, db_path: Path) -> None:
        self.api = RagliteAPI(RagliteConfig(db_path))
        self.api.init_db()

    def _get_relevant_documents(self, query: str) -> List[Document]:
        results = self.api.query(query, top_k=5)
        return [Document(page_content=r.text, metadata=r.metadata) for r in results]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:  # pragma: no cover - async wrapper
        return self._get_relevant_documents(query)


if __name__ == "__main__":
    retriever = RagliteRetriever(Path("langchain.db"))
    demo_dir = Path(__file__).resolve().parents[1] / "demo" / "mini_corpus"
    retriever.api.index(demo_dir, strategy="fixed")
    docs = retriever.get_relevant_documents("quick start guide")
    for doc in docs:
        print(doc)
