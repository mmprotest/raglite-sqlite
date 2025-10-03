from __future__ import annotations

from pathlib import Path
from typing import Any, List

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import VectorStore

from raglite.api import RagliteAPI, RagliteConfig


class RagliteVectorStore(VectorStore):
    def __init__(self, db_path: Path) -> None:
        self.api = RagliteAPI(RagliteConfig(db_path))
        self.api.init_db()

    def add(self, nodes: List[TextNode]) -> List[str]:  # pragma: no cover - integration shim
        temp_dir = Path("llamaindex_tmp")
        temp_dir.mkdir(exist_ok=True)
        ids = []
        for node in nodes:
            path = temp_dir / f"node_{node.node_id}.txt"
            path.write_text(node.get_content(), encoding="utf-8")
            self.api.index(path.parent, strategy="fixed")
            ids.append(node.node_id)
        return ids

    def query(self, query: str, top_k: int = 5) -> List[TextNode]:  # pragma: no cover - integration shim
        results = self.api.query(query, top_k=top_k)
        return [TextNode(text=r.text, id_=str(r.chunk_id), metadata=r.metadata) for r in results]


if __name__ == "__main__":
    store = RagliteVectorStore(Path("llamaindex.db"))
    demo_dir = Path(__file__).resolve().parents[1] / "demo" / "mini_corpus"
    store.api.index(demo_dir, strategy="fixed")
    print(store.query("quick start guide")[0])
