from __future__ import annotations

from pathlib import Path

from raglite_sqlite.rerank import register_reranker


class ReverseReranker:
    def rerank(self, query: str, results):  # type: ignore[override]
        return list(reversed(list(results)))


def test_custom_reranker(temp_db: tuple[Path, object, object]) -> None:
    db_path, rag, backend = temp_db
    data_dir = Path(__file__).parent / "data"
    rag.index([str(data_dir)], embedding_backend=backend)

    class PrefixReranker:
        def rerank(self, query: str, results):  # type: ignore[override]
            ordered = sorted(results, key=lambda item: item["doc_id"], reverse=True)
            for idx, item in enumerate(ordered):
                item["rerank_score"] = float(len(ordered) - idx)
            return ordered

    results = rag.search("sample", embedding_backend=backend, reranker=PrefixReranker())
    assert results
    assert results[0]["rerank_score"] >= results[-1]["rerank_score"]


def test_registry_reranker(temp_db: tuple[Path, object, object]) -> None:
    db_path, rag, backend = temp_db
    data_dir = Path(__file__).parent / "data"
    rag.index([str(data_dir)], embedding_backend=backend)

    register_reranker("reverse", lambda **_: ReverseReranker())
    baseline = rag.search("sample", embedding_backend=backend)
    results = rag.search("sample", embedding_backend=backend, reranker="reverse")
    assert results
    if len(baseline) > 1:
        assert results[0]["chunk_id"] == baseline[-1]["chunk_id"]
