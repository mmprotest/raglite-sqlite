from __future__ import annotations

from pathlib import Path

from raglite_sqlite.embeddings.base import DummyBackend


def test_index_and_search(temp_db: tuple[Path, object, object]) -> None:
    db_path, rag, backend = temp_db
    data_dir = Path(__file__).parent / "data"
    result = rag.index([str(data_dir)], embedding_backend=backend)
    stats = rag.stats()
    assert result["chunks"] > 0
    assert stats["documents"] >= 1
    results = rag.search("sample", embedding_backend=backend)
    assert results, "Expected at least one search result"
    top = results[0]
    assert "sample" in top["text"].lower()


def test_idempotent_index(temp_db: tuple[Path, object, object]) -> None:
    db_path, rag, backend = temp_db
    data_dir = Path(__file__).parent / "data"
    rag.index([str(data_dir)], embedding_backend=backend)
    result = rag.index([str(data_dir)], embedding_backend=backend)
    assert result["skipped"] >= 1


def test_embedding_cache(temp_db: tuple[Path, object, object]) -> None:
    db_path, rag, backend = temp_db
    data_dir = Path(__file__).parent / "data"
    backend.calls = 0
    rag.index([str(data_dir)], embedding_backend=backend)
    first_calls = backend.calls
    rag.index([str(data_dir)], embedding_backend=backend, skip_unchanged=False)
    assert backend.calls == first_calls
