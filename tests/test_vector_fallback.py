import sqlite3
from pathlib import Path

import raglite.vector.backend as backend_module
from raglite.embed import DebugEmbeddingStore, embedding_from_bytes
from raglite.vector.backend import detect_backend
from raglite.vector.python_fallback import PythonFallbackBackend


def test_python_fallback_similarity(tmp_path: Path):
    conn = sqlite3.connect(tmp_path / "vec.db")
    conn.executescript(
        """
        CREATE TABLE embeddings(chunk_id INTEGER PRIMARY KEY, embedding BLOB);
        """
    )
    conn.execute(
        "INSERT INTO embeddings VALUES (?, ?)",
        (1, DebugEmbeddingStore(4).embed_many(["hello"])[0]),
    )
    conn.execute(
        "INSERT INTO embeddings VALUES (?, ?)",
        (2, DebugEmbeddingStore(4).embed_many(["another"])[0]),
    )
    conn.commit()
    backend = PythonFallbackBackend()
    store = DebugEmbeddingStore(4)
    query = embedding_from_bytes(store.embed_many(["hello"])[0])
    results = backend.search(conn, query, top_n=2)
    assert results and results[0].chunk_id == 1


def test_detect_backend_prefers_extension(monkeypatch, tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "vec.db")
    conn.execute("CREATE TABLE embeddings(id INTEGER PRIMARY KEY)")

    class DummyBackend:
        name = "sqlite-extension"

        def search(self, *args, **kwargs):
            return []

    def fake_create(_conn):  # pragma: no cover - simple stub
        return DummyBackend()

    monkeypatch.setattr(
        backend_module.SQLiteExtensionBackend,
        "create",
        staticmethod(fake_create),
    )
    backend = detect_backend(conn)
    assert backend.name == "sqlite-extension"


def test_detect_backend_fallback(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "vec_fb.db")
    conn.execute("CREATE TABLE embeddings(id INTEGER PRIMARY KEY)")
    backend = detect_backend(conn)
    assert backend.name in {"python-fallback", "none"}
