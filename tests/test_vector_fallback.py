import sqlite3
from pathlib import Path

from raglite.embed import DebugEmbeddingStore, embedding_from_bytes
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
