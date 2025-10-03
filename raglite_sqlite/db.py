from __future__ import annotations

import sqlite3
from array import array
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

from .utils import dumps_json, ensure_directory, loads_json

PRAGMAS = {
    "journal_mode": "WAL",
    "synchronous": "NORMAL",
    "temp_store": "MEMORY",
    "mmap_size": 268435456,
    "foreign_keys": 1,
}


class Database:
    def __init__(self, path: Path, create: bool = True) -> None:
        self.path = path
        ensure_directory(path)
        need_init = create and not path.exists()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._apply_pragmas()
        if need_init:
            self.run_schema()
        self._ensure_fts5()

    def _apply_pragmas(self) -> None:
        cur = self.conn.cursor()
        for key, value in PRAGMAS.items():
            cur.execute(f"PRAGMA {key}={value}")
        cur.close()

    def _ensure_fts5(self) -> None:
        cur = self.conn.execute("PRAGMA compile_options")
        options = {row[0] for row in cur.fetchall()}
        if "ENABLE_FTS5" not in options:
            raise RuntimeError("SQLite build does not support FTS5")

    def run_schema(self) -> None:
        schema_sql = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
        self.conn.executescript(schema_sql)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # Document helpers
    def upsert_document(
        self,
        doc_id: str,
        *,
        source_path: str,
        mime: str,
        tags: str | None,
        created_at: int,
        updated_at: int,
        sha256: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO documents(doc_id, source_path, mime, tags, created_at, updated_at, sha256)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                source_path=excluded.source_path,
                mime=excluded.mime,
                tags=excluded.tags,
                updated_at=excluded.updated_at,
                sha256=excluded.sha256
            """,
            (doc_id, source_path, mime, tags, created_at, updated_at, sha256),
        )

    def get_document(self, doc_id: str) -> Optional[sqlite3.Row]:
        cur = self.conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        return cur.fetchone()

    def delete_document(self, doc_id: str) -> None:
        self.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def list_documents(self) -> List[sqlite3.Row]:
        cur = self.conn.execute("SELECT * FROM documents")
        return list(cur.fetchall())

    # Chunk helpers
    def upsert_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        position: int,
        text: str,
        text_norm: str,
        section: str | None,
        sha256: str,
        extra: dict | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO chunks(chunk_id, doc_id, position, text, text_norm, section, sha256, extra)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                text=excluded.text,
                text_norm=excluded.text_norm,
                section=excluded.section,
                sha256=excluded.sha256,
                extra=excluded.extra
            """,
            (chunk_id, doc_id, position, text, text_norm, section, sha256, dumps_json(extra or {})),
        )
        self.conn.execute(
            "DELETE FROM chunk_fts WHERE chunk_id = ?",
            (chunk_id,),
        )
        self.conn.execute(
            "INSERT INTO chunk_fts(chunk_id, text_norm) VALUES(?, ?)",
            (chunk_id, text_norm),
        )

    def delete_chunk(self, chunk_id: str) -> None:
        self.conn.execute("DELETE FROM chunks WHERE chunk_id = ?", (chunk_id,))
        self.conn.execute("DELETE FROM chunk_fts WHERE chunk_id = ?", (chunk_id,))

    def iter_chunks_for_doc(self, doc_id: str) -> Iterable[sqlite3.Row]:
        cur = self.conn.execute("SELECT * FROM chunks WHERE doc_id = ? ORDER BY position", (doc_id,))
        return cur.fetchall()

    # Vector helpers
    def upsert_vector(
        self,
        chunk_id: str,
        vector: Sequence[float],
        model_name: str,
    ) -> None:
        blob = array("f", vector).tobytes()
        dim = len(vector)
        dtype = "float32"
        self.conn.execute(
            """
            INSERT INTO vectors(chunk_id, dim, dtype, vec, model_name, created_at)
            VALUES(?, ?, ?, ?, ?, strftime('%s','now'))
            ON CONFLICT(chunk_id) DO UPDATE SET
                dim=excluded.dim,
                dtype=excluded.dtype,
                vec=excluded.vec,
                model_name=excluded.model_name,
                created_at=excluded.created_at
            """,
            (chunk_id, dim, dtype, blob, model_name),
        )

    def get_vectors_by_ids(self, chunk_ids: Sequence[str]) -> dict[str, list[float]]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        cur = self.conn.execute(
            f"SELECT chunk_id, dim, dtype, vec FROM vectors WHERE chunk_id IN ({placeholders})",
            tuple(chunk_ids),
        )
        vectors: dict[str, list[float]] = {}
        for row in cur.fetchall():
            data = array("f")
            data.frombytes(row["vec"])
            vectors[row["chunk_id"]] = list(data)
        return vectors

    def get_all_vectors(self, model_name: str | None = None) -> tuple[list[list[float]], list[str]]:
        if model_name:
            cur = self.conn.execute(
                "SELECT chunk_id, dim, dtype, vec FROM vectors WHERE model_name = ?",
                (model_name,),
            )
        else:
            cur = self.conn.execute("SELECT chunk_id, dim, dtype, vec FROM vectors")
        rows = cur.fetchall()
        if not rows:
            return [], []
        chunk_ids: list[str] = []
        vectors: list[list[float]] = []
        for row in rows:
            data = array("f")
            data.frombytes(row["vec"])
            vectors.append(list(data))
            chunk_ids.append(row["chunk_id"])
        return vectors, chunk_ids

    def iter_vectors(self, model_name: str | None = None) -> Iterator[tuple[str, list[float]]]:
        if model_name:
            cur = self.conn.execute(
                "SELECT chunk_id, dim, dtype, vec FROM vectors WHERE model_name = ?",
                (model_name,),
            )
        else:
            cur = self.conn.execute("SELECT chunk_id, dim, dtype, vec FROM vectors")
        for row in cur:
            data = array("f")
            data.frombytes(row["vec"])
            yield row["chunk_id"], list(data)

    def get_embedding_cache(self, content_sha: str, model_name: str) -> Optional[list[float]]:
        cur = self.conn.execute(
            "SELECT dim, dtype, vec FROM cache_embeddings WHERE content_sha = ? AND model_name = ?",
            (content_sha, model_name),
        )
        row = cur.fetchone()
        if not row:
            return None
        data = array("f")
        data.frombytes(row["vec"])
        return list(data)

    def upsert_embedding_cache(self, content_sha: str, model_name: str, vector: Sequence[float]) -> None:
        self.conn.execute(
            """
            INSERT INTO cache_embeddings(content_sha, model_name, dim, dtype, vec)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(content_sha, model_name) DO UPDATE SET
                dim=excluded.dim,
                dtype=excluded.dtype,
                vec=excluded.vec
            """,
            (content_sha, model_name, len(vector), "float32", array("f", vector).tobytes()),
        )

    def commit(self) -> None:
        self.conn.commit()

    def vacuum(self) -> None:
        self.conn.execute("VACUUM")

    def stats(self) -> dict[str, object]:
        cur = self.conn.cursor()
        counts = {
            "documents": cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
            "chunks": cur.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            "vectors": cur.execute("SELECT COUNT(*) FROM vectors").fetchone()[0],
        }
        models = [row[0] for row in cur.execute("SELECT DISTINCT model_name FROM vectors").fetchall()]
        return {**counts, "models": models}


def cosine_search(matrix: Sequence[Sequence[float]], query_vec: Sequence[float], top_k: int) -> list[tuple[int, float]]:
    if not matrix:
        return []
    def norm(vec: Sequence[float]) -> float:
        return sum(value * value for value in vec) ** 0.5

    query_norm = norm(query_vec)
    if query_norm == 0:
        return []
    sims: list[float] = []
    for row in matrix:
        row_norm = norm(row)
        if row_norm == 0:
            sims.append(0.0)
            continue
        dot = sum(a * b for a, b in zip(row, query_vec))
        sims.append(dot / (row_norm * query_norm))
    indexed = list(enumerate(sims))
    indexed.sort(key=lambda item: item[1], reverse=True)
    return [(idx, score) for idx, score in indexed[:top_k]]
