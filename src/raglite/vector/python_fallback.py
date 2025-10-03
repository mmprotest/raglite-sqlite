"""Python fallback vector backend."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Iterable, List, Optional

from ..embed import embedding_from_bytes
from .types import Candidate


@dataclass
class PythonFallbackBackend:
    name: str = "python"

    def search(
        self,
        conn: sqlite3.Connection,
        query_vector,
        *,
        top_n: int,
        prefilter_ids: Optional[Iterable[int]] = None,
    ) -> List[Candidate]:
        ids = list(prefilter_ids) if prefilter_ids is not None else self._all_chunk_ids(conn)
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        cur = conn.execute(
            f"SELECT chunk_id, embedding FROM embeddings WHERE chunk_id IN ({placeholders})",
            ids,
        )
        query_vec = query_vector
        query_norm = self._norm(query_vec)
        if not query_norm:
            return []
        scored: List[Candidate] = []
        for row in cur.fetchall():
            vector = embedding_from_bytes(row[1])
            norm = self._norm(vector)
            if not norm:
                continue
            score = float(self._dot(query_vec, vector) / (query_norm * norm))
            scored.append(Candidate(int(row[0]), score))
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_n]

    def _all_chunk_ids(self, conn: sqlite3.Connection) -> List[int]:
        cur = conn.execute("SELECT DISTINCT chunk_id FROM embeddings")
        return [int(row[0]) for row in cur.fetchall()]

    def _dot(self, a, b) -> float:
        length = min(len(a), len(b))
        return float(sum(a[i] * b[i] for i in range(length)))

    def _norm(self, vec) -> float:
        return float(sum(v * v for v in vec) ** 0.5)
