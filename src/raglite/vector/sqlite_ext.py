"""Backend powered by sqlite extensions."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from .python_fallback import PythonFallbackBackend
from .types import Candidate


EXTENSION_NAMES = ["sqlite_vec", "vec0", "sqlite_vss"]


@dataclass
class SQLiteExtensionBackend:
    name: str = "sqlite-extension"
    _fallback: PythonFallbackBackend = field(default_factory=PythonFallbackBackend)

    @classmethod
    def create(cls, conn: sqlite3.Connection) -> Optional["SQLiteExtensionBackend"]:
        if not hasattr(conn, "enable_load_extension"):
            return None
        for ext in EXTENSION_NAMES:
            try:
                conn.enable_load_extension(True)
                conn.load_extension(ext)
                conn.enable_load_extension(False)
                return cls()
            except sqlite3.OperationalError:
                continue
            finally:
                if hasattr(conn, "enable_load_extension"):
                    try:
                        conn.enable_load_extension(False)
                    except sqlite3.OperationalError:
                        pass
        return None

    def search(
        self,
        conn: sqlite3.Connection,
        query_vector,
        *,
        top_n: int,
        prefilter_ids: Optional[Iterable[int]] = None,
    ) -> List[Candidate]:
        try:
            cur = conn.execute(
                "SELECT chunk_id, score FROM embedding_search(?, ?) ORDER BY score DESC LIMIT ?",
                (getattr(query_vector, "tobytes", lambda: bytes(query_vector))(), len(query_vector), top_n),
            )
            rows = cur.fetchall()
            if rows:
                return [Candidate(int(row[0]), float(row[1])) for row in rows]
        except sqlite3.OperationalError:
            pass
        return self._fallback.search(
            conn,
            query_vector,
            top_n=top_n,
            prefilter_ids=prefilter_ids,
        )
