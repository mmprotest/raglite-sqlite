"""Vector backend protocol and detection helpers."""

from __future__ import annotations

import sqlite3
from array import array
from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol

from .python_fallback import PythonFallbackBackend
from .sqlite_ext import SQLiteExtensionBackend
from .types import Candidate


class Backend(Protocol):
    name: str

    def search(
        self,
        conn: sqlite3.Connection,
        query_vector: array,
        *,
        top_n: int,
        prefilter_ids: Optional[Iterable[int]] = None,
    ) -> List[Candidate]: ...


@dataclass(frozen=True)
class VectorBackend:
    """Wrapper that records the active backend."""

    name: str
    backend: Optional[Backend]

    def search(
        self,
        conn: sqlite3.Connection,
        query_vector: array,
        *,
        top_n: int,
        prefilter_ids: Optional[Iterable[int]] = None,
    ) -> List[Candidate]:
        if self.backend is None:
            return []
        return self.backend.search(
            conn,
            query_vector,
            top_n=top_n,
            prefilter_ids=prefilter_ids,
        )

    @property
    def available(self) -> bool:
        return self.backend is not None


def detect_backend(conn: sqlite3.Connection) -> VectorBackend:
    """Detect the best available vector backend."""

    if not _has_embeddings_table(conn):
        return VectorBackend(name="none", backend=None)
    backend = SQLiteExtensionBackend.create(conn)
    if backend is not None:
        return VectorBackend(name=backend.name, backend=backend)
    return VectorBackend(name="python-fallback", backend=PythonFallbackBackend())


def get_backend(conn: sqlite3.Connection) -> Backend:
    """Backward compatible helper returning a concrete backend instance."""

    detected = detect_backend(conn)
    if detected.backend is None:
        return PythonFallbackBackend()
    return detected.backend


def _has_embeddings_table(conn: sqlite3.Connection) -> bool:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'")
    return cur.fetchone() is not None
