"""Vector backend protocol and factory."""

from __future__ import annotations

import sqlite3
from typing import Iterable, List, Optional, Protocol

from array import array

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
    ) -> List[Candidate]:
        ...


def get_backend(conn: sqlite3.Connection) -> Backend:
    backend = SQLiteExtensionBackend.create(conn)
    if backend is not None:
        return backend
    return PythonFallbackBackend()
