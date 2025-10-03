"""SQLite database helpers for raglite."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


PRAGMAS = {
    "journal_mode": "WAL",
    "synchronous": "NORMAL",
    "temp_store": "MEMORY",
    "mmap_size": 268435456,
}


class RagliteDatabaseError(RuntimeError):
    """Raised for database specific errors."""


def connect(db_path: Path | str, *, read_only: bool = False) -> sqlite3.Connection:
    path = Path(db_path)
    if read_only:
        uri = f"file:{path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
    else:
        conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn)
    return conn


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    for pragma, value in PRAGMAS.items():
        conn.execute(f"PRAGMA {pragma}={value}")


def apply_migrations(conn: sqlite3.Connection, schema_path: Optional[Path] = None) -> None:
    path = schema_path or SCHEMA_PATH
    sql = path.read_text(encoding="utf-8")
    with conn:
        conn.executescript(sql)


@contextmanager
def temp_connection(db_path: Path | str) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        yield conn
    finally:
        conn.close()
