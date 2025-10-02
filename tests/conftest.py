from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raglite_sqlite.api import RagLite
from raglite_sqlite.embeddings.base import DummyBackend


@pytest.fixture()
def temp_db(tmp_path: Path) -> Iterator[tuple[Path, RagLite, DummyBackend]]:
    db_path = tmp_path / "knowledge.db"
    rag = RagLite(str(db_path))
    backend = DummyBackend(dim=16)
    yield db_path, rag, backend
    rag.close()
