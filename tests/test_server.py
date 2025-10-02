from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="FastAPI is required for server tests")
from fastapi.testclient import TestClient

from raglite_sqlite.server import create_app


def test_server_query_endpoint(temp_db: tuple[Path, object, object]) -> None:
    db_path, rag, backend = temp_db
    data_dir = Path(__file__).parent / "data"
    rag.index([str(data_dir)], embedding_backend=backend)

    app = create_app(str(db_path), embedding_backend=backend)
    client = TestClient(app)
    response = client.post("/query", json={"query": "sample", "use_semantic": False})
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert data["results"], "Expected at least one result from the REST API"
