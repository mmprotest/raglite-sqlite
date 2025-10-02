from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("typer")

from typer.testing import CliRunner

from raglite_sqlite.cli import app
from raglite_sqlite.embeddings.base import DummyBackend


def test_cli_flow(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    db_path = tmp_path / "knowledge.db"

    # init
    result = runner.invoke(app, ["init", "--db", str(db_path)])
    assert result.exit_code == 0

    # monkeypatch backend to avoid heavy model
    dummy = DummyBackend(dim=16)

    monkeypatch.setattr("raglite_sqlite.cli.get_backend", lambda model: dummy)

    data_dir = Path(__file__).parent / "data"
    result = runner.invoke(
        app,
        [
            "index",
            str(data_dir),
            "--db",
            str(db_path),
        ],
    )
    assert result.exit_code == 0

    result = runner.invoke(app, ["query", "sample", "--db", str(db_path)])
    assert result.exit_code == 0

    result = runner.invoke(app, ["stats", "--db", str(db_path)])
    assert result.exit_code == 0
