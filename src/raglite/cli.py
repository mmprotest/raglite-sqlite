"""Typer CLI for raglite."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from .api import RagliteAPI
from .config import RagliteConfig

app = typer.Typer(help="Local-first RAG toolkit on SQLite")


def get_api(db: Path, embed_model: Optional[str] = None) -> RagliteAPI:
    config = RagliteConfig(db_path=db)
    if embed_model:
        config.embed_model = embed_model
    api = RagliteAPI(config)
    api.init_db()
    return api


@app.command()
def init_db(db: Path = typer.Option(Path("raglite.db"), help="Database path")) -> None:
    api = get_api(db)
    typer.echo(f"Initialised database at {api.db_path}")


@app.command()
def ingest(
    path: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=True),
    db: Path = typer.Option(Path("raglite.db")),
    strategy: str = typer.Option("recursive"),
    embed_model: Optional[str] = typer.Option(None),
    ocr: bool = typer.Option(False, help="Enable OCR for PDFs"),
) -> None:
    api = get_api(db, embed_model)
    result = api.index(path, strategy=strategy, ocr=ocr)
    typer.echo(json.dumps(result.__dict__, indent=2))


@app.command()
def query(
    text: str,
    db: Path = typer.Option(Path("raglite.db")),
    k: int = typer.Option(5),
    alpha: Optional[float] = typer.Option(None),
    rerank: bool = typer.Option(False),
) -> None:
    api = get_api(db)
    results = api.query(text, top_k=k, alpha=alpha, rerank=rerank)
    typer.echo(json.dumps([r.__dict__ for r in results], indent=2))


@app.command()
def serve(
    db: Path = typer.Option(Path("raglite.db")),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
) -> None:
    env = dict(os.environ)
    env["RAGLITE_DB"] = str(db)
    subprocess.run([sys.executable, "-m", "uvicorn", "raglite.server.app:app", "--host", host, "--port", str(port)], check=True, env=env)


@app.command()
def stats(db: Path = typer.Option(Path("raglite.db"))) -> None:
    api = get_api(db)
    typer.echo(json.dumps(api.stats(), indent=2))


@app.command("self-test")
def self_test() -> None:
    temp_db = Path("selftest.db")
    if temp_db.exists():
        temp_db.unlink()
    api = get_api(temp_db, embed_model="debug")
    demo_path = Path(__file__).resolve().parents[1] / "demo" / "mini_corpus"
    api.index(demo_path, strategy="fixed")
    results = api.query("quick start guide", top_k=3)
    typer.echo(json.dumps({"backend": "python", "results": [r.text for r in results]}, indent=2))


@app.command()
def benchmark(db: Path = typer.Option(Path("raglite.db"))) -> None:
    subprocess.run([sys.executable, "scripts/bench_basic.py", str(db)], check=True)


@app.command()
def eval(db: Path = typer.Option(Path("raglite.db"))) -> None:
    subprocess.run([sys.executable, "scripts/eval_beir.py", str(db)], check=True)


if __name__ == "__main__":
    app()
