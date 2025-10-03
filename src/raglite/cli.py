"""Typer CLI for raglite."""

from __future__ import annotations

import importlib.resources as resources
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import typer

from .api import RagliteAPI
from .config import RagliteConfig

app = typer.Typer(help="Local-first RAG toolkit on SQLite")


def get_api(
    db: Path,
    embed_model: Optional[str] = None,
    *,
    alpha: Optional[float] = None,
) -> RagliteAPI:
    config = RagliteConfig(db_path=db)
    if embed_model:
        config.embed_model = embed_model
    if alpha is not None:
        config.alpha = alpha
    api = RagliteAPI(config)
    api.init_db()
    return api


@app.command()
def init_db(
    db: Path = typer.Option(Path("raglite.db"), help="Database path")  # noqa: B008
) -> None:
    api = get_api(db)
    typer.echo(f"Initialised database at {api.db_path}")


@app.command()
def ingest(
    path: Path = typer.Option(..., exists=True, file_okay=True, dir_okay=True),  # noqa: B008
    db: Path = typer.Option(Path("raglite.db")),  # noqa: B008
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
    db: Path = typer.Option(Path("raglite.db")),  # noqa: B008
    k: int = typer.Option(5),
    alpha: Optional[float] = typer.Option(None),
    rerank: bool = typer.Option(False),
    embed_model: Optional[str] = typer.Option(None),
) -> None:
    api = get_api(db, embed_model, alpha=alpha)
    results = api.query(text, top_k=k, alpha=alpha, rerank=rerank)
    typer.echo(json.dumps([r.__dict__ for r in results], indent=2))


@app.command()
def serve(
    db: Path = typer.Option(Path("raglite.db")),  # noqa: B008
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    embed_model: Optional[str] = typer.Option(None),
) -> None:
    env = dict(os.environ)
    env["RAGLITE_DB"] = str(db)
    if embed_model:
        env["RAGLITE_EMBED_MODEL"] = embed_model
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "raglite.server.app:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        check=True,
        env=env,
    )


@app.command()
def stats(
    db: Path = typer.Option(Path("raglite.db")),  # noqa: B008
    embed_model: Optional[str] = typer.Option(None),
) -> None:
    api = get_api(db, embed_model)
    typer.echo(json.dumps(api.stats(), indent=2))


@app.command("self-test")
def self_test(alpha: float = typer.Option(0.6)) -> None:  # noqa: B008
    """Run an end-to-end smoke test against the bundled demo corpus."""

    demo_path = Path(__file__).resolve().parents[2] / "demo" / "mini_corpus"
    if not demo_path.exists():
        try:
            with resources.as_file(
                resources.files("raglite").joinpath("data/mini_corpus")
            ) as packaged:
                demo_path = packaged
        except FileNotFoundError:
            typer.echo(f"Demo corpus not found at {demo_path}", err=True)
            raise typer.Exit(code=1) from None
    queries: List[str] = [
        "quick start guide",
        "backup schedule",
        "hybrid alpha guidance",
    ]
    with tempfile.TemporaryDirectory(prefix="raglite-selftest-") as tmpdir:
        db_path = Path(tmpdir) / "selftest.db"
        api = get_api(db_path, embed_model="debug", alpha=alpha)
        ingest_result = api.index(demo_path, strategy="fixed")
        stats = api.stats()
        stats["documents_indexed"] = ingest_result.documents
        stats["chunks_indexed"] = ingest_result.chunks
        stats["embeddings_indexed"] = ingest_result.embeddings
        stats["alpha"] = alpha
        typer.echo("raglite self-test")
        typer.echo("================")
        typer.echo(f"Demo corpus: {demo_path}")
        typer.echo(f"Database: {db_path}")
        typer.echo(f"Vector backend: {stats['vector_backend']}")
        typer.echo("")
        typer.echo("Queries:")
        for q in queries:
            typer.echo(f"- {q}")
            results = api.query(q, top_k=3, alpha=alpha)
            if not results:
                typer.echo("  (no results)")
                continue
            for item in results:
                title = item.metadata.get("title", "") if item.metadata else ""
                snippet = item.text.replace("\n", " ")[:160]
                typer.echo(f"  • {title or 'Untitled'} (chunk {item.chunk_id}): {snippet}")
        typer.echo("")
        typer.echo("Stats:")
        typer.echo(json.dumps(stats, indent=2))
        typer.echo("")
        typer.echo(f"Vector backend detected: {stats['vector_backend']}")


@app.command()
def benchmark(db: Path = typer.Option(Path("raglite.db"))) -> None:  # noqa: B008
    subprocess.run([sys.executable, "scripts/bench_basic.py", str(db)], check=True)


@app.command()
def eval(db: Path = typer.Option(Path("raglite.db"))) -> None:  # noqa: B008
    subprocess.run([sys.executable, "scripts/eval_small.py", str(db)], check=True)


if __name__ == "__main__":
    app()
