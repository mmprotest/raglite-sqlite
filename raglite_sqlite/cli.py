from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .api import RagLite
app = typer.Typer(help="RagLite SQLite CLI")
console = Console()


def get_rag(db: Path) -> RagLite:
    return RagLite(str(db))


def get_backend(backend: str, model: Optional[str]):
    backend_name = (backend or "hash").lower()
    if backend_name == "hash":
        from .embeddings.hash_backend import HashingBackend

        return HashingBackend()
    if backend_name in {"sentence-transformers", "st"}:
        try:
            from .embeddings.sentence_transformers_backend import SentenceTransformersBackend
        except ImportError as exc:  # pragma: no cover - optional dependency missing
            raise typer.BadParameter(
                "Sentence Transformers backend requires raglite-sqlite[embeddings]"
            ) from exc

        return SentenceTransformersBackend(
            model_name=model or "sentence-transformers/all-MiniLM-L6-v2"
        )
    raise typer.BadParameter(f"Unknown embedding backend '{backend}'")


@app.command()
def init(db: Path = typer.Option(..., help="Database path")) -> None:
    rag = get_rag(db)
    rag.close()
    console.print(f"Initialized database at [bold]{db}[/bold]")


@app.command()
def index(
    path: Path = typer.Argument(..., exists=True, file_okay=True, dir_okay=True),
    db: Path = typer.Option(..., help="Database path"),
    tags: Optional[str] = typer.Option(None, help="Comma-separated tags"),
    backend: str = typer.Option("hash", help="Embedding backend to use (hash or sentence-transformers)"),
    model: Optional[str] = typer.Option(None, help="Embedding model name"),
    chunk_size: int = typer.Option(512, help="Chunk size in tokens"),
    overlap: int = typer.Option(64, help="Chunk overlap"),
    glob: Optional[str] = typer.Option(None, help="Glob pattern"),
    recursive: bool = typer.Option(True, help="Recurse into directories"),
    skip_unchanged: bool = typer.Option(True, help="Skip unchanged files"),
) -> None:
    backend_impl = get_backend(backend, model)
    rag = get_rag(db)
    result = rag.index(
        [str(path)],
        tags=tags,
        chunk_size_tokens=chunk_size,
        chunk_overlap_tokens=overlap,
        embedding_backend=backend_impl,
        model_name=model,
        glob=glob,
        recurse=recursive,
        skip_unchanged=skip_unchanged,
    )
    console.print(
        f"Indexed {result['files']} files, {result['chunks']} chunks (skipped {result['skipped']} unchanged)"
    )


@app.command()
def query(
    text: str = typer.Argument(..., help="Query text"),
    db: Path = typer.Option(..., help="Database path"),
    k: int = typer.Option(8, help="Number of results"),
    hybrid: float = typer.Option(0.6, min=0.0, max=1.0, help="Hybrid weight"),
    max_per_doc: int = typer.Option(3, help="Max results per document"),
    filters: Optional[list[str]] = typer.Option(None, "--filter", help="Filter key=value"),
    reranker: Optional[str] = typer.Option(None, help="Name of reranker to apply"),
    reranker_option: Optional[list[str]] = typer.Option(
        None, "--reranker-option", help="Reranker option key=value"
    ),
) -> None:
    rag = get_rag(db)
    filter_dict: dict[str, str] | None = None
    if filters:
        filter_dict = {}
        for item in filters:
            if "=" not in item:
                raise typer.BadParameter("Filters must be in key=value format")
            key, value = item.split("=", 1)
            filter_dict[key] = value
    reranker_opts: dict[str, object] | None = None
    if reranker_option:
        reranker_opts = {}
        for item in reranker_option:
            if "=" not in item:
                raise typer.BadParameter("Reranker options must be in key=value format")
            key, value = item.split("=", 1)
            reranker_opts[key] = value
    try:
        results = rag.search(
            text,
            k=k,
            hybrid_weight=hybrid,
            max_per_doc=max_per_doc,
            filters=filter_dict,
            reranker=reranker,
            reranker_options=reranker_opts,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Score", justify="right")
    table.add_column("Doc ID")
    table.add_column("Section")
    table.add_column("Snippet")
    for item in results:
        table.add_row(
            f"{item['score']:.3f}",
            item.get("doc_id", ""),
            item.get("section") or "",
            (item.get("snippet") or item.get("text", ""))[:120],
        )
    console.print(table)


@app.command()
def stats(db: Path = typer.Option(..., help="Database path")) -> None:
    rag = get_rag(db)
    info = rag.stats()
    console.print(json.dumps(info, indent=2))


@app.command()
def export(
    db: Path = typer.Option(..., help="Database path"),
    to: Path = typer.Option(..., help="Destination NDJSON"),
    include_vectors: bool = typer.Option(
        False, "--include-vectors/--no-include-vectors", help="Include vector blobs"
    ),
) -> None:
    rag = get_rag(db)
    rag.export(str(to), include_vectors=include_vectors)
    console.print(f"Exported data to {to}")


@app.command()
def vacuum(db: Path = typer.Option(..., help="Database path")) -> None:
    rag = get_rag(db)
    rag.vacuum()
    console.print("VACUUM completed")


@app.command()
def serve(
    db: Path = typer.Option(..., help="Database path"),
    host: str = typer.Option("127.0.0.1", help="Host to bind"),
    port: int = typer.Option(8000, help="Port to bind"),
    reload: bool = typer.Option(False, help="Enable auto-reload"),
) -> None:
    """Run the optional REST server."""

    try:
        from .server import create_app
    except ImportError as exc:  # pragma: no cover - optional dependency missing
        raise typer.BadParameter(
            "The REST server requires FastAPI. Install raglite-sqlite[server]."
        ) from exc
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional dependency missing
        raise typer.BadParameter(
            "Running the REST server requires uvicorn. Install raglite-sqlite[server]."
        ) from exc

    app_instance = create_app(str(db))
    uvicorn.run(app_instance, host=host, port=port, reload=reload)
