from __future__ import annotations

from threading import RLock
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .api import RagLite
from .embeddings.base import EmbeddingBackend
from .rerank import Reranker, get_reranker


def _default_backend() -> EmbeddingBackend:
    from .embeddings.hash_backend import HashingBackend

    return HashingBackend()


class QueryPayload(BaseModel):
    query: str
    k: int = 8
    hybrid_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    max_per_doc: int = 3
    filters: Optional[Dict[str, str]] = None
    model_name: Optional[str] = None
    with_snippets: bool = True
    use_semantic: bool = True
    reranker: Optional[str] = None
    reranker_options: Optional[Dict[str, Any]] = None


class IndexPayload(BaseModel):
    paths: list[str]
    tags: Optional[str] = None
    parser_opts: Optional[Dict[str, Any]] = None
    chunker: str = "recursive"
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    model_name: Optional[str] = None
    skip_unchanged: bool = True
    recurse: bool = True
    glob: Optional[str] = None


class DeletePayload(BaseModel):
    doc_id: str


def create_app(
    db_path: str,
    *,
    embedding_backend: EmbeddingBackend | None = None,
    rerankers: dict[str, Reranker] | None = None,
) -> FastAPI:
    """Create a FastAPI application exposing the RagLite API."""

    rag = RagLite(db_path)
    backend = embedding_backend or _default_backend()
    reranker_cache = rerankers or {}
    lock = RLock()

    app = FastAPI(title="RagLite SQLite", version="1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/stats")
    def stats() -> dict[str, Any]:
        with lock:
            return rag.stats()

    @app.post("/query")
    def query(payload: QueryPayload) -> dict[str, Any]:
        with lock:
            resolved_backend = backend if payload.use_semantic else None
            reranker_instance: Reranker | None = None
            if payload.reranker:
                reranker_instance = reranker_cache.get(payload.reranker)
                if reranker_instance is None:
                    try:
                        reranker_instance = get_reranker(
                            payload.reranker, **(payload.reranker_options or {})
                        )
                    except KeyError as exc:
                        raise HTTPException(status_code=400, detail=str(exc)) from exc
                    reranker_cache[payload.reranker] = reranker_instance
            results = rag.search(
                payload.query,
                k=payload.k,
                hybrid_weight=payload.hybrid_weight,
                filters=payload.filters,
                model_name=payload.model_name,
                embedding_backend=resolved_backend,
                max_per_doc=payload.max_per_doc,
                with_snippets=payload.with_snippets,
                reranker=reranker_instance,
            )
            return {"results": results}

    @app.post("/index")
    def index(payload: IndexPayload) -> dict[str, Any]:
        with lock:
            stats = rag.index(
                payload.paths,
                tags=payload.tags,
                parser_opts=payload.parser_opts,
                chunker=payload.chunker,
                chunk_size_tokens=payload.chunk_size_tokens,
                chunk_overlap_tokens=payload.chunk_overlap_tokens,
                embedding_backend=backend,
                model_name=payload.model_name,
                skip_unchanged=payload.skip_unchanged,
                recurse=payload.recurse,
                glob=payload.glob,
            )
            return stats

    @app.post("/delete")
    def delete(payload: DeletePayload) -> dict[str, str]:
        with lock:
            rag.delete(payload.doc_id)
        return {"status": "deleted", "doc_id": payload.doc_id}

    @app.on_event("shutdown")
    def shutdown() -> None:
        rag.close()

    return app
