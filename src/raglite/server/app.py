"""FastAPI application for raglite."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except Exception as exc:  # pragma: no cover - optional dependency
    raise RuntimeError("Install raglite[server] to use the FastAPI app") from exc

from ..api import RagliteAPI
from ..config import RagliteConfig


class QueryRequest(BaseModel):
    text: str
    k: int = 10
    alpha: Optional[float] = None
    rerank: bool = False
    tags: Optional[Dict[str, str]] = None


class IngestRequest(BaseModel):
    path: str
    strategy: str = "recursive"
    ocr: bool = False


def create_app(db_path: str | Path) -> FastAPI:
    config = RagliteConfig(Path(db_path))
    embed_override = os.getenv("RAGLITE_EMBED_MODEL")
    if embed_override:
        config.embed_model = embed_override
    api = RagliteAPI(config)
    api.init_db()

    app = FastAPI(title="raglite", version="0.2.0")

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/stats")
    def stats() -> Dict[str, int]:
        return api.stats()

    @app.post("/query")
    def query(request: QueryRequest):
        results = api.query(
            request.text,
            top_k=request.k,
            alpha=request.alpha,
            rerank=request.rerank,
            tags=request.tags,
        )
        return {
            "results": [
                {
                    "chunk_id": r.chunk_id,
                    "document_id": r.document_id,
                    "score": r.score,
                    "text": r.text,
                    "metadata": r.metadata,
                }
                for r in results
            ]
        }

    @app.post("/ingest")
    def ingest(request: IngestRequest):
        corpus_path = Path(request.path)
        if not corpus_path.exists():
            raise HTTPException(status_code=404, detail="Path not found")
        result = api.index(corpus_path, strategy=request.strategy, ocr=request.ocr)
        return {
            "documents": result.documents,
            "chunks": result.chunks,
            "embeddings": result.embeddings,
        }

    return app


app = create_app(Path("raglite.db"))
