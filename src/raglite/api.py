"""High level Python API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import RagliteConfig
from .db import apply_migrations, temp_connection
from .ingest import IngestResult, ingest_path
from .search import SearchResult, hybrid_search
from .vector import detect_backend


@dataclass
class RagliteAPI:
    config: RagliteConfig

    @property
    def db_path(self) -> Path:
        return self.config.db_path

    def init_db(self) -> None:
        with temp_connection(self.db_path) as conn:
            apply_migrations(conn)

    def index(
        self, corpus_path: Path, *, strategy: str = "recursive", ocr: bool = False
    ) -> IngestResult:
        return ingest_path(
            self.db_path, corpus_path, config=self.config, strategy=strategy, ocr=ocr
        )

    def query(
        self,
        text: str,
        *,
        top_k: int = 10,
        alpha: Optional[float] = None,
        rerank: bool = False,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[SearchResult]:
        with temp_connection(self.db_path) as conn:
            return hybrid_search(
                conn,
                text,
                alpha=alpha if alpha is not None else self.config.alpha,
                top_k=top_k,
                embed_model=self.config.embed_model,
                rerank=rerank,
                tags=tags,
            )

    def add_tags(self, document_id: int, tags: Dict[str, str]) -> None:
        with temp_connection(self.db_path) as conn:
            conn.execute(
                """
                UPDATE chunks
                SET tags_json = json_patch(COALESCE(tags_json, '{}'), ?)
                WHERE document_id = ?
                """,
                (json.dumps(tags), document_id),
            )

    def stats(self) -> Dict[str, Any]:
        with temp_connection(self.db_path) as conn:
            doc_count = int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
            chunk_count = int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            embed_count = int(conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
            dim_row = conn.execute(
                "SELECT dim, model FROM embeddings ORDER BY id DESC LIMIT 1"
            ).fetchone()
            fts_exists = (
                conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_fts'"
                ).fetchone()
                is not None
            )
            backend = detect_backend(conn)
        dim = int(dim_row[0]) if dim_row else 0
        model = str(dim_row[1]) if dim_row else self.config.embed_model
        return {
            "documents": doc_count,
            "chunks": chunk_count,
            "embeddings": embed_count,
            "vector_backend": backend.name,
            "embedding_model": model,
            "embedding_dim": dim,
            "fts_enabled": fts_exists,
            "alpha": self.config.alpha,
        }


def init_db(db_path: Path | str) -> None:
    RagliteAPI(RagliteConfig(Path(db_path))).init_db()


def index_corpus(
    db_path: Path | str,
    corpus_path: Path | str,
    *,
    strategy: str = "recursive",
    ocr: bool = False,
    embed_model: Optional[str] = None,
) -> IngestResult:
    config = RagliteConfig(Path(db_path))
    if embed_model:
        config.embed_model = embed_model
    api = RagliteAPI(config)
    api.init_db()
    return api.index(Path(corpus_path), strategy=strategy, ocr=ocr)


def query(
    db_path: Path | str,
    text: str,
    *,
    top_k: int = 10,
    alpha: Optional[float] = None,
    rerank: bool = False,
    tags: Optional[Dict[str, str]] = None,
    embed_model: Optional[str] = None,
) -> List[SearchResult]:
    config = RagliteConfig(Path(db_path))
    if embed_model:
        config.embed_model = embed_model
    if alpha is not None:
        config.alpha = alpha
    api = RagliteAPI(config)
    return api.query(text, top_k=top_k, alpha=alpha, rerank=rerank, tags=tags)


def add_tags(db_path: Path | str, document_id: int, tags: Dict[str, str]) -> None:
    api = RagliteAPI(RagliteConfig(Path(db_path)))
    api.add_tags(document_id, tags)


def stats(db_path: Path | str) -> Dict[str, Any]:
    api = RagliteAPI(RagliteConfig(Path(db_path)))
    return api.stats()
