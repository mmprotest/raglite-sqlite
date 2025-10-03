"""Hybrid search implementation."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import clamp_alpha
from .embed import embedding_from_bytes, get_embedding_store
from .vector import get_backend


@dataclass
class SearchResult:
    chunk_id: int
    document_id: int
    score: float
    text: str
    metadata: Dict[str, str]


@dataclass
class RankedChunk:
    chunk_id: int
    score: float


def bm25(conn: sqlite3.Connection, query: str, *, k: int = 200) -> List[RankedChunk]:
    cur = conn.execute(
        "SELECT rowid, bm25(chunk_fts) as score FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY score LIMIT ?",
        (query, k),
    )
    return [RankedChunk(int(row[0]), float(row[1])) for row in cur.fetchall()]


def normalize_scores(scores: List[RankedChunk]) -> Dict[int, float]:
    if not scores:
        return {}
    values = [c.score for c in scores]
    max_score = max(values)
    min_score = min(values)
    if max_score == min_score:
        return {c.chunk_id: 1.0 for c in scores}
    return {c.chunk_id: (max_score - c.score) / (max_score - min_score) for c in scores}


def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    alpha: float = 0.6,
    top_k: int = 10,
    embed_model: str,
    rerank: bool = False,
    tags: Optional[Dict[str, str]] = None,
) -> List[SearchResult]:
    alpha = clamp_alpha(alpha)
    candidates = bm25(conn, query)
    bm25_norm = normalize_scores(candidates)

    backend = get_backend(conn)
    embedding_store = get_embedding_store(embed_model)
    query_vec = embedding_from_bytes(embedding_store.embed_many([query])[0])
    vector_results = backend.search(
        conn,
        query_vec,
        top_n=max(top_k, len(candidates)) or top_k,
        prefilter_ids=[c.chunk_id for c in candidates] if candidates else None,
    )
    vector_scores = {c.chunk_id: c.score for c in vector_results}

    combined: List[SearchResult] = []
    ordered_ids = [c.chunk_id for c in candidates] or [c.chunk_id for c in vector_results]
    seen = set()
    for chunk_id in ordered_ids:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        bm_score = bm25_norm.get(chunk_id, 0.0)
        vec_score = vector_scores.get(chunk_id, 0.0)
        score = alpha * bm_score + (1 - alpha) * vec_score
        chunk_row = conn.execute(
            "SELECT c.id, c.document_id, c.text, c.tags_json, d.meta_json FROM chunks c JOIN documents d ON d.id = c.document_id WHERE c.id = ?",
            (chunk_id,),
        ).fetchone()
        if not chunk_row:
            continue
        tags_json = json.loads(chunk_row[3] or "{}")
        if tags and not _tags_match(tags, tags_json):
            continue
        metadata = json.loads(chunk_row[4] or "{}")
        combined.append(
            SearchResult(
                chunk_id=int(chunk_row[0]),
                document_id=int(chunk_row[1]),
                score=score,
                text=str(chunk_row[2]),
                metadata=metadata | {"tags": tags_json},
            )
        )
    combined.sort(key=lambda item: item.score, reverse=True)
    return combined[:top_k]


def _tags_match(required: Dict[str, str], existing: Dict[str, str]) -> bool:
    for key, value in required.items():
        if existing.get(key) != value:
            return False
    return True
