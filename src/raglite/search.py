"""Hybrid search implementation."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional

from .config import clamp_alpha
from .embed import embedding_from_bytes, get_embedding_store
from .vector import detect_backend


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
    normalized = _normalize_fts_query(query)
    cur = conn.execute(
        """
        SELECT rowid, bm25(chunk_fts) AS score
        FROM chunk_fts
        WHERE chunk_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (normalized, k),
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

    vector_backend = detect_backend(conn)
    embedding_store = get_embedding_store(embed_model)
    query_vec = embedding_from_bytes(embedding_store.embed_many([query])[0])
    query_norm = _norm(query_vec)
    vector_results = vector_backend.search(
        conn,
        query_vec,
        top_n=max(top_k, len(candidates)) or top_k,
        prefilter_ids=[c.chunk_id for c in candidates] if candidates else None,
    )
    if len(vector_results) < top_k and vector_backend.available:
        extra = vector_backend.search(conn, query_vec, top_n=top_k, prefilter_ids=None)
        seen_ids = {c.chunk_id for c in vector_results}
        for candidate in extra:
            if candidate.chunk_id in seen_ids:
                continue
            vector_results.append(candidate)
            seen_ids.add(candidate.chunk_id)
            if len(vector_results) >= top_k:
                break
    vector_scores = {c.chunk_id: c.score for c in vector_results}

    combined: List[SearchResult] = []
    ordered_ids = [c.chunk_id for c in candidates]
    for candidate in vector_results:
        if candidate.chunk_id not in ordered_ids:
            ordered_ids.append(candidate.chunk_id)
    seen = set()
    for chunk_id in ordered_ids:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        bm_score = bm25_norm.get(chunk_id, 0.0)
        vec_score = vector_scores.get(chunk_id, 0.0)
        score = alpha * bm_score + (1 - alpha) * vec_score
        if bm_score == 0.0 and vec_score > 0.0:
            score += 0.05
        chunk_row = conn.execute(
            """
            SELECT c.id, c.document_id, c.text, c.tags_json, d.meta_json, d.title
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id = ?
            """,
            (chunk_id,),
        ).fetchone()
        if not chunk_row:
            continue
        tags_json = json.loads(chunk_row[3] or "{}")
        if tags and not _tags_match(tags, tags_json):
            continue
        metadata = json.loads(chunk_row[4] or "{}")
        metadata.setdefault("title", chunk_row[5] or "")
        combined.append(
            SearchResult(
                chunk_id=int(chunk_row[0]),
                document_id=int(chunk_row[1]),
                score=score,
                text=str(chunk_row[2]),
                metadata=metadata | {"tags": tags_json},
            )
        )
    if rerank and combined:
        rerank_limit = min(len(combined), max(top_k * 2, 20))
        rerank_ids = [item.chunk_id for item in combined[:rerank_limit]]
        if rerank_ids:
            placeholders = ",".join("?" for _ in rerank_ids)
            cur = conn.execute(
                f"SELECT chunk_id, embedding FROM embeddings WHERE chunk_id IN ({placeholders})",
                rerank_ids,
            )
            embed_map = {int(row[0]): embedding_from_bytes(row[1]) for row in cur.fetchall()}
            for item in combined:
                chunk_vec = embed_map.get(item.chunk_id)
                if chunk_vec is None:
                    continue
                rerank_score = _cosine_similarity(query_vec, chunk_vec, query_norm)
                item.score = (item.score + rerank_score) / 2
    combined.sort(key=lambda item: item.score, reverse=True)
    return combined[:top_k]


def _tags_match(required: Dict[str, str], existing: Dict[str, str]) -> bool:
    for key, value in required.items():
        if existing.get(key) != value:
            return False
    return True


def _normalize_fts_query(query: str) -> str:
    return re.sub(r"[\-\"']", " ", query)


def _norm(vec) -> float:
    return float(math.sqrt(sum(component * component for component in vec)))


def _cosine_similarity(query_vec, chunk_vec, query_norm: float) -> float:
    chunk_norm = _norm(chunk_vec)
    if not query_norm or not chunk_norm:
        return 0.0
    length = min(len(query_vec), len(chunk_vec))
    dot = sum(query_vec[i] * chunk_vec[i] for i in range(length))
    return float(dot / (query_norm * chunk_norm))
