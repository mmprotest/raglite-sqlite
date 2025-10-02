from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

from .db import Database, cosine_search
from .embeddings.base import EmbeddingBackend
from .typing import SearchResult
from .utils import normalize_text


def bm25_search(
    db: Database,
    query: str,
    k: int,
    filters: dict | None = None,
) -> list[tuple[str, float]]:
    sql = [
        "SELECT c.chunk_id AS chunk_id, bm25(chunk_fts) AS score"
        " FROM chunk_fts JOIN chunks c USING(chunk_id)"
        " JOIN documents d ON d.doc_id = c.doc_id"
        " WHERE chunk_fts MATCH ?"
    ]
    args: List[object] = [query]
    if filters:
        if doc_id := filters.get("doc_id"):
            sql.append(" AND d.doc_id = ?")
            args.append(doc_id)
        if tags := filters.get("tags"):
            sql.append(" AND d.tags LIKE ?")
            args.append(f"%{tags}%")
        if source_prefix := filters.get("source_path"):
            sql.append(" AND d.source_path LIKE ?")
            args.append(f"{source_prefix}%")
    sql.append(" ORDER BY score LIMIT ?")
    args.append(k * 5)
    cur = db.conn.execute("".join(sql), tuple(args))
    results = [(row["chunk_id"], float(row["score"])) for row in cur.fetchall()]
    return results


def vector_search(
    db: Database,
    backend: EmbeddingBackend,
    query: str,
    model_name: str | None,
    k: int,
) -> list[tuple[str, float]]:
    matrix, chunk_ids = db.get_all_vectors(model_name=model_name)
    if not matrix:
        return []
    query_vectors = backend.embed_texts([query], model_name=model_name)
    query_vec = list(query_vectors[0])
    results = cosine_search(matrix, query_vec, top_k=min(len(chunk_ids), max(k * 5, 10)))
    return [(chunk_ids[idx], score) for idx, score in results]


def hybrid_fuse(
    lexical: list[tuple[str, float]],
    semantic: list[tuple[str, float]],
    weight: float,
    k: int,
) -> dict[str, Dict[str, float]]:
    scores: dict[str, Dict[str, float]] = {}
    if lexical:
        bm25_values = [score for _, score in lexical]
        max_bm25 = max(bm25_values)
        min_bm25 = min(bm25_values)
    else:
        max_bm25 = min_bm25 = 0.0
    if semantic:
        vector_values = [score for _, score in semantic]
        max_vec = max(vector_values)
        min_vec = min(vector_values)
    else:
        max_vec = min_vec = 0.0

    def normalize(score: float, lo: float, hi: float) -> float:
        if math.isclose(hi, lo):
            return 0.0
        return (score - lo) / (hi - lo)

    for chunk_id, score in lexical[: k * 5]:
        scores.setdefault(chunk_id, {})["bm25"] = normalize(score, min_bm25, max_bm25)
    for chunk_id, score in semantic[: k * 5]:
        scores.setdefault(chunk_id, {})["vector"] = normalize(score, min_vec, max_vec)

    fused: dict[str, Dict[str, float]] = {}
    for chunk_id, components in scores.items():
        bm25_score = components.get("bm25", 0.0)
        vector_score = components.get("vector", 0.0)
        fused_score = weight * bm25_score + (1 - weight) * vector_score
        fused[chunk_id] = {
            "fused": fused_score,
            "bm25": bm25_score,
            "vector": vector_score,
        }
    return fused


def assemble_results(db: Database, scores: dict[str, Dict[str, float]], k: int, with_snippets: bool = True, max_per_doc: int = 3) -> List[SearchResult]:
    if not scores:
        return []
    placeholders = ",".join("?" for _ in scores)
    cur = db.conn.execute(
        f"SELECT c.*, d.source_path, d.tags FROM chunks c JOIN documents d ON d.doc_id = c.doc_id WHERE c.chunk_id IN ({placeholders})",
        tuple(scores.keys()),
    )
    rows = {row["chunk_id"]: row for row in cur.fetchall()}
    per_doc: dict[str, int] = {}
    sorted_ids = sorted(scores.items(), key=lambda item: item[1]["fused"], reverse=True)
    results: List[SearchResult] = []
    for chunk_id, components in sorted_ids:
        row = rows.get(chunk_id)
        if row is None:
            continue
        doc_id = row["doc_id"]
        count = per_doc.get(doc_id, 0)
        if count >= max_per_doc:
            continue
        per_doc[doc_id] = count + 1
        snippet = row["text"][:200]
        result: SearchResult = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "position": row["position"],
            "text": row["text"],
            "section": row["section"],
            "source_path": row["source_path"],
            "tags": row["tags"],
            "score": components["fused"],
            "bm25_score": components.get("bm25", 0.0),
            "vector_score": components.get("vector", 0.0),
        }
        if with_snippets:
            result["snippet"] = snippet
        results.append(result)
        if len(results) >= k:
            break
    return results
