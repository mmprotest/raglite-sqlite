"""Evaluate raglite on a small synthetic dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from raglite.api import RagliteAPI, RagliteConfig

DATA = [
    ("doc1", "Neural retrieval quick start guide"),
    ("doc2", "SQLite hybrid search tutorial"),
    ("doc3", "Benchmarking local rag systems"),
]

QUERIES = {
    "quickstart": "quick start guide",
    "sqlite": "sqlite hybrid",
    "benchmark": "benchmark rag",
}


def main(db_path: str | None = None) -> None:
    db = Path(db_path or "eval.db")
    if db.exists():
        db.unlink()
    api = RagliteAPI(RagliteConfig(db))
    api.init_db()
    corpus_dir = db.parent / "eval_corpus"
    corpus_dir.mkdir(exist_ok=True)
    for name, text in DATA:
        (corpus_dir / f"{name}.txt").write_text(text, encoding="utf-8")
    api.index(corpus_dir, strategy="fixed")
    results = {}
    for name, query in QUERIES.items():
        hits = api.query(query, top_k=3)
        results[name] = [hit.chunk_id for hit in hits]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
