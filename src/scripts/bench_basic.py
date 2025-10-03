"""Basic benchmarking harness for raglite."""

from __future__ import annotations

import json
import time
from pathlib import Path

from raglite.api import RagliteAPI, RagliteConfig


def main(db_path: str | None = None) -> None:
    db = Path(db_path or "bench.db")
    if db.exists():
        db.unlink()
    api = RagliteAPI(RagliteConfig(db))
    api.init_db()
    corpus_dir = db.parent / "bench_corpus"
    corpus_dir.mkdir(exist_ok=True)
    doc = "Raglite benchmarking quick start guide with SQLite hybrid search."
    for i in range(200):
        (corpus_dir / f"doc_{i}.txt").write_text(f"{doc} repetition {i}", encoding="utf-8")
    start = time.time()
    api.index(corpus_dir, strategy="fixed")
    ingest_time = time.time() - start
    query_latencies = []
    for _ in range(20):
        q_start = time.time()
        api.query("quick start guide", top_k=3)
        query_latencies.append(time.time() - q_start)
    stats = {
        "ingest_seconds": ingest_time,
        "query_latency_p50": percentile(query_latencies, 50),
        "query_latency_p95": percentile(query_latencies, 95),
    }
    print(json.dumps(stats, indent=2))


def percentile(values, percent):
    values = sorted(values)
    k = (len(values) - 1) * percent / 100
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[int(k)]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
