#!/usr/bin/env python3
"""Basic offline benchmark for Raglite using the demo corpus."""

from __future__ import annotations

import argparse
import importlib.resources as resources
import math
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path and SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

from raglite.api import RagliteAPI, RagliteConfig  # noqa: E402

QUERIES = [
    "quick start guide",
    "backup schedule",
    "enable wal",
    "tune hybrid alpha",
    "air-gapped embeddings",
    "vector cache",
    "local-first raglite",
    "nightly exports",
    "query builder",
    "sync service",
]


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    lower_value = ordered[lower]
    upper_value = ordered[upper]
    fraction = position - lower
    return lower_value + (upper_value - lower_value) * fraction


def demo_corpus_dir() -> Path:
    fs_path = Path(__file__).resolve().parents[1] / "demo" / "mini_corpus"
    if fs_path.exists():
        return fs_path
    with resources.as_file(resources.files("raglite").joinpath("data/mini_corpus")) as packaged:
        return packaged


def clone_corpus(base_dir: Path, target_docs: int) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="raglite-bench-corpus-"))
    files = list(base_dir.iterdir())
    if not files:
        raise RuntimeError(f"No files in {base_dir}")
    written = 0
    copy_idx = 0
    while written < target_docs:
        for src in files:
            text = src.read_text(encoding="utf-8", errors="ignore")
            dest = temp_dir / f"{src.stem}_{copy_idx}{src.suffix or '.txt'}"
            dest.write_text(text, encoding="utf-8")
            written += 1
            if written >= target_docs:
                break
        copy_idx += 1
    return temp_dir


def run_queries(api: RagliteAPI, queries: Iterable[str], *, alpha: float) -> List[float]:
    latencies: List[float] = []
    for query in queries:
        start = time.perf_counter()
        api.query(query, top_k=10, alpha=alpha)
        latencies.append(time.perf_counter() - start)
    return latencies


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", nargs="?", type=Path, help="Optional database path")
    parser.add_argument("--embed-model", default="debug", help="Embedding model to use")
    parser.add_argument("--target-chunks", type=int, default=2000, help="Approximate chunk count")
    parser.add_argument("--tiny", action="store_true", help="Use a tiny workload for tests")
    args = parser.parse_args(argv)

    demo_dir = demo_corpus_dir()
    target_docs = 60 if args.tiny else max(args.target_chunks, 60)

    tmp_dir = tempfile.TemporaryDirectory(prefix="raglite-bench-")
    corpus_dir = clone_corpus(demo_dir, target_docs)

    try:
        if args.db:
            db_path = args.db
            if db_path.exists():
                db_path.unlink()
        else:
            db_path = Path(tmp_dir.name) / "bench.db"
        config = RagliteConfig(db_path=db_path)
        config.embed_model = args.embed_model
        api = RagliteAPI(config)
        api.init_db()

        start = time.perf_counter()
        ingest_result = api.index(corpus_dir, strategy="fixed")
        index_time = max(time.perf_counter() - start, 1e-6)
        docs_per_second = ingest_result.documents / index_time

        stats = api.stats()
        query_suite = QUERIES[:5] if args.tiny else QUERIES
        latencies_vector = run_queries(api, query_suite, alpha=config.alpha)
        latencies_bm25 = run_queries(api, query_suite, alpha=1.0)

        db_size = db_path.stat().st_size if db_path.exists() else 0

        print("Raglite basic benchmark")
        print("=======================")
        print(f"Corpus copies: {ingest_result.documents}")
        print(f"Indexing throughput: {docs_per_second:.1f} docs/s ({ingest_result.chunks} chunks)")
        print(f"Vector backend: {stats['vector_backend']}")
        print(f"Database size: {db_size / 1024:.1f} KiB")
        print(f"Chunk count: {stats['chunks']}")
        print("")
        print("Query latency (k=10)")
        print("-------------------")
        hybrid_median = statistics.median(latencies_vector) * 1000
        hybrid_p95 = percentile(latencies_vector, 0.95) * 1000
        bm25_median = statistics.median(latencies_bm25) * 1000
        bm25_p95 = percentile(latencies_bm25, 0.95) * 1000

        print(
            f"Hybrid alpha={config.alpha:.1f}: p50={hybrid_median:.1f} ms, "
            f"p95={hybrid_p95:.1f} ms"
        )
        print(f"BM25 only: p50={bm25_median:.1f} ms, " f"p95={bm25_p95:.1f} ms")
    finally:
        shutil.rmtree(corpus_dir, ignore_errors=True)
        tmp_dir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
