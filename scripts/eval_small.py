#!/usr/bin/env python3
"""Tiny offline evaluation comparing BM25 and hybrid search."""

from __future__ import annotations

import argparse
import importlib.resources as resources
import json
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path and SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

from raglite.api import RagliteAPI, RagliteConfig  # noqa: E402

EVAL_QUERIES = 25


@dataclass
class QueryCase:
    query: str
    relevant_substring: str


@dataclass
class VariantMetrics:
    name: str
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float


def load_eval_set(path: Path, *, limit: int | None = None) -> List[QueryCase]:
    cases: List[QueryCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cases.append(
                QueryCase(query=data["query"], relevant_substring=data["relevant_substring"])
            )
            if limit is not None and len(cases) >= limit:
                break
    return cases


def _demo_corpus_dir() -> Path:
    fs_path = Path(__file__).resolve().parents[1] / "demo" / "mini_corpus"
    if fs_path.exists():
        return fs_path
    with resources.as_file(resources.files("raglite").joinpath("data/mini_corpus")) as packaged:
        return packaged


def _eval_set_path() -> Path:
    fs_path = Path(__file__).resolve().parents[1] / "demo" / "eval_set.jsonl"
    if fs_path.exists():
        return fs_path
    with resources.as_file(resources.files("raglite").joinpath("data/eval_set.jsonl")) as packaged:
        return packaged


def ensure_api(db_path: Path, *, embed_model: str, alpha: float) -> RagliteAPI:
    if db_path.exists():
        db_path.unlink()
    config = RagliteConfig(db_path=db_path)
    config.embed_model = embed_model
    config.alpha = alpha
    api = RagliteAPI(config)
    api.init_db()
    api.index(_demo_corpus_dir(), strategy="fixed")
    return api


def evaluate_variant(
    api: RagliteAPI,
    cases: Sequence[QueryCase],
    *,
    alpha: float,
    rerank: bool = False,
    top_k: int = 10,
) -> VariantMetrics:
    hits_5 = 0
    hits_10 = 0
    reciprocal_ranks: List[float] = []
    for case in cases:
        results = api.query(case.query, top_k=top_k, alpha=alpha, rerank=rerank)
        rank = hit_rank(results, case.relevant_substring)
        if rank is not None:
            if rank < 5:
                hits_5 += 1
            if rank < 10:
                hits_10 += 1
            reciprocal_ranks.append(1.0 / (rank + 1))
        else:
            reciprocal_ranks.append(0.0)
    total = len(cases) or 1
    return VariantMetrics(
        name="Hybrid+Rerank" if rerank else ("BM25" if alpha >= 0.99 else f"Hybrid(α={alpha:.1f})"),
        recall_at_5=hits_5 / total,
        recall_at_10=hits_10 / total,
        mrr_at_10=statistics.fmean(reciprocal_ranks) if reciprocal_ranks else 0.0,
    )


def hit_rank(results: Iterable, needle: str) -> int | None:
    needle_lower = needle.lower()
    for idx, result in enumerate(results):
        if idx >= 10:
            break
        text = getattr(result, "text", "")
        if needle_lower in text.lower():
            return idx
    return None


def print_table(metrics: Sequence[VariantMetrics]) -> None:
    print("| Variant | R@5 | R@10 | MRR@10 |")
    print("| --- | --- | --- | --- |")
    for row in metrics:
        print(
            f"| {row.name} | {row.recall_at_5:.2f} | {row.recall_at_10:.2f} | {row.mrr_at_10:.3f} |"
        )


def rerank_available() -> bool:
    try:
        import sentence_transformers  # noqa: F401
    except Exception:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", nargs="?", type=Path, help="Optional database path to reuse")
    parser.add_argument("--embed-model", default="debug", help="Embedding model to use")
    parser.add_argument("--alpha", type=float, default=0.6, help="Hybrid alpha for default run")
    parser.add_argument("--tiny", action="store_true", help="Evaluate only the first 5 queries")
    args = parser.parse_args(argv)

    eval_path = _eval_set_path()
    limit = 5 if args.tiny else None
    cases = load_eval_set(eval_path, limit=limit)
    if not cases:
        print("No evaluation cases found", file=sys.stderr)
        return 1

    tmp_dir = None
    if args.db:
        db_path = args.db
    else:
        tmp_dir = tempfile.TemporaryDirectory(prefix="raglite-eval-")
        db_path = Path(tmp_dir.name) / "eval_small.db"
    api = ensure_api(db_path, embed_model=args.embed_model, alpha=args.alpha)

    metrics: List[VariantMetrics] = []
    metrics.append(evaluate_variant(api, cases, alpha=1.0, rerank=False))
    metrics.append(evaluate_variant(api, cases, alpha=args.alpha, rerank=False))

    if rerank_available():
        metrics.append(evaluate_variant(api, cases, alpha=args.alpha, rerank=True))
    else:
        print("(Hybrid+Rerank skipped: install raglite-sqlite[rerank] to enable)")

    print_table(metrics)
    if tmp_dir is not None:
        tmp_dir.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
