"""Utilities for chunking text."""

from __future__ import annotations

import re
from typing import List

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def estimate_tokens(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def split_fixed_tokens(text: str, *, max_tokens: int = 350, overlap: int = 50) -> List[str]:
    tokens = TOKEN_PATTERN.findall(text)
    if not tokens:
        return []
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if overlap >= max_tokens:
        raise ValueError("overlap must be smaller than max_tokens")

    chunks: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(len(tokens), start + max_tokens)
        chunk_tokens = tokens[start:end]
        chunks.append(" ".join(chunk_tokens).strip())
        if end == len(tokens):
            break
        start = end - overlap
    return [c for c in chunks if c]


def split_recursive(
    text: str,
    *,
    max_tokens: int = 350,
    overlap: int = 50,
    prefer_headings: bool = True,
) -> List[str]:
    if not text.strip():
        return []
    sections = _split_by_headings(text) if prefer_headings else [text]
    results: List[str] = []
    for section in sections:
        tokens = TOKEN_PATTERN.findall(section)
        if len(tokens) <= max_tokens:
            results.append(section.strip())
            continue
        results.extend(split_fixed_tokens(section, max_tokens=max_tokens, overlap=overlap))
    return [c for c in results if c]


def _split_by_headings(text: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    for line in text.splitlines():
        if line.strip().startswith(("#", "=", "-")) and current:
            parts.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        parts.append("\n".join(current).strip())
    return parts or [text]


def chunk_text(
    text: str,
    *,
    strategy: str = "recursive",
    max_tokens: int = 350,
    overlap: int = 50,
) -> List[str]:
    if strategy == "recursive":
        return split_recursive(text, max_tokens=max_tokens, overlap=overlap)
    if strategy == "fixed":
        return split_fixed_tokens(text, max_tokens=max_tokens, overlap=overlap)
    raise ValueError(f"Unknown strategy: {strategy}")
