from __future__ import annotations

from typing import Iterable, Optional, Protocol, TypedDict


class ParsedBlock(TypedDict, total=False):
    text: str
    section: Optional[str]


class ChunkDict(TypedDict):
    chunk_id: str
    doc_id: str
    position: int
    text: str
    text_norm: str
    section: Optional[str]
    sha256: str


class SearchResult(TypedDict, total=False):
    doc_id: str
    score: float
    snippet: str
    section: Optional[str]
    source_path: str
    position: int
    text: str
    tags: Optional[str]
    bm25_score: float
    vector_score: float
    rerank_score: float


class Parser(Protocol):
    def parse(self, path: str, **options: object) -> Iterable[ParsedBlock]:
        ...
