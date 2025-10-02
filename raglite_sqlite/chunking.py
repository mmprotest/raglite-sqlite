from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, List

from .typing import ParsedBlock
from .utils import normalize_text


@dataclass
class Chunk:
    position: int
    text: str
    text_norm: str
    section: str | None = None


def fixed_tokens(text: str, size: int = 512, overlap: int = 64) -> Iterator[Chunk]:
    tokens = text.split()
    start = 0
    position = 0
    while start < len(tokens):
        end = min(start + size, len(tokens))
        chunk_tokens = tokens[start:end]
        raw = " ".join(chunk_tokens)
        yield Chunk(position=position, text=raw, text_norm=normalize_text(raw))
        position += 1
        if end == len(tokens):
            break
        start = max(end - overlap, 0)


def recursive(text: str, size: int = 512, overlap: int = 64, break_on: List[str] | None = None) -> Iterator[Chunk]:
    if break_on is None:
        break_on = ["\n\n", ". "]

    def split(text_block: str) -> Iterable[str]:
        for delimiter in break_on:
            if delimiter in text_block and len(text_block) > size * 2:
                parts = text_block.split(delimiter)
                for i, part in enumerate(parts):
                    suffix = delimiter if i < len(parts) - 1 else ""
                    yield from split(part + suffix)
                return
        yield text_block

    pieces = [piece.strip() for piece in split(text) if piece.strip()]
    buffer: list[str] = []
    position = 0
    for piece in pieces:
        buffer.append(piece)
        candidate = " ".join(buffer)
        if len(candidate.split()) >= size:
            yield Chunk(position=position, text=candidate, text_norm=normalize_text(candidate))
            position += 1
            buffer = buffer[-1:]  # keep last element for overlap
    if buffer:
        remainder = " ".join(buffer)
        if remainder:
            yield Chunk(position=position, text=remainder, text_norm=normalize_text(remainder))


def chunk_blocks(blocks: Iterable[ParsedBlock], strategy: str = "recursive", size: int = 512, overlap: int = 64) -> list[Chunk]:
    chunks: list[Chunk] = []
    for idx, block in enumerate(blocks):
        section = block.get("section")
        text = block.get("text", "")
        chunk_iter = recursive if strategy == "recursive" else fixed_tokens
        for chunk in chunk_iter(text, size=size, overlap=overlap):  # type: ignore[arg-type]
            chunk.section = section
            chunk.position += len(chunks)
            chunks.append(chunk)
    return chunks
