"""Corpus ingestion utilities."""

from __future__ import annotations

import mimetypes
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

from . import chunk as chunk_utils
from .config import RagliteConfig
from .db import apply_migrations, connect
from .embed import get_embedding_store

try:
    from readability import Document
except Exception:  # pragma: no cover - optional dependency
    Document = None

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional dependency
    BeautifulSoup = None

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None

try:  # optional OCR
    import pytesseract
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None
    Image = None


@dataclass
class IngestedChunk:
    id: int
    document_id: int
    chunk_idx: int
    text: str
    tokens: int


@dataclass
class IngestResult:
    documents: int
    chunks: int
    embeddings: int


TEXT_MIME_TYPES = {
    "text/plain",
    "text/markdown",
    "text/html",
    "application/json",
}


class UnsupportedDocument(RuntimeError):
    pass


def discover_files(path: Path) -> Iterator[Path]:
    if path.is_file():
        yield path
        return
    for root, _, files in os.walk(path):
        for name in files:
            yield Path(root) / name


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_html_file(path: Path) -> str:
    html = path.read_text(encoding="utf-8", errors="ignore")
    if Document is None or BeautifulSoup is None:
        return html
    doc = Document(html)
    summary = doc.summary()
    soup = BeautifulSoup(summary, "html.parser")
    return soup.get_text(" ")


def read_pdf_file(path: Path, *, ocr: bool = False) -> str:
    if PdfReader is None:
        raise UnsupportedDocument("pypdf is required to read PDF files")
    reader = PdfReader(str(path))
    texts: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if not text.strip() and ocr and pytesseract and Image:
            try:
                images = page.images
            except AttributeError:  # pragma: no cover
                images = []
            for image in images:
                try:
                    img = Image.open(image.data)
                except Exception:  # pragma: no cover
                    continue
                texts.append(pytesseract.image_to_string(img))
        texts.append(text)
    return "\n".join(filter(None, texts))


def ingest_path(
    db_path: Path,
    corpus_path: Path,
    *,
    config: RagliteConfig,
    strategy: str = "recursive",
    ocr: bool = False,
) -> IngestResult:
    conn = connect(db_path)
    apply_migrations(conn)
    embedding_store = get_embedding_store(config.embed_model)
    total_docs = 0
    total_chunks = 0
    total_embeddings = 0

    with conn:
        for file_path in discover_files(corpus_path):
            try:
                doc_text, mime = load_file(file_path, ocr=ocr)
            except UnsupportedDocument:
                continue
            if not doc_text.strip():
                continue
            total_docs += 1
            doc_id = insert_document(conn, file_path, mime)
            chunk_texts = chunk_utils.chunk_text(
                doc_text,
                strategy=strategy,
                max_tokens=config.chunk_tokens,
                overlap=config.chunk_overlap,
            )
            chunks = insert_chunks(conn, doc_id, chunk_texts)
            total_chunks += len(chunks)
            embeddings = embedding_store.embed_many([c.text for c in chunks])
            total_embeddings += len(embeddings)
            insert_embeddings(
                conn, chunks, embeddings, embedding_store.model_name, embedding_store.dimension
            )
    conn.close()
    return IngestResult(total_docs, total_chunks, total_embeddings)


def load_file(path: Path, *, ocr: bool = False) -> Tuple[str, str]:
    mime, _ = mimetypes.guess_type(path.name)
    mime = mime or "application/octet-stream"
    if path.suffix.lower() in {".md", ".markdown"}:
        text = read_text_file(path)
        return text, "text/markdown"
    if mime in {"text/plain", "application/json"}:
        return read_text_file(path), mime
    if mime == "text/html":
        return read_html_file(path), mime
    if path.suffix.lower() == ".pdf":
        return read_pdf_file(path, ocr=ocr), "application/pdf"
    raise UnsupportedDocument(f"Unsupported file: {path}")


def insert_document(conn: sqlite3.Connection, path: Path, mime: str) -> int:
    cur = conn.execute(
        "INSERT INTO documents(path, title, mime) VALUES (?, ?, ?)",
        (str(path), path.stem, mime),
    )
    row_id = cur.lastrowid
    assert row_id is not None
    return int(row_id)


def insert_chunks(
    conn: sqlite3.Connection, document_id: int, chunk_texts: Sequence[str]
) -> List[IngestedChunk]:
    chunks: List[IngestedChunk] = []
    for idx, text in enumerate(chunk_texts):
        tokens = chunk_utils.estimate_tokens(text)
        cur = conn.execute(
            "INSERT INTO chunks(document_id, chunk_idx, text, tokens) VALUES (?, ?, ?, ?)",
            (document_id, idx, text, tokens),
        )
        chunk_row_id = cur.lastrowid
        assert chunk_row_id is not None
        chunk_id = int(chunk_row_id)
        chunks.append(IngestedChunk(chunk_id, document_id, idx, text, tokens))
    return chunks


def insert_embeddings(
    conn: sqlite3.Connection,
    chunks: Sequence[IngestedChunk],
    vectors: Sequence[bytes],
    model: str,
    dim: int,
) -> None:
    for chunk, vector in zip(chunks, vectors, strict=False):
        conn.execute(
            "INSERT INTO embeddings(chunk_id, model, dim, embedding) VALUES (?, ?, ?, ?)",
            (chunk.id, model, dim, vector),
        )
