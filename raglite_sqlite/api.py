from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .chunking import chunk_blocks
from .db import Database
from .embeddings.base import EmbeddingBackend
from .embeddings.hash_backend import HashingBackend
from .parsers.csv import CSVParser
from .parsers.docx import DocxParser
from .parsers.html import HTMLParser
from .parsers.image import ImageParser
from .parsers.json import JSONParser
from .parsers.md import MarkdownParser
from .parsers.pdf import PDFParser
from .parsers.pptx import PptxParser
from .parsers.txt import TextParser
from .search import assemble_results, bm25_search, hybrid_fuse, vector_search
from .typing import SearchResult
from .rerank import Reranker, get_reranker
from .utils import detect_mime, iter_files, loads_json, now_ts, normalize_text, sha256_file, sha256_text

PARSER_REGISTRY = {
    "text/plain": TextParser(),
    "text/markdown": MarkdownParser(),
    "text/html": HTMLParser(),
    "application/pdf": PDFParser(),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocxParser(),
    "text/csv": CSVParser(),
    "application/json": JSONParser(),
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": PptxParser(),
    "image/png": ImageParser(),
    "image/jpeg": ImageParser(),
    "image/tiff": ImageParser(),
}


def _default_backend() -> EmbeddingBackend:
    return HashingBackend()


class RagLite:
    def __init__(self, db_path: str, create: bool = True) -> None:
        self.db_path = Path(db_path)
        self.db = Database(self.db_path, create=create)

    def _doc_id(self, path: Path) -> str:
        return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()

    def _select_parser(self, path: Path) -> Any:
        mime = detect_mime(path)
        return PARSER_REGISTRY.get(mime, TextParser())

    def index(
        self,
        paths: Sequence[str] | str,
        *,
        tags: str | None = None,
        parser_opts: dict | None = None,
        chunker: str = "recursive",
        chunk_size_tokens: int = 512,
        chunk_overlap_tokens: int = 64,
        embedding_backend: EmbeddingBackend | None = None,
        model_name: str | None = None,
        skip_unchanged: bool = True,
        recurse: bool = True,
        glob: str | None = None,
    ) -> dict[str, int]:
        if isinstance(paths, str):
            input_paths = [paths]
        else:
            input_paths = list(paths)
        files = iter_files(input_paths, recurse=recurse, glob=glob)
        backend = embedding_backend or _default_backend()
        stats = {"files": 0, "chunks": 0, "skipped": 0}
        for file_path in files:
            stats["files"] += 1
            doc_id = self._doc_id(file_path)
            existing = self.db.get_document(doc_id)
            doc_sha = sha256_file(file_path)
            if skip_unchanged and existing and existing["sha256"] == doc_sha:
                stats["skipped"] += 1
                continue
            parser = self._select_parser(file_path)
            blocks = parser.parse(str(file_path), **(parser_opts or {}))
            if isinstance(blocks, Iterable):
                blocks_list = list(blocks)
            else:
                blocks_list = [blocks]
            chunks = chunk_blocks(blocks_list, strategy=chunker, size=chunk_size_tokens, overlap=chunk_overlap_tokens)
            created_at = now_ts()
            self.db.upsert_document(
                doc_id,
                source_path=str(file_path),
                mime=detect_mime(file_path),
                tags=tags,
                created_at=created_at,
                updated_at=created_at,
                sha256=doc_sha,
            )
            chunk_hashes: list[str] = []
            for idx, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}:{idx}"
                chunk_hash = sha256_text(chunk.text_norm)
                chunk_hashes.append(chunk_hash)
                self.db.upsert_chunk(
                    chunk_id,
                    doc_id,
                    idx,
                    chunk.text,
                    chunk.text_norm,
                    chunk.section,
                    chunk_hash,
                )
            stats["chunks"] += len(chunks)
            texts = [chunk.text_norm for chunk in chunks]
            if texts:
                embeddings = self._embed_chunks(
                    texts,
                    hashes=chunk_hashes,
                    backend=backend,
                    model_name=model_name or getattr(backend, "model_name", None),
                )
                for idx, vector in enumerate(embeddings):
                    chunk_id = f"{doc_id}:{idx}"
                    self.db.upsert_vector(chunk_id, vector, model_name or getattr(backend, "model_name", "unknown"))
        self.db.commit()
        return stats

    def _embed_chunks(
        self,
        texts: Sequence[str],
        *,
        hashes: Sequence[str],
        backend: EmbeddingBackend,
        model_name: str | None,
    ) -> list[list[float]]:
        cached_vectors: list[list[float] | None] = []
        missing_texts: list[str] = []
        missing_indices: list[int] = []
        model = model_name or getattr(backend, "model_name", "default")
        for idx, (text, content_sha) in enumerate(zip(texts, hashes)):
            cached = self.db.get_embedding_cache(content_sha, model)
            if cached is not None:
                cached_vectors.append(list(cached))
            else:
                cached_vectors.append(None)
                missing_texts.append(text)
                missing_indices.append(idx)
        if missing_texts:
            new_vectors = backend.embed_texts(missing_texts, model_name=model_name)
            for offset, idx in enumerate(missing_indices):
                vector = list(new_vectors[offset])
                cached_vectors[idx] = vector
                self.db.upsert_embedding_cache(hashes[idx], model, vector)
        return [vec for vec in cached_vectors if vec is not None]

    def search(
        self,
        query: str,
        *,
        k: int = 8,
        hybrid_weight: float = 0.6,
        filters: dict | None = None,
        model_name: str | None = None,
        embedding_backend: EmbeddingBackend | None = None,
        max_per_doc: int = 3,
        with_snippets: bool = True,
        reranker: Reranker | str | None = None,
        reranker_options: dict[str, object] | None = None,
    ) -> List[SearchResult]:
        norm_query = normalize_text(query)
        lexical = bm25_search(self.db, norm_query, k, filters)
        semantic: list[tuple[str, float]] = []
        backend = embedding_backend
        if backend is not None:
            semantic = vector_search(self.db, backend, norm_query, model_name, k)
        fused = hybrid_fuse(lexical, semantic, hybrid_weight, k)
        results = assemble_results(
            self.db, fused, k, with_snippets=with_snippets, max_per_doc=max_per_doc
        )
        reranker_instance: Reranker | None
        if isinstance(reranker, str):
            try:
                reranker_instance = get_reranker(reranker, **(reranker_options or {}))
            except KeyError as exc:
                raise ValueError(f"Unknown reranker '{reranker}'") from exc
        else:
            reranker_instance = reranker
        if reranker_instance is not None:
            results = list(reranker_instance.rerank(query, results))
        return results

    def delete(self, doc_id: str) -> None:
        self.db.delete_document(doc_id)
        self.db.commit()

    def stats(self) -> dict[str, Any]:
        return self.db.stats()

    def export(self, to_path: str, include_vectors: bool = False) -> None:
        import json

        with open(to_path, "w", encoding="utf-8") as handle:
            for doc in self.db.list_documents():
                handle.write(json.dumps({"type": "document", **dict(doc)}))
                handle.write("\n")
            cur = self.db.conn.execute("SELECT * FROM chunks")
            for row in cur.fetchall():
                data = dict(row)
                if data.get("extra"):
                    data["extra"] = loads_json(data["extra"])
                handle.write(json.dumps({"type": "chunk", **data}))
                handle.write("\n")
            if include_vectors:
                cur = self.db.conn.execute("SELECT * FROM vectors")
                from array import array

                for row in cur.fetchall():
                    data = dict(row)
                    arr = array("f")
                    arr.frombytes(data["vec"])
                    data["vec"] = list(arr)
                    handle.write(json.dumps({"type": "vector", **data}))
                    handle.write("\n")

    def vacuum(self) -> None:
        self.db.vacuum()

    def close(self) -> None:
        self.db.close()
