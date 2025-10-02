# RagLite SQLite

Local-first Retrieval-Augmented Generation (RAG) toolkit built entirely on top of a single SQLite file. RagLite makes it easy to ingest documents, chunk them, embed with pluggable backends, and perform hybrid lexical/vector search—without running servers or Docker images.

## Why RagLite?

- **Zero infrastructure** – everything lives inside one SQLite database (`knowledge.db`).
- **Deterministic and offline** – default embedding model is local; no network calls unless explicitly configured.
- **Hybrid retrieval** – combines BM25 via FTS5 with cosine similarity over stored vectors.
- **Python and CLI** – flexible API plus a friendly Typer-based CLI for scripting.
- **Extensible** – pluggable parsers, chunkers, embedding backends, and adapters for LangChain / LlamaIndex.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -e .
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -e .
raglite init --db knowledge.db
raglite index tests/data --db knowledge.db --model sentence-transformers/all-MiniLM-L6-v2
raglite query "example text" --db knowledge.db -k 8 --hybrid 0.6
```

## CLI Examples

- Initialize a new database:
  ```bash
  raglite init --db knowledge.db
  ```
- Index a directory recursively with tags:
  ```bash
  raglite index docs --db knowledge.db --tags project,internal --recursive
  ```
- Query for relevant chunks:
  ```bash
  raglite query "How do I deploy?" --db knowledge.db -k 5
  ```
- Inspect statistics:
  ```bash
  raglite stats --db knowledge.db
  ```
- Export metadata:
  ```bash
  raglite export --db knowledge.db --to export.ndjson --include-vectors
  ```

## Python API

```python
from raglite_sqlite import RagLite
from raglite_sqlite.embeddings.sentence_transformers_backend import SentenceTransformersBackend

rag = RagLite("knowledge.db")
backend = SentenceTransformersBackend()
rag.index(["tests/data"], tags="demo", embedding_backend=backend)
results = rag.search("example text", embedding_backend=backend)
for item in results:
    print(item["score"], item["snippet"])
```

## Design Notes

- Uses SQLite with WAL mode and FTS5 for zero-config deployment.
- Chunker strategies allow fixed token or recursive splitting with overlaps.
- Embeddings cached via SHA-256 to avoid redundant model calls.
- Hybrid retrieval fuses normalized BM25 and cosine similarity scores.

## Security & Privacy

RagLite does not perform any network calls by default. Remote embedding backends (such as OpenAI) are opt-in and require explicit configuration via environment variables.

## Roadmap

- Optional REST server for multi-user access.
- Support for additional document formats and OCR.
- Pluggable reranking models.

## License

MIT License. See [LICENSE](LICENSE).
