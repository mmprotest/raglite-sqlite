# raglite

Local-first retrieval augmented generation toolkit built on SQLite. Raglite bundles ingestion, chunking, embeddings, hybrid BM25/vector search, a Typer CLI, and a FastAPI microservice. Everything runs on CPU and stores state in a single SQLite database.

## Features

- SQLite FTS5 BM25 search with optional vector reranking.
- Pluggable embedding backends with automatic fallback to a deterministic hash model for testing.
- Python API, Typer CLI, and FastAPI server.
- Demo corpus and integration examples for LangChain and LlamaIndex.

## Quickstart

```bash
pip install -e .[server,dev]
raglite init-db --db demo.db
raglite ingest --db demo.db --path src/demo/mini_corpus --embed-model debug
raglite query --db demo.db --text "quick start guide"
```

## Architecture

```
+-----------------+
|   Typer CLI     |
+-----------------+
        |
+-----------------+     +-----------------------+
|    API Layer    |<--->|    FastAPI Service    |
+-----------------+     +-----------------------+
        |
+-----------------+     +-----------------------+
|  Search Engine  |<--->| Vector Backends       |
+-----------------+     +-----------------------+
        |
+-----------------------------------------------+
|                   SQLite DB                    |
+-----------------------------------------------+
```

See `src/raglite/schema.sql` for the full schema.

## Tests

```bash
pytest -q
```

## License

MIT
