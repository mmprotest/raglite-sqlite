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
- Index a directory recursively with tags and progress reporting:
  ```bash
  raglite index docs --db knowledge.db --tags project,internal --recursive --chunk-size 400 --overlap 50
  ```
- Re-index only changed files by disabling recursion and pointing at a single file:
  ```bash
  raglite index docs/faq.md --db knowledge.db --no-recursive --skip-unchanged
  ```
- Query for relevant chunks:
  ```bash
  raglite query "How do I deploy?" --db knowledge.db -k 5 --hybrid 0.7
  ```
- Filter by tag or doc id:
  ```bash
  raglite query "release checklist" --db knowledge.db --filter tag=internal --filter doc_id=release-notes
  ```
- Inspect statistics:
  ```bash
  raglite stats --db knowledge.db
  ```
- Export metadata:
  ```bash
  raglite export --db knowledge.db --to export.ndjson --include-vectors
  ```
- Vacuum the database to reclaim space:
  ```bash
  raglite vacuum --db knowledge.db
  ```

## Python API

The high-level API is built around `raglite_sqlite.api.RagLite`. Below is a minimal end-to-end script that indexes a directory,
queries it, and prints rich metadata for each hit:

```python
from raglite_sqlite import RagLite
from raglite_sqlite.embeddings.sentence_transformers_backend import SentenceTransformersBackend

rag = RagLite("knowledge.db")
backend = SentenceTransformersBackend(model_name="sentence-transformers/all-MiniLM-L6-v2")

rag.index(
    paths=["tests/data"],
    tags="demo",
    embedding_backend=backend,
    chunker="recursive",
    chunk_size_tokens=384,
    chunk_overlap_tokens=48,
)

results = rag.search(
    query="example text",
    embedding_backend=backend,
    k=5,
    hybrid_weight=0.6,
    max_per_doc=2,
)

for item in results:
    print(f"{item['score']:.3f} | {item['doc_id']} | {item['section'] or 'No section'}")
    print(item["snippet"])
    print("-" * 40)
```

### Advanced ingestion tips

- **Custom parsers and options** – pass `parser_opts` to `RagLite.index` to control parser behaviour (e.g. CSV column filters).
- **Chunking strategies** – choose between `fixed_tokens` (uniform splits) and `recursive` (paragraph-aware) chunkers.
- **Embedding reuse** – keep `skip_unchanged=True` (default) to leverage hashing + cache table for faster re-index runs.
- **Multiple models** – provide different `model_name` values per run; embeddings are cached per `(sha256, model)` pair.
- **Export/backup** – `raglite export` produces NDJSON that can be restored or version-controlled for auditing.

### Operating the database

- SQLite is safe to sync via file-sharing tools (Dropbox, Syncthing) when only one process writes at a time.
- Use `raglite vacuum` periodically after large deletions to compact the file.
- WAL mode is enabled automatically; consider copying the `.db` + `-wal` file pair while the app is running.
- For cloud backups, store the DB file and optionally NDJSON exports in object storage.

## Ecosystem Integrations

RagLite ships with lightweight adapters so you can plug the SQLite-backed retriever into popular orchestration frameworks.

### LangChain

```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

from raglite_sqlite import RagLite
from raglite_sqlite.adapters.langchain import RagLiteRetriever
from raglite_sqlite.embeddings.sentence_transformers_backend import SentenceTransformersBackend

rag = RagLite("knowledge.db")
backend = SentenceTransformersBackend()

# Ensure documents are indexed before wiring up the retriever.
rag.index(["tests/data"], embedding_backend=backend)

retriever = RagLiteRetriever(rag, embedding_backend=backend, k=6, hybrid_weight=0.5)

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=retriever,
    chain_type="stuff",
)

response = qa_chain.run("Summarize the sample docs")
print(response)
```

The adapter defers LangChain imports until used, keeping the core package lightweight. You can also supply filters (`tag=...`)
via the retriever call (`retriever.get_relevant_documents(query, filters={"tags": "internal"})`).

### LlamaIndex

```python
from llama_index.core import ServiceContext, VectorStoreIndex
from llama_index.llms import OpenAI

from raglite_sqlite import RagLite
from raglite_sqlite.adapters.llamaindex import RagLiteVectorStore
from raglite_sqlite.embeddings.sentence_transformers_backend import SentenceTransformersBackend

rag = RagLite("knowledge.db")
backend = SentenceTransformersBackend()

rag.index(["tests/data"], embedding_backend=backend)

service_context = ServiceContext.from_defaults(llm=OpenAI())
vector_store = RagLiteVectorStore(rag, embedding_backend=backend)

index = VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)
query_engine = index.as_query_engine(similarity_top_k=5)

answer = query_engine.query("What is contained in the sample docs?")
print(answer)
```

The vector store wrapper exposes RagLite search semantics to LlamaIndex while leaving indexing/ingest under your control.

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
