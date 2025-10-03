# Local-first RAG in practice

Raglite keeps retrieval pipelines on your laptop. The SQLite core fits alongside existing
notebooks, so teams avoid provisioning a remote vector database. Hybrid ranking blends
BM25 filtering with cosine similarity to surface precise answers from even tiny datasets.
Because ingestion produces deterministic chunks, you can commit the resulting database to
version control and reproduce experiments later without drift.
