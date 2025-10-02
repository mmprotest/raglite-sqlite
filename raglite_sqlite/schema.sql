PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    mime TEXT,
    tags TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    sha256 TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    text TEXT NOT NULL,
    text_norm TEXT NOT NULL,
    section TEXT,
    sha256 TEXT NOT NULL,
    extra JSON,
    FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    chunk_id,
    text_norm,
    tokenize = 'unicode61'
);

CREATE TABLE IF NOT EXISTS vectors (
    chunk_id TEXT PRIMARY KEY,
    dim INTEGER NOT NULL,
    dtype TEXT NOT NULL,
    vec BLOB NOT NULL,
    model_name TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS cache_embeddings (
    content_sha TEXT NOT NULL,
    model_name TEXT NOT NULL,
    dim INTEGER NOT NULL,
    dtype TEXT NOT NULL,
    vec BLOB NOT NULL,
    PRIMARY KEY(content_sha, model_name)
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_sha ON documents(sha256);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, position);
CREATE INDEX IF NOT EXISTS idx_chunks_sha ON chunks(sha256);
CREATE INDEX IF NOT EXISTS idx_vectors_model ON vectors(model_name);
