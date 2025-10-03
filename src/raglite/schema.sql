PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL,
    title TEXT,
    mime TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    meta_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_idx INTEGER NOT NULL,
    text TEXT NOT NULL,
    tokens INTEGER DEFAULT 0,
    tags_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY,
    chunk_id INTEGER NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    embedding BLOB NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
    text,
    content='chunks',
    content_rowid='id'
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_model ON embeddings(chunk_id, model);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunk_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES('delete', old.id, old.text);
    INSERT INTO chunk_fts(rowid, text) VALUES(new.id, new.text);
END;
