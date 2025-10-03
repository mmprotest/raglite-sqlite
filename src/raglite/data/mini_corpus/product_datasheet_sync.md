# Raglite Sync Service Overview

Raglite Sync replicates a SQLite index between laptops. It watches the `.db` and `.db-wal`
files, compresses new pages, and pushes them over SSH. Conflict resolution is simple: the
latest file wins, so schedule syncs right after ingestion jobs finish.
