# Backup strategy for local indexes

Treat the Raglite database like source code. Commit small demo indexes, but snapshot
production data using nightly exports. Store both the `.db` file and its `.db-wal` partner,
then verify the archive by opening it with `sqlite3` in read-only mode. Regular integrity
checks keep retrieval fast and avoid fragmented pages.
