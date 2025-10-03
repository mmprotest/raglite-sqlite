# Troubleshooting FAQ

**Q: My search returns empty results.**
A: Verify the ingest step ran with the same embedding model that your server uses.

**Q: Why is the CLI slow on first run?**
A: The vector cache builds the first time you query. Subsequent runs are instant.

**Q: How do I reset the index?**
A: Delete the SQLite file and re-run `raglite init-db` followed by ingestion.
