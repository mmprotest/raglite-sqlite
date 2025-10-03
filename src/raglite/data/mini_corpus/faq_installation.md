# Installation FAQ

**Q: Do I need admin rights?**
A: No. Raglite ships as pure Python and SQLite. Install inside a virtual environment.

**Q: What about embeddings?**
A: Use `--embed-model debug` for air-gapped demos or install `sentence-transformers` for production.

**Q: Can I run without GPUs?**
A: Yes. All pipelines run on CPU and complete within seconds for the demo corpus.
