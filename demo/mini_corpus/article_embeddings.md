# Why embeddings still matter

Even with a small corpus, embeddings capture paraphrased questions. The debug embedding
model hashes tokens for speed yet still distinguishes "backup schedule" from "restore run".
For production, plug in a sentence-transformer to get smoother similarity curves and better
recall for conversations that wander between topics.
