"""raglite - local-first RAG toolkit built on SQLite."""

from importlib import metadata as importlib_metadata

from .api import RagliteAPI, add_tags, index_corpus, init_db, query, stats
from .config import RagliteConfig

try:
    __version__ = importlib_metadata.version("raglite")
except importlib_metadata.PackageNotFoundError:  # pragma: no cover - fallback for dev installs
    __version__ = "0.0.0"

__all__ = [
    "RagliteAPI",
    "RagliteConfig",
    "add_tags",
    "index_corpus",
    "init_db",
    "query",
    "stats",
    "__version__",
]
