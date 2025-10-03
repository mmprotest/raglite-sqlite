"""Vector backend selection."""

from .backend import Backend, VectorBackend, detect_backend, get_backend

__all__ = ["Backend", "VectorBackend", "detect_backend", "get_backend"]
