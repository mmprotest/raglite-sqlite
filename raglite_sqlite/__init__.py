"""RagLite SQLite package."""

from .api import RagLite

try:  # pragma: no cover - optional dependency
    from .server import create_app
except Exception:  # pragma: no cover - FastAPI not installed
    create_app = None  # type: ignore[assignment]

__all__ = ["RagLite", "create_app"]
