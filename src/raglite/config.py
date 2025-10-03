"""Configuration utilities for raglite."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHUNK_TOKENS = 350
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_ALPHA = 0.6


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "raglite"


@dataclass(slots=True)
class RagliteConfig:
    """Runtime configuration for raglite components."""

    db_path: Path
    embed_model: str = DEFAULT_EMBED_MODEL
    cache_dir: Path = field(default_factory=_default_cache_dir)
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    alpha: float = DEFAULT_ALPHA
    rerank_model: Optional[str] = None
    extra_metadata: Dict[str, str] = field(default_factory=dict)

    def ensure_cache_dir(self) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir

    @classmethod
    def from_env(
        cls,
        db_path: str | Path,
        *,
        embed_model: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        alpha: Optional[float] = None,
    ) -> "RagliteConfig":
        path = Path(db_path)
        cfg = cls(db_path=path)
        if embed_model:
            cfg.embed_model = embed_model
        if cache_dir:
            cfg.cache_dir = Path(cache_dir)
        if alpha is not None:
            cfg.alpha = alpha
        return cfg

    def model_cache_path(self, model_name: Optional[str] = None) -> Path:
        name = model_name or self.embed_model
        return self.ensure_cache_dir() / "models" / name.replace("/", "_")

    def chunk_options(self) -> Dict[str, int]:
        return {"max_tokens": self.chunk_tokens, "overlap": self.chunk_overlap}


def clamp_alpha(value: float) -> float:
    return max(0.0, min(1.0, value))
