from __future__ import annotations

import hashlib
import json
import os
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

try:
    import orjson  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - fallback when orjson unavailable
    orjson = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback when tqdm unavailable
    def tqdm(iterable: Iterable[object], total: int | None = None, desc: str = ""):
        for item in iterable:
            yield item


@dataclass
class Progress:
    total: int
    description: str = ""

    def track(self, iterable: Iterable[object]) -> Iterator[object]:
        yield from tqdm(iterable, total=self.total, desc=self.description)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text)
    return " ".join(normalized.split())


def detect_mime(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
        ".html": "text/html",
        ".htm": "text/html",
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".csv": "text/csv",
    }.get(ext, "application/octet-stream")


def now_ts() -> int:
    return int(time.time())


def iter_files(paths: Sequence[str], recurse: bool = True, glob: str | None = None) -> list[Path]:
    candidates: list[Path] = []
    for input_path in paths:
        path = Path(input_path)
        if path.is_dir():
            pattern = glob or "**/*" if recurse else "*"
            for child in path.glob(pattern):
                if child.is_file():
                    candidates.append(child)
        elif path.is_file():
            candidates.append(path)
    unique = {p.resolve(): p for p in candidates}
    return list(unique.values())


def dumps_json(data: object) -> str:
    if orjson is not None:
        return orjson.dumps(data).decode("utf-8")
    import json

    return json.dumps(data)


def loads_json(data: str) -> object:
    import json

    return json.loads(data)
