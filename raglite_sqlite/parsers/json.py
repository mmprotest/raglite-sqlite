from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .base import BaseParser
from ..typing import ParsedBlock
from ..utils import normalize_text


class JSONParser(BaseParser):
    """Parse JSON documents by rendering them as a stable, human-readable string."""

    def parse(self, path: str, **options: object) -> Iterable[ParsedBlock]:
        indent = int(options.get("indent", 2) or 0)
        sort_keys = bool(options.get("sort_keys", True))
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        text = json.dumps(data, indent=indent or None, sort_keys=sort_keys, ensure_ascii=False)
        yield ParsedBlock(text=normalize_text(text), section=None)
