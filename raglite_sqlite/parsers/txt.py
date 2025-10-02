from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .base import BaseParser
from ..typing import ParsedBlock
from ..utils import normalize_text


class TextParser(BaseParser):
    def parse(self, path: str, **options: object) -> Iterable[ParsedBlock]:
        content = Path(path).read_text(encoding="utf-8")
        yield ParsedBlock(text=normalize_text(content), section=None)
