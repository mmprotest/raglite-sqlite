from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .base import BaseParser
from ..typing import ParsedBlock
from ..utils import normalize_text


class PDFParser(BaseParser):
    def parse(self, path: str, **options: object) -> Iterable[ParsedBlock]:
        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(Path(path).open("rb"))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            text = ""
        yield ParsedBlock(text=normalize_text(text), section=None)
