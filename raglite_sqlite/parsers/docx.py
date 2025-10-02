from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .base import BaseParser
from ..typing import ParsedBlock
from ..utils import normalize_text


class DocxParser(BaseParser):
    def parse(self, path: str, **options: object) -> Iterable[ParsedBlock]:
        try:
            from docx import Document

            document = Document(Path(path))
            texts = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
        except Exception:
            texts = []
        yield ParsedBlock(text=normalize_text("\n".join(texts)), section=None)
