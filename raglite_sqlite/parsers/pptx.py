from __future__ import annotations

from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree
from zipfile import ZipFile

from .base import BaseParser
from ..typing import ParsedBlock
from ..utils import normalize_text


class PptxParser(BaseParser):
    """Lightweight PPTX parser that extracts text from slide XML payloads."""

    SLIDE_PREFIX = "ppt/slides/"
    SLIDE_SUFFIX = ".xml"

    def parse(self, path: str, **options: object) -> Iterable[ParsedBlock]:
        pptx_path = Path(path)
        texts: list[str] = []
        with ZipFile(pptx_path) as archive:
            slide_names = sorted(
                name
                for name in archive.namelist()
                if name.startswith(self.SLIDE_PREFIX) and name.endswith(self.SLIDE_SUFFIX)
            )
            for slide_name in slide_names:
                with archive.open(slide_name) as handle:
                    xml_bytes = handle.read()
                try:
                    root = ElementTree.fromstring(xml_bytes)
                except ElementTree.ParseError:
                    continue
                namespaces = {
                    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
                }
                slide_text: list[str] = []
                for node in root.findall('.//a:t', namespaces):
                    if node.text:
                        slide_text.append(node.text)
                if slide_text:
                    texts.append(" ".join(slide_text))
        combined = normalize_text("\n\n".join(texts))
        yield ParsedBlock(text=combined, section=None)
