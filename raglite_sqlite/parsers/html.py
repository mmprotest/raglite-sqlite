from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .base import BaseParser
from ..typing import ParsedBlock
from ..utils import normalize_text


class HTMLParser(BaseParser):
    def parse(self, path: str, **options: object) -> Iterable[ParsedBlock]:
        content = Path(path).read_text(encoding="utf-8")
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(content, "html.parser")
            extractor = "soup"
        except Exception:
            soup = None
            extractor = "plain"
        blocks: List[ParsedBlock] = []
        current_section: str | None = None
        if extractor == "soup" and soup is not None:
            for element in soup.find_all(["h1", "h2", "h3", "p"]):
                if element.name in {"h1", "h2", "h3"}:
                    current_section = normalize_text(element.get_text(" "))
                else:
                    text = normalize_text(element.get_text(" "))
                    if text:
                        blocks.append(ParsedBlock(text=text, section=current_section))
            if not blocks:
                text = normalize_text(soup.get_text(" "))
                blocks.append(ParsedBlock(text=text, section=None))
        else:
            clean = normalize_text(
                content.replace("<", " ").replace(">", " ")
            )
            blocks.append(ParsedBlock(text=clean, section=None))
        return blocks
