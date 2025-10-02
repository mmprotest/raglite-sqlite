from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .base import BaseParser
from ..typing import ParsedBlock
from ..utils import normalize_text


class MarkdownParser(BaseParser):
    def parse(self, path: str, **options: object) -> Iterable[ParsedBlock]:
        content = Path(path).read_text(encoding="utf-8")
        try:
            import frontmatter

            post = frontmatter.loads(content)
            body = post.content
        except Exception:
            body = content
        lines = body.splitlines()
        blocks: List[ParsedBlock] = []
        current_section: str | None = None
        buffer: list[str] = []
        for line in lines:
            if line.startswith("#"):
                if buffer:
                    blocks.append(ParsedBlock(text=normalize_text("\n".join(buffer)), section=current_section))
                    buffer = []
                current_section = normalize_text(line.lstrip("# "))
            elif line.strip().startswith("```"):
                continue
            else:
                buffer.append(line)
        if buffer:
            blocks.append(ParsedBlock(text=normalize_text("\n".join(buffer)), section=current_section))
        if not blocks:
            blocks.append(ParsedBlock(text=normalize_text(body), section=current_section))
        return blocks
