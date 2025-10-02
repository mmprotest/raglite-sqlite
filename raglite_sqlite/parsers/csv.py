from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

from .base import BaseParser
from ..typing import ParsedBlock
from ..utils import normalize_text


class CSVParser(BaseParser):
    def parse(self, path: str, **options: object) -> Iterable[ParsedBlock]:
        columns = options.get("columns") if isinstance(options, dict) else None
        selected: Sequence[str] | None = None
        if isinstance(columns, Sequence):
            selected = list(columns)
        rows: list[str] = []
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if selected:
                    values = [str(row.get(col, "")) for col in selected]
                else:
                    values = [str(value) for value in row.values()]
                rows.append(" ".join(values))
        yield ParsedBlock(text=normalize_text("\n".join(rows)), section=None)
