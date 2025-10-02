from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .base import BaseParser
from ..typing import ParsedBlock
from ..utils import normalize_text


class ImageParser(BaseParser):
    """Parse image files using pytesseract-based OCR."""

    def __init__(self, default_lang: str = "eng") -> None:
        self.default_lang = default_lang

    def parse(self, path: str, **options: object) -> Iterable[ParsedBlock]:
        lang = str(options.get("lang", self.default_lang))
        config = options.get("tesseract_config")
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError("Pillow is required for OCR parsing") from exc
        try:
            import pytesseract
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError("pytesseract is required for OCR parsing") from exc
        image_path = Path(path)
        with Image.open(image_path) as image:
            try:
                text = pytesseract.image_to_string(image, lang=lang, config=config)
            except pytesseract.pytesseract.TesseractNotFoundError as exc:  # pragma: no cover - environment specific
                raise RuntimeError(
                    "Tesseract OCR binary not found. Install it or adjust TESSDATA_PREFIX."
                ) from exc
        yield ParsedBlock(text=normalize_text(text), section=None)
