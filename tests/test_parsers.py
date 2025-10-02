from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

from raglite_sqlite.parsers.image import ImageParser
from raglite_sqlite.parsers.json import JSONParser
from raglite_sqlite.parsers.pptx import PptxParser


def test_json_parser(tmp_path: Path) -> None:
    data_path = tmp_path / "sample.json"
    data_path.write_text("{" "\"title\": \"Hello\", \"items\": [1, 2]}", encoding="utf-8")
    parser = JSONParser()
    blocks = list(parser.parse(str(data_path)))
    assert blocks
    assert "Hello" in blocks[0]["text"]
    assert "items" in blocks[0]["text"]


def test_pptx_parser(tmp_path: Path) -> None:
    pptx_path = tmp_path / "sample.pptx"
    slide_xml = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
    <p:sld xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\"
           xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\">
      <p:cSld>
        <p:spTree>
          <p:sp>
            <p:txBody>
              <a:p><a:r><a:t>Hello PPTX</a:t></a:r></a:p>
            </p:txBody>
          </p:sp>
        </p:spTree>
      </p:cSld>
    </p:sld>
    """
    with ZipFile(pptx_path, "w") as archive:
        archive.writestr("ppt/slides/slide1.xml", slide_xml)
    parser = PptxParser()
    blocks = list(parser.parse(str(pptx_path)))
    assert blocks
    assert "Hello PPTX" in blocks[0]["text"]


def test_image_parser(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    pytest.importorskip("pytesseract")
    from PIL import Image

    image_path = tmp_path / "hello.png"
    image = Image.new("RGB", (10, 10), color="white")
    image.save(image_path)

    import pytesseract

    def fake_ocr(image, lang="eng", config=None):  # type: ignore[no-untyped-def]
        return "Hello OCR"

    monkeypatch.setattr(pytesseract, "image_to_string", fake_ocr)
    parser = ImageParser()
    blocks = list(parser.parse(str(image_path)))
    assert blocks
    assert blocks[0]["text"] == "Hello OCR"
