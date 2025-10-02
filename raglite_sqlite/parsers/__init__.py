"""Document parsers."""

from .txt import TextParser
from .md import MarkdownParser
from .html import HTMLParser
from .pdf import PDFParser
from .docx import DocxParser
from .csv import CSVParser
from .json import JSONParser
from .pptx import PptxParser
from .image import ImageParser

__all__ = [
    "TextParser",
    "MarkdownParser",
    "HTMLParser",
    "PDFParser",
    "DocxParser",
    "CSVParser",
    "JSONParser",
    "PptxParser",
    "ImageParser",
]
