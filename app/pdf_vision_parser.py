"""Vision-based PDF Parser — SOTA parsing using Docling (IBM) for tables & structure.

Replaces the simple PyMuPDF `page.get_text()` approach which destroys tables
and multi-column layouts. Uses Docling's AI Vision pipeline to:
  - Preserve table structure → Markdown tables
  - Detect and remove headers/footers
  - Handle multi-column layouts
  - Extract figure captions

Fallback chain: Docling → PyMuPDF (current behavior)

━━━ INSTALLATION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
pip install docling
# ONNX-based, no GPU required, runs completely offline
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Track Docling availability
_DOCLING_AVAILABLE: bool | None = None


def _check_docling() -> bool:
    """Check if Docling is installed and available."""
    global _DOCLING_AVAILABLE
    if _DOCLING_AVAILABLE is not None:
        return _DOCLING_AVAILABLE
    try:
        from docling.document_converter import DocumentConverter  # noqa: F401
        _DOCLING_AVAILABLE = True
        logger.info("VisionPdfParser: Docling available ✓")
    except ImportError:
        _DOCLING_AVAILABLE = False
        logger.info("VisionPdfParser: Docling not installed — will use PyMuPDF fallback")
    return _DOCLING_AVAILABLE


def _clean_text(text: str) -> str:
    """Clean extracted text."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_pdf_with_docling(pdf_path: str | Path) -> str:
    """Parse a PDF file using Docling (IBM) vision-based pipeline.

    Returns Markdown text with tables preserved.

    Parameters
    ----------
    pdf_path : str | Path
        Path to the PDF file.

    Returns
    -------
    str
        Extracted text in Markdown format.

    Raises
    ------
    ImportError
        If Docling is not installed.
    """
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))

    # Export as Markdown (preserves tables, headings, structure)
    markdown_text = result.document.export_to_markdown()

    return _clean_text(markdown_text)


def parse_pdf_with_pymupdf(pdf_path: str | Path | bytes) -> str:
    """Parse a PDF file using PyMuPDF (fallback).

    Simple text extraction — tables will be flattened.
    """
    import fitz

    if isinstance(pdf_path, bytes):
        pdf = fitz.open(stream=pdf_path, filetype="pdf")
    else:
        pdf = fitz.open(str(pdf_path))

    pages: list[str] = []
    for page in pdf:
        text = page.get_text() or ""
        text = _clean_text(text)
        if text:
            pages.append(text)
    pdf.close()
    return "\n".join(pages)


def parse_pdf(pdf_path: str | Path | bytes, prefer_docling: bool = True) -> str:
    """Parse a PDF file with the best available parser.

    Fallback chain: Docling → PyMuPDF

    Parameters
    ----------
    pdf_path : str | Path | bytes
        Path to the PDF file, or raw bytes.
    prefer_docling : bool
        Whether to prefer Docling over PyMuPDF.

    Returns
    -------
    str
        Extracted text (Markdown if Docling, plain if PyMuPDF).
    """
    # Docling doesn't support bytes, only file paths
    if isinstance(pdf_path, bytes):
        logger.debug("VisionPdfParser: Received bytes — using PyMuPDF (Docling needs file path)")
        return parse_pdf_with_pymupdf(pdf_path)

    if prefer_docling and _check_docling():
        try:
            text = parse_pdf_with_docling(pdf_path)
            if text:
                logger.info(
                    "VisionPdfParser: Docling extracted %d chars from %s",
                    len(text), str(pdf_path)[:60],
                )
                return text
            # Docling returned empty — fall through to PyMuPDF
            logger.warning("VisionPdfParser: Docling returned empty — falling back to PyMuPDF")
        except Exception as exc:
            logger.warning("VisionPdfParser: Docling failed (%s) — falling back to PyMuPDF", exc)

    return parse_pdf_with_pymupdf(pdf_path)


# ── Strategy Pattern Integration ────────────────────────────────

class VisionPdfParserStrategy:
    """ContentParserStrategy compatible with collect_au_sources.py.

    Drop-in replacement for PdfParserStrategy.
    """
    def parse(self, content: Any, source_url: str = "") -> str:
        return parse_pdf(content, prefer_docling=True)
