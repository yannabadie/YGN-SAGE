"""src/discover/extractor.py — PDF-to-structured-text via Docling."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from docling.document_converter import DocumentConverter
    HAS_DOCLING = True
except ImportError:
    DocumentConverter = None
    HAS_DOCLING = False

_SECTION_PATTERNS = {
    "introduction": re.compile(r"^#+\s*(introduction|1\.\s*introduction)", re.IGNORECASE),
    "methodology": re.compile(r"^#+\s*(method|methodology|approach|2\.\s*method)", re.IGNORECASE),
    "results": re.compile(r"^#+\s*(results|experiments|evaluation|3\.\s*results)", re.IGNORECASE),
    "conclusion": re.compile(r"^#+\s*(conclusion|discussion|summary|4\.\s*conclusion)", re.IGNORECASE),
}


def extract_sections_from_markdown(md: str) -> dict[str, str | None]:
    """Extract named sections from markdown text."""
    sections: dict[str, str | None] = {k: None for k in _SECTION_PATTERNS}
    lines = md.split("\n")
    current_section: str | None = None
    current_lines: list[str] = []

    def _flush():
        nonlocal current_section, current_lines
        if current_section and current_lines:
            sections[current_section] = "\n".join(current_lines).strip()
        current_lines = []

    for line in lines:
        matched = False
        for section_name, pattern in _SECTION_PATTERNS.items():
            if pattern.match(line.strip()):
                _flush()
                current_section = section_name
                matched = True
                break
        if not matched and current_section:
            if line.strip().startswith("#"):
                _flush()
                current_section = None
            else:
                current_lines.append(line)

    _flush()
    return sections


def extract_full_text(pdf_path: Path) -> dict[str, Any]:
    """Extract structured content from PDF using Docling."""
    if not HAS_DOCLING:
        return {
            "full_text": None,
            "sections": {k: None for k in _SECTION_PATTERNS},
            "tables": [],
            "error": "docling not installed",
        }

    try:
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        doc = result.document
        md = doc.export_to_markdown()
        sections = extract_sections_from_markdown(md)
        tables = []
        if hasattr(doc, "tables"):
            for t in doc.tables:
                try:
                    tables.append(t.export_to_markdown())
                except Exception:
                    pass
        return {"full_text": md, "sections": sections, "tables": tables, "error": None}
    except Exception as e:
        logger.warning("PDF extraction failed for %s: %s", pdf_path, e)
        return {
            "full_text": None,
            "sections": {k: None for k in _SECTION_PATTERNS},
            "tables": [],
            "error": str(e),
        }
