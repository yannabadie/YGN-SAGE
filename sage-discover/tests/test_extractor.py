"""tests/test_extractor.py — Docling PDF extractor tests."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from discover.extractor import extract_full_text, extract_sections_from_markdown


def test_extract_sections_from_markdown():
    md = """# Introduction
This is the introduction.

## Methodology
We use method X.

## Results
We found Y.

## Conclusion
In conclusion Z.

## References
[1] Ref A
"""
    sections = extract_sections_from_markdown(md)
    assert "introduction" in sections
    assert "methodology" in sections
    assert "results" in sections
    assert "conclusion" in sections
    assert "We use method X." in sections["methodology"]


def test_extract_sections_handles_missing():
    md = "Just a plain abstract with no sections."
    sections = extract_sections_from_markdown(md)
    assert sections["introduction"] is None
    assert sections["methodology"] is None


@patch("discover.extractor.DocumentConverter")
def test_extract_full_text_returns_structured(mock_converter_cls):
    mock_result = MagicMock()
    mock_result.document.export_to_markdown.return_value = "# Title\nContent here"
    mock_result.document.tables = []
    mock_result.document.pictures = []
    mock_converter = MagicMock()
    mock_converter.convert.return_value = mock_result
    mock_converter_cls.return_value = mock_converter

    result = extract_full_text(Path("/fake/paper.pdf"))
    assert "full_text" in result
    assert "sections" in result
    assert "tables" in result


@patch("discover.extractor.DocumentConverter")
def test_extract_full_text_fallback_on_error(mock_converter_cls):
    mock_converter_cls.return_value.convert.side_effect = Exception("PDF corrupted")
    result = extract_full_text(Path("/fake/bad.pdf"))
    assert result["full_text"] is None
    assert result["error"] is not None
