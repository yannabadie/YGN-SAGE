"""Tests for the migration module: NotebookLM markdown bootstrap to ExoCortex."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from discover.migration import (
    DEFAULT_MIGRATION_DIR,
    NOTEBOOK_DOMAINS,
    extract_arxiv_ids,
    migrate_markdown,
    migrate_notebooks,
)


# ---------------------------------------------------------------------------
# extract_arxiv_ids
# ---------------------------------------------------------------------------


def test_extract_arxiv_ids_from_text():
    """Text containing 3 different arXiv ID formats should find all 3 unique IDs."""
    text = """
    Check out arXiv:2503.01234 for the first result.
    Also see https://arxiv.org/abs/2401.56789 for related work.
    The version-tagged paper 2310.07842v2 is also relevant.
    """
    ids = extract_arxiv_ids(text)
    assert len(ids) == 3
    assert "2503.01234" in ids
    assert "2401.56789" in ids
    assert "2310.07842" in ids
    # Should be sorted
    assert ids == sorted(ids)


def test_extract_arxiv_ids_empty():
    """Text with no arXiv IDs returns empty list."""
    text = "This is a normal paragraph with no paper references at all."
    ids = extract_arxiv_ids(text)
    assert ids == []


# ---------------------------------------------------------------------------
# migrate_markdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_migrate_markdown_uploads_to_exocortex(tmp_path: Path):
    """Create temp md file, mock exocortex, verify upload called with correct args."""
    md_file = tmp_path / "technical.md"
    md_file.write_text("# Technical Notes\n\nSome content about architectures.")

    exocortex = AsyncMock()
    await migrate_markdown(md_file, exocortex, domain="cognitive_architectures")

    exocortex.upload.assert_called_once_with(
        file_path=str(md_file),
        display_name="[Migration] technical",
    )


# ---------------------------------------------------------------------------
# migrate_notebooks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_migrate_notebooks_skips_missing_dir():
    """Nonexistent migration directory returns 0 with no error."""
    exocortex = AsyncMock()
    nonexistent = Path("/tmp/sage_test_nonexistent_dir_12345")
    # Ensure it really doesn't exist
    assert not nonexistent.exists()

    count = await migrate_notebooks(exocortex, migration_dir=nonexistent)

    assert count == 0
    exocortex.upload.assert_not_called()


@pytest.mark.asyncio
async def test_migrate_notebooks_processes_all_md_files(tmp_path: Path):
    """All .md files in migration dir should be processed."""
    (tmp_path / "technical.md").write_text("# Tech notes")
    (tmp_path / "discover_ai.md").write_text("# Discover AI")
    (tmp_path / "random.md").write_text("# Random")
    (tmp_path / "not_markdown.txt").write_text("ignore me")

    exocortex = AsyncMock()
    count = await migrate_notebooks(exocortex, migration_dir=tmp_path)

    assert count == 3
    assert exocortex.upload.call_count == 3


def test_default_migration_dir():
    """DEFAULT_MIGRATION_DIR is under ~/.sage/migration."""
    assert DEFAULT_MIGRATION_DIR == Path.home() / ".sage" / "migration"


def test_notebook_domains_mapping():
    """NOTEBOOK_DOMAINS maps known stems to domain tags."""
    assert NOTEBOOK_DOMAINS["technical"] == "cognitive_architectures"
    assert NOTEBOOK_DOMAINS["exocortex"] == "cognitive_architectures"
    assert NOTEBOOK_DOMAINS["ygn"] == "evolutionary_computation"
    assert NOTEBOOK_DOMAINS["discover"] == "marl"
    assert NOTEBOOK_DOMAINS["discover_ai"] == "marl"
