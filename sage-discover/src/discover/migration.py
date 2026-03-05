"""Migration module: bootstrap ExoCortex from NotebookLM markdown exports.

Scans a migration directory for exported markdown files, extracts arXiv IDs,
determines domain tags, and uploads to ExoCortex for persistent RAG indexing.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MIGRATION_DIR: Path = Path.home() / ".sage" / "migration"

# ---------------------------------------------------------------------------
# Domain mapping
# ---------------------------------------------------------------------------

NOTEBOOK_DOMAINS: dict[str, str] = {
    "technical": "cognitive_architectures",
    "exocortex": "cognitive_architectures",
    "ygn": "evolutionary_computation",
    "discover": "marl",
    "discover_ai": "marl",
}

# ---------------------------------------------------------------------------
# arXiv ID extraction
# ---------------------------------------------------------------------------

# Patterns:
#   arXiv:2503.01234       (explicit prefix)
#   arxiv.org/abs/2503.01234  (URL form)
#   2503.01234v2           (bare ID with optional version)
_ARXIV_PATTERNS = [
    re.compile(r"arXiv:(\d{4}\.\d{4,5})", re.IGNORECASE),
    re.compile(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", re.IGNORECASE),
    re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)", re.IGNORECASE),
]


def extract_arxiv_ids(text: str) -> list[str]:
    """Extract unique arXiv IDs from *text*, stripping version suffixes.

    Matches patterns: ``arXiv:2503.01234``, ``arxiv.org/abs/2503.01234``,
    ``2503.01234v2``.

    Returns a sorted list of unique IDs (without version suffix).
    """
    ids: set[str] = set()
    for pattern in _ARXIV_PATTERNS:
        for match in pattern.finditer(text):
            ids.add(match.group(1))
    return sorted(ids)


# ---------------------------------------------------------------------------
# Single file migration
# ---------------------------------------------------------------------------


async def migrate_markdown(
    md_path: Path,
    exocortex: Any,
    domain: str | None = None,
) -> None:
    """Upload a single markdown file to ExoCortex.

    Parameters
    ----------
    md_path : Path
        Path to the markdown file.
    exocortex : Any
        ExoCortex instance with an ``upload(file_path, display_name)`` method.
    domain : str | None
        Optional domain tag. If None, derived from NOTEBOOK_DOMAINS or
        defaults to ``"general"``.
    """
    if not md_path.exists():
        logger.warning("Migration file not found: %s", md_path)
        return

    if domain is None:
        stem = md_path.stem.lower()
        domain = NOTEBOOK_DOMAINS.get(stem, "general")

    display_name = f"[Migration] {md_path.stem}"
    logger.info("Migrating %s (domain=%s)", md_path.name, domain)
    await exocortex.upload(file_path=str(md_path), display_name=display_name)


# ---------------------------------------------------------------------------
# Batch migration
# ---------------------------------------------------------------------------


async def migrate_notebooks(
    exocortex: Any,
    migration_dir: Path = DEFAULT_MIGRATION_DIR,
) -> int:
    """Migrate all ``*.md`` files from *migration_dir* to ExoCortex.

    Parameters
    ----------
    exocortex : Any
        ExoCortex instance with an ``upload(file_path, display_name)`` method.
    migration_dir : Path
        Directory containing exported markdown files.

    Returns
    -------
    int
        Number of files migrated.
    """
    if not migration_dir.exists():
        logger.info("Migration directory does not exist: %s", migration_dir)
        return 0

    md_files = sorted(migration_dir.glob("*.md"))
    if not md_files:
        logger.info("No markdown files found in %s", migration_dir)
        return 0

    count = 0
    for md_path in md_files:
        stem = md_path.stem.lower()
        domain = NOTEBOOK_DOMAINS.get(stem)
        await migrate_markdown(md_path, exocortex, domain=domain)
        count += 1

    logger.info("Migrated %d markdown files from %s", count, migration_dir)
    return count
