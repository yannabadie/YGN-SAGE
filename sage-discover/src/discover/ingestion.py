"""Ingestion module: upload curated papers to ExoCortex + local manifest tracking.

Downloads PDFs, uploads them to the ExoCortex (Google GenAI File Search API),
and records each ingested paper in a local JSON manifest to avoid duplicates.
"""
from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from discover.curator import CuratedPaper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MANIFEST_PATH: Path = Path.home() / ".sage" / "manifest.json"
DEFAULT_PAPERS_DIR: Path = Path.home() / ".sage" / "papers"

# ---------------------------------------------------------------------------
# Manifest dataclass
# ---------------------------------------------------------------------------


@dataclass
class Manifest:
    """Local tracking manifest for ingested papers."""

    store_name: str = ""
    papers: dict[str, dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> Manifest:
    """Load manifest from JSON file. Return empty Manifest if missing or invalid."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Manifest(
            store_name=data.get("store_name", ""),
            papers=data.get("papers", {}),
        )
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return Manifest()


def save_manifest(manifest: Manifest, path: Path = DEFAULT_MANIFEST_PATH) -> None:
    """Save manifest to JSON, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "store_name": manifest.store_name,
        "papers": manifest.papers,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def is_already_ingested(paper_id: str, manifest: Manifest) -> bool:
    """Check whether *paper_id* is already recorded in the manifest."""
    return paper_id in manifest.papers


# ---------------------------------------------------------------------------
# PDF download
# ---------------------------------------------------------------------------


async def download_pdf(url: str, dest: Path) -> bool:
    """Download a PDF from *url* to *dest* via urllib. Skip if dest exists.

    Uses ``asyncio.to_thread`` to avoid blocking the event loop.
    Returns True on success, False on failure.
    """
    if dest.exists():
        logger.debug("PDF already exists: %s", dest)
        return True

    def _download() -> bool:
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, str(dest))
            logger.info("Downloaded PDF: %s -> %s", url, dest)
            return True
        except Exception as exc:
            logger.error("PDF download failed (%s): %s", url, exc)
            return False

    return await asyncio.to_thread(_download)


# ---------------------------------------------------------------------------
# Ingest single paper
# ---------------------------------------------------------------------------


async def ingest(
    paper: CuratedPaper,
    exocortex: Any,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> bool:
    """Ingest a single curated paper into ExoCortex.

    1. Load manifest and check for duplicates (skip if already ingested).
    2. Download PDF if ``paper.pdf_path`` is None and ``pdf_url`` is available.
    3. Upload to ExoCortex via ``exocortex.upload(file_path, display_name)``.
    4. Record paper metadata in manifest and save.

    Returns True if the paper was newly ingested, False if skipped.
    """
    manifest = load_manifest(manifest_path)
    pid = paper.candidate.paper_id

    if is_already_ingested(pid, manifest):
        logger.debug("Already ingested, skipping: %s", pid)
        return False

    # Resolve PDF path
    file_path = paper.pdf_path
    if file_path is None and paper.candidate.pdf_url:
        # Derive filename from paper_id
        safe_name = pid.replace("/", "_").replace(":", "_") + ".pdf"
        dest = DEFAULT_PAPERS_DIR / safe_name
        ok = await download_pdf(paper.candidate.pdf_url, dest)
        if ok:
            file_path = dest

    if file_path is None:
        logger.warning("No PDF available for paper %s, skipping upload", pid)
        return False

    # Upload to ExoCortex
    try:
        display_name = f"{paper.candidate.title} [{pid}]"
        await exocortex.upload(str(file_path), display_name)
    except Exception as exc:
        logger.error("ExoCortex upload failed for %s: %s", pid, exc)
        return False

    # Update manifest
    manifest.papers[pid] = {
        "title": paper.candidate.title,
        "domain": paper.candidate.domain,
        "source": paper.candidate.source,
        "relevance_score": paper.relevance_score,
        "ingested_at": datetime.utcnow().isoformat(),
    }
    save_manifest(manifest, manifest_path)
    logger.info("Ingested paper: %s", pid)
    return True


# ---------------------------------------------------------------------------
# Ingest batch
# ---------------------------------------------------------------------------


async def ingest_all(
    papers: list[CuratedPaper],
    exocortex: Any,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> int:
    """Ingest a list of curated papers. Returns count of newly ingested."""
    count = 0
    for paper in papers:
        if await ingest(paper, exocortex, manifest_path):
            count += 1
    return count
