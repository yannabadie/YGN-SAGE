"""tests/test_frontier.py — MAP-Elites frontier explorer tests."""
from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from discover.frontier import (
    FrontierDescriptor,
    FrontierArchive,
    FrontierExplorer,
)


def test_descriptor_dataclass():
    d = FrontierDescriptor(domain_idx=0, recency=0.1, citation_velocity=0.5, novelty=0.8)
    assert d.domain_idx == 0


def test_archive_insert():
    archive = FrontierArchive(bins=[5, 4, 4, 3])
    desc = FrontierDescriptor(domain_idx=0, recency=0.1, citation_velocity=0.5, novelty=0.8)
    inserted = archive.try_insert("p1", desc, fitness=0.9)
    assert inserted is True
    assert archive.size() == 1


def test_archive_replaces_lower_fitness():
    archive = FrontierArchive(bins=[5, 4, 4, 3])
    desc = FrontierDescriptor(domain_idx=0, recency=0.1, citation_velocity=0.5, novelty=0.8)
    archive.try_insert("p1", desc, fitness=0.5)
    replaced = archive.try_insert("p2", desc, fitness=0.9)
    assert replaced is True
    assert archive.size() == 1


def test_archive_rejects_lower_fitness():
    archive = FrontierArchive(bins=[5, 4, 4, 3])
    desc = FrontierDescriptor(domain_idx=0, recency=0.1, citation_velocity=0.5, novelty=0.8)
    archive.try_insert("p1", desc, fitness=0.9)
    rejected = archive.try_insert("p2", desc, fitness=0.3)
    assert rejected is False


def test_archive_coverage():
    archive = FrontierArchive(bins=[5, 4, 4, 3])
    total_cells = 5 * 4 * 4 * 3
    desc = FrontierDescriptor(domain_idx=0, recency=0.1, citation_velocity=0.5, novelty=0.8)
    archive.try_insert("p1", desc, fitness=0.9)
    coverage = archive.coverage()
    assert abs(coverage - 1.0 / total_cells) < 0.001


def test_archive_get_empty_cells():
    archive = FrontierArchive(bins=[2, 2, 2, 2])
    total = 2 * 2 * 2 * 2
    desc = FrontierDescriptor(domain_idx=0, recency=0.0, citation_velocity=0.0, novelty=0.0)
    archive.try_insert("p1", desc, fitness=0.9)
    empty = archive.get_empty_cells()
    assert len(empty) == total - 1


def test_compute_descriptor():
    explorer = FrontierExplorer.__new__(FrontierExplorer)
    explorer._archive = FrontierArchive(bins=[5, 4, 4, 3])
    explorer._store = MagicMock()
    explorer._store.search_dense.return_value = []
    desc = explorer._compute_descriptor(
        domain="marl",
        published=date.today(),
        citation_count=10,
        paper_embedding=np.random.rand(768).astype(np.float32),
    )
    assert 0 <= desc.domain_idx <= 4
    assert 0.0 <= desc.recency <= 1.0
    assert 0.0 <= desc.novelty <= 1.0


@pytest.mark.asyncio
async def test_explorer_seed():
    mock_store = MagicMock()
    mock_store.search_dense.return_value = [
        {"id": "p1", "score": 0.9, "payload": {
            "title": "A", "abstract": "B", "domain": "marl",
            "year": 2025, "citation_count": 5, "_paper_id": "p1",
        }},
    ]
    mock_embedder = MagicMock()
    mock_embedder.embed_text.return_value = np.random.rand(768).astype(np.float32)

    explorer = FrontierExplorer(store=mock_store, embedder=mock_embedder)
    await explorer.seed()
    assert explorer._archive.size() >= 1


def test_archive_cell_to_descriptor():
    archive = FrontierArchive(bins=[5, 4, 4, 3])
    desc = archive.cell_to_descriptor((2, 1, 3, 0))
    assert desc.domain_idx == 2
    assert 0.0 <= desc.recency <= 1.0
