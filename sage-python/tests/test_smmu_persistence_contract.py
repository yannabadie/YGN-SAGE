"""S-MMU persistence contract — Phase 3 (AUDITRUST.md).

Verifies that MultiViewMMU.save_json / load_json roundtrip is
retrieval-equivalent at the Python boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_smmu_save_load_roundtrip_retrieval_equivalent(tmp_path: Path) -> None:
    """save_json → load_json must preserve retrieval equivalence."""
    import sage_core

    smmu = sage_core.MultiViewMMU()
    a = smmu.register_chunk(
        0, 1, "rust memory graph",
        ["rust", "memory"],
        [1.0, 0.0, 0.5],
        None,
    )
    smmu.register_chunk(
        2, 3, "rust graph verifier",
        ["rust", "graph"],
        [1.0, 0.0, 0.5],
        None,
    )

    before = smmu.retrieve_relevant(a, 2)

    path = tmp_path / "smmu.json"
    smmu.save_json(str(path))

    restored = sage_core.MultiViewMMU.load_json(str(path))
    after = restored.retrieve_relevant(a, 2)

    assert after == before, (
        f"retrieval equivalence broken: "
        f"before={before!r}, after={after!r}"
    )
    assert restored.chunk_count() == 2


def test_smmu_save_json_creates_valid_json(tmp_path: Path) -> None:
    """save_json output must be parseable as valid JSON."""
    import sage_core

    smmu = sage_core.MultiViewMMU()
    smmu.register_chunk(0, 1, "test", ["test"], None, None)

    path = tmp_path / "smmu.json"
    smmu.save_json(str(path))

    raw = path.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert "version" in parsed
    assert parsed["version"] == 1
    assert "chunks" in parsed
    assert len(parsed["chunks"]) == 1
    assert "edges" in parsed
    assert "recent_ids" in parsed


def test_smmu_empty_roundtrip(tmp_path: Path) -> None:
    """Empty S-MMU must survive save/load roundtrip."""
    import sage_core

    smmu = sage_core.MultiViewMMU()
    assert smmu.chunk_count() == 0

    path = tmp_path / "empty_smmu.json"
    smmu.save_json(str(path))

    restored = sage_core.MultiViewMMU.load_json(str(path))
    assert restored.chunk_count() == 0


def test_smmu_load_nonexistent_raises(tmp_path: Path) -> None:
    """load_json on a nonexistent file must raise IOError."""
    import sage_core

    path = tmp_path / "does_not_exist.json"
    with pytest.raises(OSError):
        sage_core.MultiViewMMU.load_json(str(path))
