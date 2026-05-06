"""Tests for `scripts/sync_doc_counters.py`.

Cycle-13 K Phase 0.2: validates the propagation rule from
`docs/status/current.json` into the README badge, AI-ARCHITECTURE.md
header, and `.claude/rules/architecture.md`. The contract is unidirectional
— current.json is source of truth; the other docs are downstream.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_sync_module() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "sync_doc_counters.py"
    if not script_path.is_file():
        pytest.skip(f"{script_path} not present")
    spec = importlib.util.spec_from_file_location("sync_doc_counters", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sync_doc_counters"] = mod
    spec.loader.exec_module(mod)
    return mod


def _build_repo(tmp_path: Path, *, py: int, rust: int, disc: int) -> Path:
    """Lay out a minimal repo skeleton under tmp_path for sync tests."""
    (tmp_path / "docs" / "status").mkdir(parents=True)
    (tmp_path / "docs" / "status" / "current.json").write_text(
        json.dumps(
            {
                "schema_version": "v1",
                "git": {"branch": "main", "commit_sha": "deadbee", "dirty": False},
                "sage_core_tests": {"available": True, "total": rust},
                "sage_python_tests": {"available": True, "total": py},
                "sage_discover_tests": {"available": True, "total": disc},
            }
        ),
        encoding="utf-8",
    )

    (tmp_path / "README.md").write_text(
        '<img src="https://img.shields.io/badge/tests-1%20Py%20%2B%202%20Rust-brightgreen" alt="Tests">\n'
        "Some other text...\n",
        encoding="utf-8",
    )

    (tmp_path / "AI-ARCHITECTURE.md").write_text(
        "| Attribut | Valeur |\n"
        "|---|---|\n"
        "| **Tests Python collected** | **3** (canonical : `docs/status/current.json`) |\n"
        "| **Tests Rust** | **4** (`cargo test --features smt`) |\n"
        "| **Tests sage-discover** | **5** |\n",
        encoding="utf-8",
    )

    (tmp_path / ".claude" / "rules").mkdir(parents=True)
    (tmp_path / ".claude" / "rules" / "architecture.md").write_text(
        "## Project Structure\n"
        "- `sage-core/` — Rust orchestrator (PyO3). **6 tests** with `--features smt`.\n"
        "- `sage-python/` — Python SDK. **7 collected** (canonical: ...).\n"
        "- `sage-discover/` — Knowledge pipeline (arXiv → ExoCortex). **8 tests**.\n",
        encoding="utf-8",
    )

    return tmp_path


def test_sync_writes_counters_into_all_three_docs(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path, py=3114, rust=553, disc=100)
    sync_mod = _load_sync_module()

    rc = sync_mod.main(["--repo-root", str(repo)])
    assert rc == 0

    readme = (repo / "README.md").read_text(encoding="utf-8")
    assert "tests-3114%20Py%20%2B%20553%20Rust" in readme

    ai_arch = (repo / "AI-ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "**Tests Python collected** | **3114**" in ai_arch
    assert "**Tests Rust** | **553**" in ai_arch
    assert "**Tests sage-discover** | **100**" in ai_arch

    rules = (repo / ".claude" / "rules" / "architecture.md").read_text(encoding="utf-8")
    assert "**553 tests**" in rules
    assert "**3114 collected**" in rules
    assert "**100 tests**" in rules


def test_check_mode_passes_when_in_sync(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path, py=3114, rust=553, disc=100)
    sync_mod = _load_sync_module()
    sync_mod.main(["--repo-root", str(repo)])

    rc = sync_mod.main(["--check", "--repo-root", str(repo)])
    assert rc == 0


def test_check_mode_fails_on_drift(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path, py=3114, rust=553, disc=100)
    sync_mod = _load_sync_module()
    # README starts at "1 Py + 2 Rust" — drift vs current.json's 3114 / 553.
    rc = sync_mod.main(["--check", "--repo-root", str(repo)])
    assert rc == 1


def test_drift_detection_reports_each_doc_separately(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path, py=3114, rust=553, disc=100)
    sync_mod = _load_sync_module()
    ok, drifts = sync_mod.sync(repo, check_only=True)
    assert ok is False
    files_seen = {d.file for d in drifts}
    assert files_seen == {"README.md", "AI-ARCHITECTURE.md", ".claude/rules/architecture.md"}


def test_write_mode_is_idempotent(tmp_path: Path) -> None:
    repo = _build_repo(tmp_path, py=3114, rust=553, disc=100)
    sync_mod = _load_sync_module()
    sync_mod.main(["--repo-root", str(repo)])
    first_readme = (repo / "README.md").read_text(encoding="utf-8")
    sync_mod.main(["--repo-root", str(repo)])
    second_readme = (repo / "README.md").read_text(encoding="utf-8")
    assert first_readme == second_readme


def test_json_output_is_valid(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _build_repo(tmp_path, py=3114, rust=553, disc=100)
    sync_mod = _load_sync_module()
    sync_mod.main(["--check", "--json", "--repo-root", str(repo)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["mode"] == "check"
    assert isinstance(payload["drifts"], list)


def test_real_repo_after_sync_has_zero_drift() -> None:
    """After Phase 0.2 ships, --check on the real repo MUST exit 0."""
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / "docs" / "status" / "current.json").is_file():
        pytest.skip("current.json not present in this checkout")
    sync_mod = _load_sync_module()
    ok, drifts = sync_mod.sync(repo_root, check_only=True)
    assert ok, f"Real repo has counter drift after Phase 0.2 sync: {drifts}"


# ---------------------------------------------------------------------------
# Phase 0.3 — invariant count from runtime-integrity-ledger.md
# ---------------------------------------------------------------------------


def _build_repo_with_ledger(
    tmp_path: Path, *, heading_count: int, table_rows: int,
) -> Path:
    """Lay out a minimal repo with current.json + a fake ledger.

    `heading_count` populates the `## The N invariants` heading line,
    `table_rows` controls how many `| **Label** | ... |` rows follow.
    """
    repo = _build_repo(tmp_path, py=100, rust=10, disc=5)
    ledger_dir = repo / "docs" / "contracts"
    ledger_dir.mkdir(parents=True)
    body_rows = "\n".join(
        f"| **Invariant {i + 1}** | declared | verified | side-effect |"
        for i in range(table_rows)
    )
    (ledger_dir / "runtime-integrity-ledger.md").write_text(
        f"# Runtime Integrity Ledger\n\n"
        f"## The {heading_count} invariants\n\n"
        f"| Invariant | Declared label | Verified content | Side-effect |\n"
        f"|---|---|---|---|\n"
        f"{body_rows}\n\n"
        f"## Module cross-reference\n\n"
        f"placeholder\n",
        encoding="utf-8",
    )
    return repo


def test_load_invariant_count_reads_ledger_heading(tmp_path: Path) -> None:
    repo = _build_repo_with_ledger(tmp_path, heading_count=9, table_rows=9)
    sync_mod = _load_sync_module()
    assert sync_mod.load_invariant_count(repo) == 9


def test_load_invariant_count_raises_on_heading_table_mismatch(tmp_path: Path) -> None:
    repo = _build_repo_with_ledger(tmp_path, heading_count=9, table_rows=8)
    sync_mod = _load_sync_module()
    with pytest.raises(ValueError, match="9 invariants but section"):
        sync_mod.load_invariant_count(repo)


def test_invariant_count_propagates_into_readme(tmp_path: Path) -> None:
    repo = _build_repo_with_ledger(tmp_path, heading_count=9, table_rows=9)
    # README pre-state: says "8 invariants" — sync should bump to 9.
    (repo / "README.md").write_text(
        "Some text\n8 invariants in the ledger.\nOther text.\n",
        encoding="utf-8",
    )
    sync_mod = _load_sync_module()
    sync_mod.main(["--repo-root", str(repo)])
    out = (repo / "README.md").read_text(encoding="utf-8")
    assert "9 invariants" in out
    assert "8 invariants" not in out


def test_bump_commit_sha_writes_v2_schema_and_two_fields(tmp_path: Path) -> None:
    """Phase 0.1b: --bump-commit-sha sets schema_version=v2 + two SHA fields."""
    repo = _build_repo(tmp_path, py=100, rust=10, disc=5)
    sync_mod = _load_sync_module()
    sync_mod.bump_commit_sha(repo, "abc123def456")
    payload = json.loads((repo / "docs" / "status" / "current.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == "v2"
    assert payload["git"]["snapshot_commit_sha"] == "abc123def456"
    assert payload["git"]["generated_for_commit_sha"] == "abc123def456"
    # commit_sha kept for v1 backward-compat.
    assert payload["git"]["commit_sha"] == "abc123def456"


def test_bump_commit_sha_preserves_other_payload_fields(tmp_path: Path) -> None:
    """The bump operation must not destructively edit unrelated content."""
    repo = _build_repo(tmp_path, py=100, rust=10, disc=5)
    sync_mod = _load_sync_module()
    # Inject extra unrelated fields.
    payload_path = repo / "docs" / "status" / "current.json"
    p = json.loads(payload_path.read_text(encoding="utf-8"))
    p["sage_python_tests"]["extra_meta"] = "preserve me"
    p["custom_top_level"] = ["a", "b"]
    payload_path.write_text(json.dumps(p), encoding="utf-8")

    sync_mod.bump_commit_sha(repo, "deadbeef")
    after = json.loads(payload_path.read_text(encoding="utf-8"))
    assert after["sage_python_tests"]["extra_meta"] == "preserve me"
    assert after["custom_top_level"] == ["a", "b"]
    assert after["git"]["snapshot_commit_sha"] == "deadbeef"


def test_main_with_bump_commit_sha_flag(tmp_path: Path) -> None:
    """--bump-commit-sha CLI flag also runs the regular sync pass."""
    repo = _build_repo(tmp_path, py=42, rust=5, disc=3)
    sync_mod = _load_sync_module()
    rc = sync_mod.main(["--repo-root", str(repo), "--bump-commit-sha", "feedfaceb000"])
    assert rc == 0
    payload = json.loads((repo / "docs" / "status" / "current.json").read_text(encoding="utf-8"))
    assert payload["git"]["snapshot_commit_sha"] == "feedfaceb000"
    # And README badge got updated too (proves the regular sync ran).
    readme = (repo / "README.md").read_text(encoding="utf-8")
    assert "tests-42%20Py%20%2B%205%20Rust" in readme


def test_invariant_count_does_not_touch_claude_md(tmp_path: Path) -> None:
    """CLAUDE.md is INTENTIONALLY excluded — historical references like
    `5 invariants at cycle-8 closure` must NOT be rewritten."""
    repo = _build_repo_with_ledger(tmp_path, heading_count=9, table_rows=9)
    claude_text = (
        "Pre-cycle-8 had 5 invariants.\n"
        "Cycle-9 added the 7th.\n"
        "Current ledger total is 9.\n"
    )
    (repo / "CLAUDE.md").write_text(claude_text, encoding="utf-8")
    sync_mod = _load_sync_module()
    sync_mod.main(["--repo-root", str(repo)])
    after = (repo / "CLAUDE.md").read_text(encoding="utf-8")
    # Bytes-identical: CLAUDE.md isn't in _DOC_TARGETS so it MUST be untouched.
    assert after == claude_text
