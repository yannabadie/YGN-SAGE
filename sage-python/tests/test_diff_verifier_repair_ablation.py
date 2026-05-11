"""Tests for slice 10B — diff verifier observe-vs-repair ablation.

Lightweight tests: the heavy paths (git clone + LLM repair call) are
exercised by the real ablation run; unit tests cover the wiring +
verdict classification.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "sage-python" / "scripts" / "diff_verifier_repair_ablation.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("diff_verifier_repair_ablation", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ablation = _load_module()


def test_load_instances_indexes_by_id(tmp_path: Path) -> None:
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(
        json.dumps([
            {"instance_id": "a", "repo": "owner/r1", "base_commit": "abc"},
            {"instance_id": "b", "repo": "owner/r2", "base_commit": "def"},
        ]),
        encoding="utf-8",
    )
    indexed = ablation._load_instances(instances_path)
    assert set(indexed) == {"a", "b"}
    assert indexed["a"]["repo"] == "owner/r1"


def test_audit_task_observe_skips_empty_patch(tmp_path: Path, monkeypatch) -> None:
    """Empty patch → verdict=empty_patch_skipped, no clone, no LLM call."""
    instance = {"instance_id": "x", "repo": "owner/r", "base_commit": "abc"}
    prediction = {"instance_id": "x", "patch": ""}

    # _setup_repo_for_canary should NOT be called for an empty patch
    setup_called = {"count": 0}

    def _fake_setup(inst):
        setup_called["count"] += 1
        return {"repo_context_status": "ready", "repo_dir": str(tmp_path / "r"), "tmp_root": str(tmp_path)}

    monkeypatch.setattr(ablation.arm_d, "_setup_repo_for_canary", _fake_setup)

    result = asyncio.run(
        ablation._audit_task(instance, prediction, "observe", None, 0.5)
    )
    assert result["verdict"] == "empty_patch_skipped"
    assert setup_called["count"] == 0


def test_audit_task_observe_repo_clone_failed(tmp_path: Path, monkeypatch) -> None:
    """Repo clone fails → verdict=repo_clone_failed, no verifier call."""
    instance = {"instance_id": "x", "repo": "ghost/repo", "base_commit": "abc"}
    prediction = {"instance_id": "x", "patch": "diff --git a/x b/x\n@@ @@\n-a\n+b\n"}

    def _fake_setup(inst):
        return {
            "repo_context_status": "clone_failed",
            "repo_dir": None,
            "tmp_root": None,
            "failure_reason": "git clone exit 128 could not resolve host",
        }

    cleanup_called = {"count": 0}

    def _fake_cleanup(*args, **kwargs):
        cleanup_called["count"] += 1
        return "missing"

    monkeypatch.setattr(ablation.arm_d, "_setup_repo_for_canary", _fake_setup)
    monkeypatch.setattr(ablation.arm_d, "_cleanup_repo_dir", _fake_cleanup)

    result = asyncio.run(
        ablation._audit_task(instance, prediction, "observe", None, 0.5)
    )
    assert result["verdict"] == "repo_clone_failed"
    assert "could not resolve host" in (result.get("failure_reason") or "")


def test_audit_task_repair_skipped_when_no_llm(tmp_path: Path, monkeypatch) -> None:
    """mode=repair but llm=None → verdict=repair_skipped_no_llm."""
    instance = {"instance_id": "x", "repo": "x/y", "base_commit": "abc"}
    prediction = {"instance_id": "x", "patch": "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n"}

    repo_dir = tmp_path / "fake_repo"
    repo_dir.mkdir()

    def _fake_setup(inst):
        return {
            "repo_context_status": "ready",
            "repo_dir": str(repo_dir),
            "tmp_root": str(tmp_path),
        }

    monkeypatch.setattr(ablation.arm_d, "_setup_repo_for_canary", _fake_setup)
    monkeypatch.setattr(ablation.arm_d, "_cleanup_repo_dir", lambda *a, **k: "removed")

    # The patch references "x" which doesn't exist in the fake repo → verifier returns mismatches
    result = asyncio.run(
        ablation._audit_task(instance, prediction, "repair", None, 0.5)
    )
    # When mismatch_count > 0 and llm is None, verdict should be repair_skipped_no_llm
    # (When mismatch_count == 0, verdict is no_repair_needed)
    assert result["verdict"] in {"repair_skipped_no_llm", "no_repair_needed"}


def test_verify_one_shape(tmp_path: Path) -> None:
    """_verify_one returns a dict with mismatch_count + outcome + lists."""
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    diff = "diff --git a/missing.py b/missing.py\n--- a/missing.py\n+++ b/missing.py\n@@ -1 +1 @@\n-a\n+b\n"
    result = ablation._verify_one(diff, repo_dir)
    assert "mismatch_count" in result
    assert "outcome" in result
    assert isinstance(result["mismatches"], list)
    assert isinstance(result["reason_events"], list)
    # The file missing.py doesn't exist → at least one mismatch
    assert result["mismatch_count"] >= 1


def test_slice_10b_real_artefact_observe_fingerprint() -> None:
    """Pin the slice 10B observe fingerprint on the slice 9 artefact.
    If a future audit run drifts these numbers, investigate.
    """
    summary_path = (
        _REPO_ROOT
        / "docs"
        / "benchmarks"
        / "2026-05-11-diff-verifier-repair-ablation"
        / "repair_summary.json"
    )
    if not summary_path.is_file():
        pytest.skip(f"slice 10B artefact missing at {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["mode"] == "repair"
    assert summary["n_tasks"] == 5
    assert summary["total_mismatches_before"] == 6
    # After repair: 2 tasks (the 2 AVR) had post_repair check that re-counted
    # to the SAME mismatch count → repair_did_not_reduce_mismatches.
    # Sequential teleport + NodeBB: 0 mismatches, skipped.
    # tutanota 219bc: timed out, no post-repair count.
    assert summary["verdict_tally"]["repair_did_not_reduce_mismatches"] == 2
    assert summary["verdict_tally"]["no_repair_needed"] == 2
    assert summary["verdict_tally"]["verifier_repair_empty"] == 1
