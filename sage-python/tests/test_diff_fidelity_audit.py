"""Tests for sage-python/scripts/diff_fidelity_audit.py.

Slice 10C (cgpro VERIFY 2026-05-11 RF#B MODIFY). Covers:

- ``_extract_patch`` mirrors swebench_bench's behaviour (raw, fenced,
  embedded, sentinel rejection).
- ``_classify_fidelity`` returns the right verdict for the 4 canonical
  cases (preserved / cosmetic_drift / files_altered / rewritten).
- ``audit_task`` reads events.jsonl + predictions.json and produces a
  well-shaped per-task record.
- A real-canary smoke pins the slice 9 artifact's verdict_tally —
  if a future run drifts this fingerprint, this test catches it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "sage-python" / "scripts" / "diff_fidelity_audit.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("diff_fidelity_audit", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


audit_mod = _load_module()


def test_extract_patch_sentinel_returns_empty() -> None:
    assert audit_mod._extract_patch("[sage: agent exited after 5 steps with no content]") == ""


def test_extract_patch_raw_diff() -> None:
    raw = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
    extracted = audit_mod._extract_patch(raw)
    assert "diff --git" in extracted


def test_extract_patch_fenced_diff() -> None:
    raw = (
        "Reasoning before the diff.\n\n"
        "```diff\n"
        "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
        "```\n\n"
        "Reasoning after."
    )
    extracted = audit_mod._extract_patch(raw)
    assert "diff --git" in extracted
    assert "Reasoning before" not in extracted
    assert "Reasoning after" not in extracted


def test_diff_files_lists_paths_in_order() -> None:
    diff = (
        "diff --git a/src/a.py b/src/a.py\n"
        "--- a/src/a.py\n"
        "+++ b/src/a.py\n"
        "@@ -1 +1 @@\n-x\n+y\n"
        "diff --git a/src/b.py b/src/b.py\n"
        "--- a/src/b.py\n"
        "+++ b/src/b.py\n"
        "@@ -1 +1 @@\n-x\n+y\n"
    )
    assert audit_mod._diff_files(diff) == ["src/a.py", "src/b.py"]


def test_hunk_count() -> None:
    diff = "diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n@@ -5 +5 @@\n-c\n+d\n"
    assert audit_mod._hunk_count(diff) == 2


def test_classify_fidelity_preserved() -> None:
    pre = "diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n"
    assert audit_mod._classify_fidelity(pre, pre, ["x"], ["x"]) == "preserved"


def test_classify_fidelity_cosmetic_drift() -> None:
    pre = "diff --git a/x b/x\n" + ("@@ -1 +1 @@\n-a\n+b\n" * 50)  # ~700 chars
    post = pre[:-5]  # 5 chars shorter (<5% of 700) — trailing whitespace
    assert audit_mod._classify_fidelity(pre, post, ["x"], ["x"]) == "cosmetic_drift"


def test_classify_fidelity_files_altered() -> None:
    pre = "diff --git a/x b/x\n@@ -1 +1 @@\n-a\n+b\n"
    post = "diff --git a/x b/x\n@@ -1,100 +1,100 @@\n" + ("-line\n+new\n" * 50)
    # Same file, content shifted significantly (more than 5% delta)
    assert audit_mod._classify_fidelity(pre, post, ["x"], ["x"]) == "files_altered"


def test_classify_fidelity_rewritten_disjoint_files() -> None:
    """The keystone case from the slice 9 artifact: pre and post touch
    ENTIRELY DIFFERENT FILES. This is the path-hallucination signature.
    """
    pre = "diff --git a/packages/foo.tsx b/packages/foo.tsx\n@@ @@\n-a\n+b\n"
    post = "diff --git a/src/app/foo.tsx b/src/app/foo.tsx\n@@ @@\n-a\n+b\n"
    pre_files = ["packages/foo.tsx"]
    post_files = ["src/app/foo.tsx"]
    assert audit_mod._classify_fidelity(pre, post, pre_files, post_files) == "rewritten"


def test_classify_fidelity_partial_overlap_is_files_altered() -> None:
    """post touches a.py + c.py (kept a.py, dropped b.py, added c.py).
    Texts MUST differ otherwise the byte-identical short-circuit
    returns 'preserved'.
    """
    pre_files = ["a.py", "b.py"]
    post_files = ["a.py", "c.py"]
    pre = (
        "diff --git a/a.py b/a.py\n@@ -1 +1 @@\n-x\n+y\n"
        "diff --git a/b.py b/b.py\n@@ -1 +1 @@\n-x\n+y\n"
    )
    post = (
        "diff --git a/a.py b/a.py\n@@ -1 +1 @@\n-x\n+y\n"
        "diff --git a/c.py b/c.py\n@@ -1 +1 @@\n-x\n+y\n"
    )
    assert audit_mod._classify_fidelity(pre, post, pre_files, post_files) == "files_altered"


def test_audit_task_handles_sentinel_first_node(tmp_path: Path) -> None:
    """First node sentinel-ed → use second non-sentinel as 'pre'."""
    events = [
        # Node 0: planner sentinel
        {
            "event_type": "node_completed",
            "node_role": "planner",
            "model_id": "deepseek-v4-flash",
            "provider_id": "deepseek",
            "payload": "[sage: agent exited after 5 steps with no content]",
        },
        # Node 1: coder produces a real diff
        {
            "event_type": "node_completed",
            "node_role": "coder",
            "model_id": "deepseek-v4-pro",
            "provider_id": "deepseek",
            "payload": "```diff\ndiff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n```",
        },
        # Node 2: synth preserves
        {
            "event_type": "node_completed",
            "node_role": "synthesizer",
            "model_id": "gemini-2.5-flash",
            "provider_id": "google",
            "payload": "```diff\ndiff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n```",
        },
    ]
    events_path = tmp_path / "tt.events.jsonl"
    events_path.write_text(
        "\n".join(json.dumps(e) for e in events),
        encoding="utf-8",
    )

    final_patch = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
    result = audit_mod.audit_task(events_path, final_patch)

    assert result["verdict"] == "preserved"
    assert result["first_substantive_node"]["role"] == "coder"
    assert result["last_node"]["role"] == "synthesizer"


def test_audit_task_detects_rewritten_files(tmp_path: Path) -> None:
    """Reproduces the webclients/db90 AVR case where the judge emits a
    diff for a different file than the actor.
    """
    events = [
        {
            "event_type": "node_completed",
            "node_role": "actor",
            "model_id": "deepseek-v4-pro",
            "provider_id": "deepseek",
            "payload": "```diff\ndiff --git a/packages/foo.tsx b/packages/foo.tsx\n--- a/packages/foo.tsx\n+++ b/packages/foo.tsx\n@@ -1 +1 @@\n-a\n+b\n```",
        },
        {
            "event_type": "node_completed",
            "node_role": "verifier",
            "model_id": "gemini-3-flash-preview",
            "provider_id": "google",
            "payload": "[sage: agent exited after 5 steps with no content]",
        },
        {
            "event_type": "node_completed",
            "node_role": "judge",
            "model_id": "gemini-2.5-flash",
            "provider_id": "google",
            "payload": "```diff\ndiff --git a/src/app/foo.tsx b/src/app/foo.tsx\n--- a/src/app/foo.tsx\n+++ b/src/app/foo.tsx\n@@ -1 +1 @@\n-a\n+b\n```",
        },
    ]
    events_path = tmp_path / "tt.events.jsonl"
    events_path.write_text(
        "\n".join(json.dumps(e) for e in events),
        encoding="utf-8",
    )

    final_patch = "diff --git a/src/app/foo.tsx b/src/app/foo.tsx\n--- a/src/app/foo.tsx\n+++ b/src/app/foo.tsx\n@@ -1 +1 @@\n-a\n+b\n"
    result = audit_mod.audit_task(events_path, final_patch)

    assert result["verdict"] == "rewritten"
    assert result["first_substantive_node"]["role"] == "actor"
    assert result["last_node"]["role"] == "judge"
    assert result["files_dropped_pre_to_post"] == ["packages/foo.tsx"]
    assert result["files_added_pre_to_post"] == ["src/app/foo.tsx"]


def test_audit_task_all_sentinel_returns_special_verdict(tmp_path: Path) -> None:
    """Every node sentinel-ed → can't pick a 'pre' substantive node.
    Returns ``verdict="all_sentinel"`` so the caller knows to skip.
    """
    events = [
        {
            "event_type": "node_completed",
            "node_role": "planner",
            "model_id": "X",
            "provider_id": "X",
            "payload": "[sage: agent exited after 5 steps with no content]",
        },
        {
            "event_type": "node_completed",
            "node_role": "coder",
            "model_id": "X",
            "provider_id": "X",
            "payload": "[sage: agent exited after 10 steps with no content]",
        },
    ]
    events_path = tmp_path / "tt.events.jsonl"
    events_path.write_text(
        "\n".join(json.dumps(e) for e in events),
        encoding="utf-8",
    )

    result = audit_mod.audit_task(events_path, "")
    assert result["verdict"] == "all_sentinel"
    assert result["first_substantive_node"] is None


def test_slice_9_real_artefact_verdict_fingerprint() -> None:
    """Pin the slice 9 N=5 audit fingerprint. If a future test run
    drifts these numbers, either the audit logic changed or the slice 9
    artifact was regenerated. Either way: investigate.
    """
    run_dir = (
        _REPO_ROOT
        / "docs"
        / "benchmarks"
        / "2026-05-11-canary-patch-focused-prompt-profile"
        / "run"
    )
    if not run_dir.is_dir():
        pytest.skip(f"slice 9 artefact missing at {run_dir}")

    audit = audit_mod.run_audit(run_dir)
    assert audit["n_tasks"] == 5
    # 2 preserved (NodeBB sequential 2-node coder→mixer,
    #               teleport sequential 3-node coder→synth)
    # 1 cosmetic_drift (tutanota 219bc sequential 3-node, 43-char delta)
    # 2 rewritten (webclients + tutanota db90, both AVR judge/output)
    assert audit["verdict_tally"] == {
        "preserved": 2,
        "cosmetic_drift": 1,
        "rewritten": 2,
    }
    assert audit["any_rewritten"] is True


def test_main_writes_output_and_exits_nonzero_when_rewritten(tmp_path: Path) -> None:
    """End-to-end CLI: when any task verdict is 'rewritten', main()
    returns 1 so CI can fail-fast on diff fidelity regressions.
    """
    run_dir = tmp_path / "run"
    per_task = run_dir / "per_task"
    per_task.mkdir(parents=True)

    # Single task that triggers 'rewritten'
    events = [
        {
            "event_type": "node_completed",
            "node_role": "actor",
            "model_id": "X", "provider_id": "X",
            "payload": "```diff\ndiff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n```",
        },
        {
            "event_type": "node_completed",
            "node_role": "judge",
            "model_id": "X", "provider_id": "X",
            "payload": "```diff\ndiff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-x\n+y\n```",
        },
    ]
    (per_task / "instance_foo.events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events),
        encoding="utf-8",
    )
    (run_dir / "predictions.json").write_text(
        json.dumps([{
            "instance_id": "instance_foo",
            "patch": "diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n@@ -1 +1 @@\n-x\n+y\n",
        }]),
        encoding="utf-8",
    )

    output_path = tmp_path / "result.json"
    exit_code = audit_mod.main([
        "--run-dir", str(run_dir),
        "--output", str(output_path),
    ])
    assert exit_code == 1, "any_rewritten should trigger exit 1"
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["any_rewritten"] is True


def test_main_returns_zero_when_all_preserved(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    per_task = run_dir / "per_task"
    per_task.mkdir(parents=True)

    events = [
        {
            "event_type": "node_completed",
            "node_role": "coder",
            "model_id": "X", "provider_id": "X",
            "payload": "```diff\ndiff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n```",
        },
        {
            "event_type": "node_completed",
            "node_role": "synthesizer",
            "model_id": "X", "provider_id": "X",
            "payload": "```diff\ndiff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n```",
        },
    ]
    (per_task / "instance_bar.events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events),
        encoding="utf-8",
    )
    (run_dir / "predictions.json").write_text(
        json.dumps([{
            "instance_id": "instance_bar",
            "patch": "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n",
        }]),
        encoding="utf-8",
    )

    output_path = tmp_path / "result.json"
    exit_code = audit_mod.main([
        "--run-dir", str(run_dir),
        "--output", str(output_path),
    ])
    assert exit_code == 0
