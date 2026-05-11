"""Tests for sage-python/scripts/swebench_pro_fetch.py.

Block ``canary-stage-timing-budget`` slice 4 (cgpro DESIGN 2026-05-11).
Covers:

- Non-leakage difficulty proxy key: order is driven by
  ``problem_statement`` length + language + instance_id only.
- ``_DIFFICULTY_TRIAGE_BANNED_FIELDS`` enumerates patch / test_patch /
  fail_to_pass / pass_to_pass (lower + upper casing) and the proxy
  source code does NOT reference any of those names (AST-based check).
- ``fetch(..., difficulty_first=True)`` re-orders the stratified
  sample without changing which tasks are selected.
- ``--difficulty-first`` CLI flag plumbs through to ``fetch``.
- Manifest records ``difficulty_first`` + the proxy inputs used + the
  banned-field allowlist so downstream readers can verify the bound.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "sage-python" / "scripts" / "swebench_pro_fetch.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "swebench_pro_fetch", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fetch_mod = _load_module()


def test_banned_fields_enumeration() -> None:
    """The frozenset of fields a difficulty proxy MUST NOT touch must
    cover both lowercase and UPPER_CASE casing variants of the SWE-bench
    Pro ground-truth columns. Drift here weakens the non-leakage bound.
    """
    assert fetch_mod._DIFFICULTY_TRIAGE_BANNED_FIELDS == frozenset(
        {
            "patch",
            "test_patch",
            "fail_to_pass",
            "pass_to_pass",
            "FAIL_TO_PASS",
            "PASS_TO_PASS",
        }
    )


def test_proxy_source_does_not_reference_banned_fields() -> None:
    """Source-level regression: the difficulty-proxy functions
    (``_problem_statement_chars``, ``_difficulty_proxy_key``,
    ``_sort_by_difficulty_proxy``) MUST NOT mention any banned field
    name in their source text. If a future refactor accidentally pulls
    in patch/test_patch/etc., this test fails before code review.
    """
    sources = "\n".join(
        inspect.getsource(fn)
        for fn in (
            fetch_mod._problem_statement_chars,
            fetch_mod._difficulty_proxy_key,
            fetch_mod._sort_by_difficulty_proxy,
        )
    )
    for banned in fetch_mod._DIFFICULTY_TRIAGE_BANNED_FIELDS:
        assert banned not in sources, (
            f"Difficulty proxy references banned field {banned!r}; "
            "this is ground-truth leakage per cgpro DESIGN 2026-05-11."
        )


def test_problem_statement_chars_happy_and_missing() -> None:
    assert fetch_mod._problem_statement_chars({"problem_statement": "abcd"}) == 4
    assert fetch_mod._problem_statement_chars({}) == 0
    assert fetch_mod._problem_statement_chars({"problem_statement": None}) == 0
    # Non-string types (defensive) return 0, never raise.
    assert fetch_mod._problem_statement_chars({"problem_statement": 123}) == 0


def test_difficulty_proxy_key_shape_and_components() -> None:
    row = {
        "instance_id": "inst-1",
        "problem_statement": "Bug in foo",  # 10 chars
        "language": "Go",
    }
    assert fetch_mod._difficulty_proxy_key(row) == (10, "Go", "inst-1")


def test_difficulty_proxy_key_falls_back_to_repo_language_then_unknown() -> None:
    row = {
        "instance_id": "inst-2",
        "problem_statement": "Issue",
        "repo_language": "Python",
    }
    assert fetch_mod._difficulty_proxy_key(row) == (5, "Python", "inst-2")

    row_missing = {"instance_id": "inst-3", "problem_statement": ""}
    assert fetch_mod._difficulty_proxy_key(row_missing) == (0, "lang_unknown", "inst-3")


def test_sort_by_difficulty_proxy_orders_by_length_asc() -> None:
    rows: list[dict[str, Any]] = [
        {"instance_id": "a", "problem_statement": "x" * 100, "language": "Go"},
        {"instance_id": "b", "problem_statement": "x" * 10, "language": "Go"},
        {"instance_id": "c", "problem_statement": "x" * 50, "language": "Python"},
    ]
    ordered = fetch_mod._sort_by_difficulty_proxy(rows)
    assert [r["instance_id"] for r in ordered] == ["b", "c", "a"]


def test_sort_by_difficulty_proxy_uses_language_then_id_as_tiebreaker() -> None:
    """Equal problem_statement length: order by language asc, then
    instance_id asc. Deterministic for any input.
    """
    rows = [
        {"instance_id": "z", "problem_statement": "same", "language": "Go"},
        {"instance_id": "a", "problem_statement": "same", "language": "Go"},
        {"instance_id": "m", "problem_statement": "same", "language": "C++"},
    ]
    ordered = fetch_mod._sort_by_difficulty_proxy(rows)
    # C++ < Go (string compare), then within Go a < z.
    assert [r["instance_id"] for r in ordered] == ["m", "a", "z"]


def test_sort_is_idempotent() -> None:
    rows = [
        {"instance_id": "a", "problem_statement": "x" * 20, "language": "Go"},
        {"instance_id": "b", "problem_statement": "x" * 5, "language": "Python"},
    ]
    once = fetch_mod._sort_by_difficulty_proxy(rows)
    twice = fetch_mod._sort_by_difficulty_proxy(once)
    assert [r["instance_id"] for r in once] == [r["instance_id"] for r in twice]


def test_fetch_difficulty_first_changes_order_not_selection(
    monkeypatch, tmp_path: Path
) -> None:
    """``fetch(difficulty_first=True)`` must reorder the post-stratified
    selection lightest-first without changing which tasks were picked.
    """
    fake_dataset = [
        {
            "instance_id": f"task-{i}",
            "repo": f"r{i % 3}/x",
            "problem_statement": "x" * length,
            "language": "Go" if i % 2 == 0 else "Python",
            "base_commit": "deadbeef",
        }
        for i, length in enumerate([100, 5, 60, 20, 300, 40, 80])
    ]

    monkeypatch.setattr(fetch_mod, "_load_dataset", lambda: fake_dataset)

    out_dir_a = tmp_path / "unsorted"
    out_dir_b = tmp_path / "difficulty"
    manifest_a = fetch_mod.fetch(n=5, output_dir=out_dir_a, seed=42, difficulty_first=False)
    manifest_b = fetch_mod.fetch(n=5, output_dir=out_dir_b, seed=42, difficulty_first=True)

    # Same instance set (same seed + same N), just reordered.
    assert set(manifest_a["instance_ids"]) == set(manifest_b["instance_ids"])

    instances_b = json.loads((out_dir_b / "instances.json").read_text(encoding="utf-8"))
    lengths_b = [len(r.get("problem_statement") or "") for r in instances_b]
    assert lengths_b == sorted(lengths_b), (
        "difficulty_first should yield non-decreasing problem_statement lengths "
        f"got {lengths_b}"
    )

    # Manifest records the metadata used and the banned list (audit
    # trail for downstream readers).
    assert manifest_b["difficulty_first"] is True
    assert manifest_b["difficulty_triage_inputs"] == [
        "problem_statement_chars",
        "language",
        "instance_id",
    ]
    assert set(manifest_b["difficulty_triage_banned_fields"]) == {
        "patch",
        "test_patch",
        "fail_to_pass",
        "pass_to_pass",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
    }
    assert manifest_a["difficulty_first"] is False
    assert manifest_a["difficulty_triage_inputs"] == []


def test_fetch_default_does_not_sort(monkeypatch, tmp_path: Path) -> None:
    """Without ``difficulty_first``, the order is whatever
    ``_stratified_sample`` produced — NOT the difficulty proxy. This
    preserves prior behavior for callers that haven't opted in.
    """
    fake_dataset = [
        {
            "instance_id": f"task-{i}",
            "repo": f"r{i % 2}/x",
            "problem_statement": "x" * length,
            "language": "Go",
            "base_commit": "deadbeef",
        }
        for i, length in enumerate([100, 5, 60, 20, 300])
    ]

    monkeypatch.setattr(fetch_mod, "_load_dataset", lambda: fake_dataset)

    manifest = fetch_mod.fetch(n=3, output_dir=tmp_path / "out", seed=42)
    instances = json.loads(
        (tmp_path / "out" / "instances.json").read_text(encoding="utf-8")
    )
    lengths = [len(r.get("problem_statement") or "") for r in instances]
    # Without difficulty_first, the lengths are NOT guaranteed sorted.
    # We only assert the count + that difficulty_first manifest flag is
    # False; ordering equality with a specific sequence depends on the
    # round-robin in _stratified_sample which is intentionally not
    # length-aware.
    assert len(lengths) == 3
    assert manifest["difficulty_first"] is False


def test_cli_difficulty_first_flag_wires_to_fetch(monkeypatch, tmp_path: Path) -> None:
    """Smoke the argparse wiring: ``--difficulty-first`` reaches
    ``fetch()`` as ``difficulty_first=True``.
    """
    captured: dict[str, Any] = {}

    def _fake_fetch(n, output_dir, seed, **kwargs):  # type: ignore[no-untyped-def]
        captured["n"] = n
        captured["output_dir"] = output_dir
        captured["seed"] = seed
        captured["kwargs"] = kwargs
        return {"instance_ids": []}

    monkeypatch.setattr(fetch_mod, "fetch", _fake_fetch)

    exit_code = fetch_mod.main(
        [
            "--n",
            "3",
            "--output-dir",
            str(tmp_path / "out"),
            "--seed",
            "7",
            "--difficulty-first",
        ]
    )
    assert exit_code == 0
    assert captured["n"] == 3
    assert captured["seed"] == 7
    assert captured["kwargs"]["difficulty_first"] is True


def test_cli_default_keeps_difficulty_first_false(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def _fake_fetch(n, output_dir, seed, **kwargs):  # type: ignore[no-untyped-def]
        captured["kwargs"] = kwargs
        return {"instance_ids": []}

    monkeypatch.setattr(fetch_mod, "fetch", _fake_fetch)

    exit_code = fetch_mod.main(
        [
            "--n",
            "3",
            "--output-dir",
            str(tmp_path / "out"),
            "--seed",
            "7",
        ]
    )
    assert exit_code == 0
    assert captured["kwargs"]["difficulty_first"] is False
