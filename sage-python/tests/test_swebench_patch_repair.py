"""Tests for sage.bench.swebench_patch_repair.

Covers:
  - _fix_hunk_header_counts on well-formed (no-op), malformed-counts,
    and no-count-shorthand patches.
  - validate_patch_apply against a real git repo (tmp_path init+commit).
  - try_repair_patch happy-path and failed-path + metadata tag.
  - repair_patch_via_llm with a stub LLM provider.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

# Ensure the swebench_bench import in repair_patch_via_llm doesn't pull
# the full swebench harness (Unix-only `resource` dep). Stub preemptively.
import sys
if sys.platform != "linux" and "resource" not in sys.modules:
    import types as _types
    _stub = _types.ModuleType("resource")
    _stub.RLIMIT_NOFILE = 7  # type: ignore[attr-defined]
    _stub.getrlimit = lambda _x: (1024, 1048576)  # type: ignore[attr-defined]
    _stub.setrlimit = lambda _x, _y: None  # type: ignore[attr-defined]
    sys.modules["resource"] = _stub

from sage.bench.swebench_patch_repair import (  # noqa: E402
    _fix_hunk_header_counts,
    validate_patch_apply,
    repair_patch_via_llm,
    try_repair_patch,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a tiny git repo with one committed file."""
    src = tmp_path / "src"
    src.mkdir()
    f = src / "foo.py"
    f.write_text(
        "def add(a, b):\n"
        "    # simple sum\n"
        "    return a + b\n"
        "\n"
        "def mul(a, b):\n"
        "    return a * b\n"
    )
    # Minimal git init. Use subprocess so we test against the actual git
    # we shell out to in validate_patch_apply.
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A"],
        cwd=tmp_path, check=True,
    )
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-q", "-m", "init"],
        cwd=tmp_path, check=True,
    )
    return tmp_path


# ---------------------------------------------------------------------------
# _fix_hunk_header_counts
# ---------------------------------------------------------------------------


def test_fix_counts_noop_on_wellformed():
    """A correct diff round-trips unchanged through the fixer.

    Body: 1 context + 1 removed + 1 added + 1 context =
      src_count = 3 (2 context + 1 removed)
      dst_count = 3 (2 context + 1 added)
    """
    patch = (
        "--- a/src/foo.py\n"
        "+++ b/src/foo.py\n"
        "@@ -1,3 +1,3 @@\n"
        " def add(a, b):\n"
        "-    return a + b\n"
        "+    return a + b  # fixed\n"
        "     return 0\n"
    )
    fixed = _fix_hunk_header_counts(patch)
    assert fixed == patch, "well-formed patch should be unchanged"


def test_fix_counts_corrects_bogus_counts():
    """The astropy-7746-class fix: LLM claims counts that don't match body."""
    # Body: 1 context, 1 removed, 1 added -> src=2, dst=2.
    # Bogus header claims -1,28 +1,35.
    bogus = (
        "--- a/src/foo.py\n"
        "+++ b/src/foo.py\n"
        "@@ -1,28 +1,35 @@\n"
        " def add(a, b):\n"
        "-    return a + b\n"
        "+    return a + b  # fixed\n"
    )
    fixed = _fix_hunk_header_counts(bogus)
    assert "@@ -1,2 +1,2 @@" in fixed, f"counts not recomputed: {fixed!r}"
    # Body must be preserved verbatim.
    assert " def add(a, b):" in fixed
    assert "-    return a + b" in fixed
    assert "+    return a + b  # fixed" in fixed


def test_fix_counts_handles_shorthand_single_line_count():
    """A hunk with only '-count' omitted (meaning 1) gets explicit counts on output."""
    # Original: @@ -2 +2 @@ (no comma means count=1 per spec)
    patch = (
        "--- a/src/foo.py\n"
        "+++ b/src/foo.py\n"
        "@@ -2 +2 @@\n"
        "-    # simple sum\n"
        "+    # simple addition\n"
    )
    fixed = _fix_hunk_header_counts(patch)
    # src=1 (one '-'), dst=1 (one '+'). Fixer always writes explicit counts.
    assert "@@ -2,1 +2,1 @@" in fixed


def test_fix_counts_multi_hunk():
    """Multiple hunks in one patch each get recomputed independently."""
    bogus = (
        "--- a/src/foo.py\n"
        "+++ b/src/foo.py\n"
        "@@ -1,99 +1,99 @@\n"
        " def add(a, b):\n"
        "-    return a + b\n"
        "+    return a + b  # fix1\n"
        "@@ -5,99 +5,99 @@\n"
        " def mul(a, b):\n"
        "-    return a * b\n"
        "+    return a * b  # fix2\n"
    )
    fixed = _fix_hunk_header_counts(bogus)
    # Both hunks should have the same recomputed counts (same shape).
    assert fixed.count("@@ -1,2 +1,2 @@") == 1
    assert fixed.count("@@ -5,2 +5,2 @@") == 1


# ---------------------------------------------------------------------------
# validate_patch_apply
# ---------------------------------------------------------------------------


def test_validate_rejects_empty_patch(git_repo):
    ok, err = validate_patch_apply("", str(git_repo))
    assert ok is False
    assert "empty" in err.lower()


def test_validate_accepts_valid_patch(git_repo):
    """A patch with correct counts + matching context applies cleanly."""
    valid = (
        "--- a/src/foo.py\n"
        "+++ b/src/foo.py\n"
        "@@ -1,3 +1,3 @@\n"
        " def add(a, b):\n"
        "-    # simple sum\n"
        "+    # corrected sum\n"
        "     return a + b\n"
    )
    ok, err = validate_patch_apply(valid, str(git_repo))
    assert ok is True, f"valid patch rejected: {err}"


def test_validate_rejects_bogus_counts(git_repo):
    """The exact failure mode from astropy-7746: header counts don't match body."""
    bogus = (
        "--- a/src/foo.py\n"
        "+++ b/src/foo.py\n"
        "@@ -1,28 +1,35 @@\n"
        " def add(a, b):\n"
        "-    # simple sum\n"
        "+    # corrected sum\n"
        "     return a + b\n"
    )
    ok, err = validate_patch_apply(bogus, str(git_repo))
    assert ok is False
    assert err  # non-empty stderr


# ---------------------------------------------------------------------------
# try_repair_patch integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repair_unchanged_when_already_valid(git_repo):
    valid = (
        "--- a/src/foo.py\n"
        "+++ b/src/foo.py\n"
        "@@ -1,3 +1,3 @@\n"
        " def add(a, b):\n"
        "-    # simple sum\n"
        "+    # corrected sum\n"
        "     return a + b\n"
    )
    out, stage = await try_repair_patch(
        patch=valid, repo_dir=str(git_repo), llm=None,
        problem_statement="fix comment", instance_id="t1",
    )
    assert stage == "unchanged"
    assert out == valid


@pytest.mark.asyncio
async def test_repair_programmatic_counts_resolves(git_repo):
    """astropy-7746 pattern — counts-fix recovers the patch, no LLM needed."""
    bogus = (
        "--- a/src/foo.py\n"
        "+++ b/src/foo.py\n"
        "@@ -1,28 +1,35 @@\n"
        " def add(a, b):\n"
        "-    # simple sum\n"
        "+    # corrected sum\n"
        "     return a + b\n"
    )
    # llm=None is fine: programmatic stage should resolve it before LLM.
    out, stage = await try_repair_patch(
        patch=bogus, repo_dir=str(git_repo), llm=None,
        problem_statement="fix comment", instance_id="t2",
    )
    assert stage == "programmatic_counts", f"unexpected stage: {stage}"
    assert "@@ -1,3 +1,3 @@" in out


@pytest.mark.asyncio
async def test_repair_llm_stage_runs_when_programmatic_insufficient(git_repo):
    """django-10914 pattern — stale context line that programmatic can't fix.
    Stub LLM returns a corrected diff."""
    # Context line " def unchanged():\n" doesn't exist in our git_repo.
    stale = (
        "--- a/src/foo.py\n"
        "+++ b/src/foo.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def nonexistent_function(x):\n"
        "-    return x\n"
        "+    return x + 1\n"
    )

    class _StubLLM:
        calls = 0
        async def generate(self, messages, **_kwargs):
            _StubLLM.calls += 1
            class _Resp:
                content = (
                    "```diff\n"
                    "--- a/src/foo.py\n"
                    "+++ b/src/foo.py\n"
                    "@@ -1,3 +1,3 @@\n"
                    " def add(a, b):\n"
                    "-    # simple sum\n"
                    "+    # repaired by llm\n"
                    "     return a + b\n"
                    "```\n"
                )
            return _Resp()

    out, stage = await try_repair_patch(
        patch=stale, repo_dir=str(git_repo), llm=_StubLLM(),
        problem_statement="fix something", instance_id="t3",
    )
    assert stage == "llm_repair", f"unexpected stage: {stage}"
    assert "# repaired by llm" in out
    assert _StubLLM.calls == 1


@pytest.mark.asyncio
async def test_repair_failed_returns_original(git_repo):
    """If both stages fail, the original (broken) patch is returned so the
    downstream swebench evaluator can still bucket it as apply-error."""
    stale = (
        "--- a/src/foo.py\n"
        "+++ b/src/foo.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def nonexistent_function(x):\n"
        "-    return x\n"
        "+    return x + 1\n"
    )

    class _BadStubLLM:
        async def generate(self, messages, **_kwargs):
            # Returns garbage — no valid diff extractable
            class _Resp:
                content = "sorry, cannot fix"
            return _Resp()

    out, stage = await try_repair_patch(
        patch=stale, repo_dir=str(git_repo), llm=_BadStubLLM(),
        problem_statement="", instance_id="t4",
    )
    assert stage == "failed"
    assert out == stale  # unchanged so bench classifier sees the failure


@pytest.mark.asyncio
async def test_repair_noop_without_repo_dir(git_repo):
    """If the repo wasn't cloned (e.g. network failure during _setup_repo),
    we skip the entire pipeline gracefully."""
    out, stage = await try_repair_patch(
        patch="anything", repo_dir=None, llm=None,
        problem_statement="", instance_id="t5",
    )
    assert stage == ""
    assert out == "anything"


# ---------------------------------------------------------------------------
# repair_patch_via_llm direct
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repair_llm_extracts_fenced_diff():
    """LLM stub returns a fenced ```diff block; repair_patch_via_llm
    extracts the diff body."""
    class _Stub:
        async def generate(self, messages, **_kw):
            class _R:
                content = (
                    "Here is the fix:\n\n"
                    "```diff\n"
                    "--- a/x\n+++ b/x\n@@ -1,1 +1,1 @@\n-old\n+new\n"
                    "```\n"
                )
            return _R()

    out = await repair_patch_via_llm(
        _Stub(), "test problem", "broken patch", "error", timeout=5.0,
    )
    assert "--- a/x" in out
    assert "+new" in out


@pytest.mark.asyncio
async def test_repair_llm_returns_empty_on_timeout():
    class _SlowStub:
        async def generate(self, messages, **_kw):
            import asyncio
            await asyncio.sleep(5)
            class _R:
                content = "too late"
            return _R()

    out = await repair_patch_via_llm(
        _SlowStub(), "", "", "", timeout=0.1,
    )
    assert out == ""
