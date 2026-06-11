"""G2 positional reground contract — cgpro GROUNDING DESIGN_LOCKED
(2026-06-11, sequence 'G2 repair'). Reground REPOSITIONS only, never
invents: exact AND unique old-side match required per hunk; anything
else blocks (no fuzzy repair)."""
from __future__ import annotations

import subprocess
from pathlib import Path

from sage.patch_artifacts import git_apply_check, positional_reground_exact

FILE_TEXT = (
    "import os\n"
    "\n"
    "def alpha():\n"
    "    return 1\n"
    "\n"
    "def beta():\n"
    "    x = 2\n"
    "    return x\n"
    "\n"
    "def gamma():\n"
    "    return 3\n"
)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "mod.py").write_text(FILE_TEXT, encoding="utf-8")
    subprocess.run(["git", "init", str(repo)], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True,
                   capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c",
         "user.name=t", "commit", "-m", "init"],
        check=True, capture_output=True,
    )
    return repo


def test_positional_reground_rewrites_wrong_hunk_start_on_exact_unique_match(
    tmp_path,
) -> None:
    repo = _repo(tmp_path)
    # Real change lives at line 6 (def beta); the model claimed line 40.
    patch = (
        "--- a/src/mod.py\n"
        "+++ b/src/mod.py\n"
        "@@ -40,3 +40,4 @@\n"
        " def beta():\n"
        "+    # guard\n"
        "     x = 2\n"
        "     return x\n"
    )
    # NOTE: git apply itself tolerates wrong positions when the context
    # matches uniquely (it searches) — G2's value is normalizing headers
    # for the strict verifier AND the order/overlap cases git rejects.
    fixed, status = positional_reground_exact(patch, str(repo))
    assert status == "reground_applied"
    assert "@@ -6,3 +6,4 @@" in fixed
    assert git_apply_check(fixed, str(repo))[0] is True


def test_positional_reground_then_recount_passes_git_apply_check(
    tmp_path,
) -> None:
    repo = _repo(tmp_path)
    # Wrong position AND wrong counts — reground fixes both (it
    # recounts from the body while rewriting the header).
    patch = (
        "--- a/src/mod.py\n"
        "+++ b/src/mod.py\n"
        "@@ -40,9 +40,9 @@\n"
        " def beta():\n"
        "+    # guard\n"
        "     x = 2\n"
        "     return x\n"
    )
    fixed, status = positional_reground_exact(patch, str(repo))
    assert status == "reground_applied"
    ok, detail = git_apply_check(fixed, str(repo))
    assert ok is True, detail


def test_positional_reground_rejects_missing_path(tmp_path) -> None:
    repo = _repo(tmp_path)
    patch = (
        "--- a/src/invented.py\n"
        "+++ b/src/invented.py\n"
        "@@ -1,2 +1,3 @@\n"
        " def alpha():\n"
        "+    pass\n"
        "     return 1\n"
    )
    fixed, status = positional_reground_exact(patch, str(repo))
    assert status == "reground_rejected:missing_path"
    assert fixed == patch  # untouched


def test_positional_reground_rejects_no_exact_match(tmp_path) -> None:
    repo = _repo(tmp_path)
    patch = (
        "--- a/src/mod.py\n"
        "+++ b/src/mod.py\n"
        "@@ -3,2 +3,3 @@\n"
        " def hallucinated():\n"
        "+    pass\n"
        "     return 99\n"
    )
    fixed, status = positional_reground_exact(patch, str(repo))
    assert status == "reground_rejected:no_exact_match"
    assert fixed == patch


def test_positional_reground_rejects_ambiguous_match(tmp_path) -> None:
    repo = _repo(tmp_path)
    (repo / "src" / "dup.py").write_text(
        "x = 1\nx = 1\n", encoding="utf-8"
    )
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True,
                   capture_output=True)
    patch = (
        "--- a/src/dup.py\n"
        "+++ b/src/dup.py\n"
        "@@ -9,1 +9,2 @@\n"
        " x = 1\n"
        "+y = 2\n"
    )
    fixed, status = positional_reground_exact(patch, str(repo))
    assert status == "reground_rejected:ambiguous_match"
    assert fixed == patch


def test_positional_reground_rejects_truncated_hunk(tmp_path) -> None:
    repo = _repo(tmp_path)
    patch = (
        "--- a/src/mod.py\n"
        "+++ b/src/mod.py\n"
        "@@ -6,3 +6,4 @@\n"
    )
    fixed, status = positional_reground_exact(patch, str(repo))
    assert status == "reground_rejected:truncated_hunk"


def test_multiple_hunks_compute_new_start_with_cumulative_delta(
    tmp_path,
) -> None:
    repo = _repo(tmp_path)
    # Hunk 1 at alpha (+1 line), hunk 2 at gamma — its NEW start must
    # shift by hunk 1's delta. Model claimed nonsense positions for both.
    patch = (
        "--- a/src/mod.py\n"
        "+++ b/src/mod.py\n"
        "@@ -70,2 +70,3 @@\n"
        " def alpha():\n"
        "+    # a\n"
        "     return 1\n"
        "@@ -90,2 +90,3 @@\n"
        " def gamma():\n"
        "+    # g\n"
        "     return 3\n"
    )
    fixed, status = positional_reground_exact(patch, str(repo))
    assert status == "reground_applied"
    assert "@@ -3,2 +3,3 @@" in fixed
    assert "@@ -10,2 +11,3 @@" in fixed  # old 10, new 11 (delta +1)
    ok, detail = git_apply_check(fixed, str(repo))
    assert ok is True, detail


def test_reground_rejects_incoherent_hunk_order(tmp_path) -> None:
    repo = _repo(tmp_path)
    # Hunk for gamma BEFORE hunk for alpha — resolved old-starts would
    # be decreasing; strict mode blocks rather than reordering.
    patch = (
        "--- a/src/mod.py\n"
        "+++ b/src/mod.py\n"
        "@@ -1,2 +1,3 @@\n"
        " def gamma():\n"
        "+    # g\n"
        "     return 3\n"
        "@@ -50,2 +50,3 @@\n"
        " def alpha():\n"
        "+    # a\n"
        "     return 1\n"
    )
    fixed, status = positional_reground_exact(patch, str(repo))
    assert status == "reground_rejected:incoherent_hunk_order"
    assert fixed == patch
