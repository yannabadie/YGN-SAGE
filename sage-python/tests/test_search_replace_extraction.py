"""TDD red-phase tests for the search/replace extractor (Track 2 task 2.1).

These tests describe the behaviour of two helpers that will live at
module level in ``sage.bench.swebench_bench``:

* ``_extract_search_replace_blocks(response, repo_dir)`` — parse one or
  more SEARCH/REPLACE blocks from an LLM response into
  ``[(file_path, search_text, replace_text), ...]``.
* ``_blocks_to_unified_diff(blocks, repo_dir)`` — materialise a unified
  diff from the blocks by locating each ``search_text`` in the referenced
  file (exact first, ``difflib`` fuzzy match with confidence >= 0.95 as a
  fallback). Returns ``(diff_text, metadata)``.

Neither helper exists yet (task 2.2 implements them), so the module
import below is expected to raise ``ImportError`` at collection time.
The failing import IS the red: T2.2 is what turns this file green.

Block syntax expected by the extractor (per
``docs/superpowers/plans/2026-04-21-semantic-quality-plan.md``)::

    ## File: path/to/module.py
    <<<<<<< SEARCH
    <exact existing text>
    =======
    <replacement text>
    >>>>>>> REPLACE
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

# Direct import — by design, this raises ImportError at collection until
# task 2.2 adds the helpers. That IS the red signal. Do NOT wrap in
# try/except or importorskip: an explicit collection-time error is the
# cleanest "these are the functions I need" statement.
from sage.bench.swebench_bench import (  # noqa: E402
    _blocks_to_unified_diff,
    _extract_search_replace_blocks,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _init_git_repo(repo_dir: Path) -> None:
    """Initialise a git repo at ``repo_dir`` and commit everything present.

    ``git apply --check`` needs an index to diff against. We commit the
    current worktree so the diffs emitted by ``_blocks_to_unified_diff``
    (which use ``a/`` and ``b/`` prefixes against the committed state)
    can be validated.
    """
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(
        ["git", "init", "-q", "-b", "main"],
        cwd=repo_dir, check=True, env={**env, "PATH": _path_env()},
    )
    subprocess.run(
        ["git", "add", "-A"],
        cwd=repo_dir, check=True, env={**env, "PATH": _path_env()},
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "seed"],
        cwd=repo_dir, check=True, env={**env, "PATH": _path_env()},
    )


def _path_env() -> str:
    import os
    return os.environ.get("PATH", "")


def _git_apply_check(repo_dir: Path, diff_text: str) -> subprocess.CompletedProcess:
    """Run ``git apply --check`` against ``diff_text`` inside ``repo_dir``.

    Returns the completed process; caller inspects ``returncode`` and
    ``stderr``. Uses stdin so we don't have to write a temp file.
    """
    return subprocess.run(
        ["git", "apply", "--check", "-"],
        cwd=repo_dir,
        input=diff_text,
        text=True,
        capture_output=True,
    )


_GIT_AVAILABLE = shutil.which("git") is not None


# ---------------------------------------------------------------------------
# _extract_search_replace_blocks
# ---------------------------------------------------------------------------


def test_extract_simple_block_with_file_marker(tmp_path: Path) -> None:
    """Case (a): single SEARCH/REPLACE block with explicit ``## File:`` marker.

    The helper must return exactly one tuple whose three slots correspond
    to the path, the verbatim search text, and the verbatim replace text.
    """
    response = (
        "Here is the fix.\n"
        "\n"
        "## File: pkg/mod.py\n"
        "<<<<<<< SEARCH\n"
        "def foo(x):\n"
        "    return x\n"
        "=======\n"
        "def foo(x):\n"
        "    return x + 1\n"
        ">>>>>>> REPLACE\n"
    )

    blocks = _extract_search_replace_blocks(response, tmp_path)

    assert blocks == [
        (
            "pkg/mod.py",
            "def foo(x):\n    return x\n",
            "def foo(x):\n    return x + 1\n",
        )
    ]


def test_extract_multiple_blocks_returned_in_order(tmp_path: Path) -> None:
    """Case (b): two blocks in the same response, different files, preserved order."""
    response = (
        "## File: pkg/a.py\n"
        "<<<<<<< SEARCH\n"
        "x = 1\n"
        "=======\n"
        "x = 2\n"
        ">>>>>>> REPLACE\n"
        "\n"
        "## File: pkg/b.py\n"
        "<<<<<<< SEARCH\n"
        "y = 3\n"
        "=======\n"
        "y = 4\n"
        ">>>>>>> REPLACE\n"
    )

    blocks = _extract_search_replace_blocks(response, tmp_path)

    assert len(blocks) == 2
    assert blocks[0][0] == "pkg/a.py"
    assert blocks[0][1] == "x = 1\n"
    assert blocks[0][2] == "x = 2\n"
    assert blocks[1][0] == "pkg/b.py"
    assert blocks[1][1] == "y = 3\n"
    assert blocks[1][2] == "y = 4\n"


def test_extract_without_file_marker_unique_content(tmp_path: Path) -> None:
    """Case (c): no ``## File:`` marker — helper scans ``repo_dir`` for a
    unique file whose contents contain the search text verbatim."""
    (tmp_path / "a.py").write_text("def unique_symbol():\n    return 42\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("# unrelated content\nz = 0\n", encoding="utf-8")

    response = (
        "<<<<<<< SEARCH\n"
        "def unique_symbol():\n"
        "    return 42\n"
        "=======\n"
        "def unique_symbol():\n"
        "    return 43\n"
        ">>>>>>> REPLACE\n"
    )

    blocks = _extract_search_replace_blocks(response, tmp_path)

    assert len(blocks) == 1
    # The path is relative to repo_dir, not absolute.
    assert Path(blocks[0][0]).name == "a.py"
    assert blocks[0][1] == "def unique_symbol():\n    return 42\n"
    assert blocks[0][2] == "def unique_symbol():\n    return 43\n"


def test_extract_without_file_marker_ambiguous_drops_block(tmp_path: Path) -> None:
    """Case (d): no ``## File:`` marker and >1 file matches — block is
    dropped. Other (unambiguous) blocks in the same response still return."""
    shared = "shared_line = 'x'\n"
    (tmp_path / "a.py").write_text(shared, encoding="utf-8")
    (tmp_path / "b.py").write_text(shared, encoding="utf-8")
    # Disambiguated block: explicit marker.
    (tmp_path / "c.py").write_text("z = 0\n", encoding="utf-8")

    response = (
        "<<<<<<< SEARCH\n"
        f"{shared}"
        "=======\n"
        "shared_line = 'y'\n"
        ">>>>>>> REPLACE\n"
        "\n"
        "## File: c.py\n"
        "<<<<<<< SEARCH\n"
        "z = 0\n"
        "=======\n"
        "z = 1\n"
        ">>>>>>> REPLACE\n"
    )

    blocks = _extract_search_replace_blocks(response, tmp_path)

    # The ambiguous first block is dropped; the second (explicit) is kept.
    assert len(blocks) == 1
    assert Path(blocks[0][0]).name == "c.py"
    assert blocks[0][1] == "z = 0\n"
    assert blocks[0][2] == "z = 1\n"


def test_extract_malformed_block_dropped_valid_kept(tmp_path: Path) -> None:
    """Case (e): a malformed block (missing ``=======`` separator) is
    dropped; a later well-formed block still comes back."""
    response = (
        "## File: pkg/bad.py\n"
        "<<<<<<< SEARCH\n"
        "this block is missing its separator\n"
        ">>>>>>> REPLACE\n"
        "\n"
        "## File: pkg/good.py\n"
        "<<<<<<< SEARCH\n"
        "keep = 1\n"
        "=======\n"
        "keep = 2\n"
        ">>>>>>> REPLACE\n"
    )

    blocks = _extract_search_replace_blocks(response, tmp_path)

    assert len(blocks) == 1
    assert blocks[0][0] == "pkg/good.py"
    assert blocks[0][1] == "keep = 1\n"
    assert blocks[0][2] == "keep = 2\n"


def test_extract_empty_response_returns_empty_list(tmp_path: Path) -> None:
    """Sanity: empty / whitespace-only response yields an empty list
    rather than raising. (The task spec bullet calls this out explicitly.)"""
    assert _extract_search_replace_blocks("", tmp_path) == []
    assert _extract_search_replace_blocks("   \n\n  ", tmp_path) == []


# ---------------------------------------------------------------------------
# _blocks_to_unified_diff
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _GIT_AVAILABLE, reason="git not on PATH")
def test_blocks_to_unified_diff_exact_match_applies_cleanly(tmp_path: Path) -> None:
    """Case (f): byte-identical ``search_text`` inside the file produces a
    diff that ``git apply --check`` accepts, and metadata records
    ``match_kind == 'exact'``."""
    file_rel = "pkg/mod.py"
    (tmp_path / "pkg").mkdir()
    original = (
        "def compute(x):\n"
        "    y = x * 2\n"
        "    return y\n"
    )
    (tmp_path / file_rel).write_text(original, encoding="utf-8")
    _init_git_repo(tmp_path)

    blocks = [
        (
            file_rel,
            "def compute(x):\n    y = x * 2\n    return y\n",
            "def compute(x):\n    y = x * 3\n    return y\n",
        )
    ]

    diff_text, metadata = _blocks_to_unified_diff(blocks, tmp_path)

    assert diff_text, "expected a non-empty diff for an exact-match block"
    result = _git_apply_check(tmp_path, diff_text)
    assert result.returncode == 0, (
        f"git apply --check rejected the diff: {result.stderr}\n\nDIFF:\n{diff_text}"
    )
    assert metadata.get("per_block"), "metadata must include a per_block list"
    assert metadata["per_block"][0]["file"].endswith("mod.py")
    assert metadata["per_block"][0]["match_kind"] == "exact"


@pytest.mark.skipif(not _GIT_AVAILABLE, reason="git not on PATH")
def test_blocks_to_unified_diff_fuzzy_match_applies_cleanly(tmp_path: Path) -> None:
    """Case (g): search text differs by 1 leading space from the file
    contents. The helper must fall back to fuzzy matching (confidence
    >= 0.95), emit a diff that applies cleanly against the ACTUAL file
    contents, and record ``match_kind == 'fuzzy'``."""
    file_rel = "pkg/mod.py"
    (tmp_path / "pkg").mkdir()
    # Actual file uses 4-space indent.
    on_disk = (
        "class C:\n"
        "    def method(self, x):\n"
        "        return x + 1\n"
    )
    (tmp_path / file_rel).write_text(on_disk, encoding="utf-8")
    _init_git_repo(tmp_path)

    # LLM-supplied search block drops one space on the method line — close
    # but not byte-identical. This is the canonical fuzzy case.
    blocks = [
        (
            file_rel,
            "class C:\n   def method(self, x):\n        return x + 1\n",
            "class C:\n    def method(self, x):\n        return x + 2\n",
        )
    ]

    diff_text, metadata = _blocks_to_unified_diff(blocks, tmp_path)

    assert diff_text, "expected a non-empty diff for a fuzzy-match block"
    result = _git_apply_check(tmp_path, diff_text)
    assert result.returncode == 0, (
        f"git apply --check rejected the diff: {result.stderr}\n\nDIFF:\n{diff_text}"
    )
    assert metadata["per_block"][0]["match_kind"] == "fuzzy"


def test_blocks_to_unified_diff_missing_records_and_skips(tmp_path: Path) -> None:
    """Case (h): ``search_text`` absent from the file — even fuzzy-wise.
    That block is skipped (no hunk for it) and metadata marks it
    ``missing``. Other blocks may still contribute a diff; here we pass
    only the missing block so the diff comes back empty."""
    file_rel = "pkg/mod.py"
    (tmp_path / "pkg").mkdir()
    (tmp_path / file_rel).write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")

    blocks = [
        (
            file_rel,
            "completely_unrelated_content_that_is_absent\n",
            "whatever = 'never applied'\n",
        )
    ]

    diff_text, metadata = _blocks_to_unified_diff(blocks, tmp_path)

    # Caller treats empty diff as "no patch".
    assert diff_text == ""
    assert metadata.get("per_block"), "metadata must include a per_block list"
    assert metadata["per_block"][0]["match_kind"] == "missing"


# ---------------------------------------------------------------------------
# Path-traversal guard (regression tests for review blocker B1)
# ---------------------------------------------------------------------------


def test_diff_rejects_parent_directory_escape(tmp_path: Path) -> None:
    """The ``## File:`` marker is LLM-controlled; a malicious marker like
    ``../../etc/passwd`` must not cause ``_blocks_to_unified_diff`` to
    read a file outside ``repo_dir``. The block drops with
    ``match_kind == 'missing'`` and no hunk is contributed.
    """
    # repo_dir is a child of tmp_path; the block tries to escape one level.
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    # Create a sibling "secret" file that the attack would try to read.
    (tmp_path / "secret.txt").write_text("SECRET\n", encoding="utf-8")
    # Also create a benign file inside repo_dir so the fixture is non-empty.
    (repo_dir / "ok.py").write_text("ok = True\n", encoding="utf-8")

    blocks = [
        (
            "../secret.txt",
            "SECRET\n",
            "owned\n",
        )
    ]

    diff_text, metadata = _blocks_to_unified_diff(blocks, repo_dir)

    assert diff_text == ""
    assert metadata["per_block"][0]["file"] == "../secret.txt"
    assert metadata["per_block"][0]["match_kind"] == "missing"


def test_diff_rejects_absolute_path(tmp_path: Path) -> None:
    """An absolute path in the ``## File:`` marker (e.g. the Windows
    ``C:/Windows/...`` case that bypasses ``Path(repo) / absolute``
    joining) must be rejected. Uses ``tmp_path`` itself — which is
    outside the ``repo_dir`` child — as the stand-in absolute path, so
    the test is platform-neutral.
    """
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("private = 'data'\n", encoding="utf-8")

    # Absolute POSIX-formatted path pointing outside repo_dir.
    abs_path_str = str(outside).replace("\\", "/")

    blocks = [
        (
            abs_path_str,
            "private = 'data'\n",
            "private = 'pwned'\n",
        )
    ]

    diff_text, metadata = _blocks_to_unified_diff(blocks, repo_dir)

    assert diff_text == ""
    assert metadata["per_block"][0]["file"] == abs_path_str
    assert metadata["per_block"][0]["match_kind"] == "missing"
