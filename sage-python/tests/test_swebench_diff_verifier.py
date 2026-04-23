"""TDD tests for the pre-emission diff-context verifier.

Spec:
    docs/superpowers/specs/2026-04-23-diff-context-verifier-design.md

The verifier is a pure function on (unified-diff-str, repo-dir) that
returns a list of ``HunkMismatch`` — one entry per hunk whose context
(space-prefixed) or removed (``-``) lines disagree with the bytes at
the claimed hunk position in the repo. It does NOT attempt repair.
The on-disk hook (``generate_patches``) runs the verifier in observe
mode only and annotates predictions.jsonl — covered in
``test_swebench_emission_wiring.py``.

Scope of tests 1-7 mirrors the spec's "Tests (TDD spec)" table
verbatim; Test 8 answers spec open question #3 (SR-converted unified
diff still goes through the verifier).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sage.bench.swebench_diff_verifier import HunkMismatch, verify_diff_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(repo: Path, relpath: str, content: str) -> None:
    """Write ``content`` to ``repo/relpath`` creating parents as needed."""
    target = repo / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 1 — clean diff against a real fixture file.
# ---------------------------------------------------------------------------


def test_clean_diff_returns_empty(tmp_path):
    """A hunk whose context and removed lines exactly match file bytes at
    the claimed position returns no mismatches."""
    _write(
        tmp_path,
        "pkg/mod.py",
        "def foo():\n    return 1\n\n\ndef bar():\n    return 2\n",
    )
    diff = (
        "diff --git a/pkg/mod.py b/pkg/mod.py\n"
        "--- a/pkg/mod.py\n"
        "+++ b/pkg/mod.py\n"
        "@@ -1,3 +1,3 @@\n"
        " def foo():\n"
        "-    return 1\n"
        "+    return 10\n"
        " \n"
    )
    mismatches = verify_diff_context(diff, tmp_path)
    assert mismatches == []


# ---------------------------------------------------------------------------
# Test 2 — astropy-14182 Arm B context hallucination. The CRITICAL test.
# ---------------------------------------------------------------------------


# Emitted patch for astropy__astropy-14182 Arm B, copied VERBATIM from
# docs/benchmarks/2026-04-22-swebench-parity-smoke/2026-04-22-parity-typed-meta.json.
# The agent asserts `Table.__init__(self)` in the hunk body; the base
# commit's real file has `FixedWidth.__init__(self)`. The hunk header's
# function-context hint also reads `class RST(Table):` (vs the real
# `class RST(FixedWidth):`) but the verifier's scope is body lines
# (space-prefix + minus-prefix), NOT the function-context hint after `@@`.
# Spec: "What gets verified (A + file-exists): ... context (` `) and
# removed (`-`) lines match the file content at the claimed hunk position."
_ASTROPY_14182_ARM_B_EMITTED_PATCH = (
    "diff --git a/astropy/io/ascii/rst.py b/astropy/io/ascii/rst.py\n"
    "--- a/astropy/io/ascii/rst.py\n"
    "+++ b/astropy/io/ascii/rst.py\n"
    "@@ -35,6 +35,7 @@ class RST(Table):\n"
    '     """\n'
    '     _format_name = "rst"\n'
    '     _description = "RestructuredText grid table"\n'
    '+    _supported_write_kwargs = ("header_rows",)\n'
    " \n"
    "     def __init__(self, **kwargs):\n"
    "         Table.__init__(self)\n"
)


def test_astropy_14182_arm_b_context_hallucination_detected(tmp_path):
    """THE test this feature exists for. The emitted patch's hunk body
    says ``Table.__init__(self)``; the base commit at rst.py:40 has
    ``FixedWidth.__init__(self)``. Verifier must flag one
    ``content_mismatch`` with ``Table`` in expected and ``FixedWidth``
    in actual."""
    # Simulate the relevant portion of astropy/io/ascii/rst.py at
    # base_commit 920f121df. Only 50 lines; verifier doesn't need the
    # full file, just enough for old_start=35 + old_count=6 to resolve.
    lines: list[str] = []
    lines.append("import something\n")  # 1
    for i in range(2, 30):  # 2..29
        lines.append(f"# filler line {i}\n")
    lines.append("\n")  # 30
    lines.append("\n")  # 31
    lines.append("class RST(FixedWidth):\n")  # 32
    lines.append('    """RST writer."""\n')  # 33
    lines.append("\n")  # 34
    lines.append('    """\n')  # 35
    lines.append('    _format_name = "rst"\n')  # 36
    lines.append('    _description = "RestructuredText grid table"\n')  # 37
    lines.append("\n")  # 38
    lines.append("    def __init__(self, **kwargs):\n")  # 39
    lines.append("        FixedWidth.__init__(self)\n")  # 40 — mismatch
    for i in range(41, 60):
        lines.append(f"    # tail filler {i}\n")

    _write(tmp_path, "astropy/io/ascii/rst.py", "".join(lines))

    mismatches = verify_diff_context(
        _ASTROPY_14182_ARM_B_EMITTED_PATCH, tmp_path
    )
    assert len(mismatches) == 1, (
        f"expected exactly one mismatch, got {len(mismatches)}: "
        f"{mismatches!r}"
    )
    m = mismatches[0]
    assert m.kind == "content_mismatch", m
    assert m.file == "astropy/io/ascii/rst.py", m
    assert m.hunk_index == 0
    assert m.old_start == 35
    assert m.old_count == 6
    # Expected is what the agent said was in the file (from the diff).
    # Actual is what the file really contains at that position.
    expected_joined = "\n".join(m.expected)
    actual_joined = "\n".join(m.actual)
    assert "Table" in expected_joined, (
        f"expected body must contain 'Table' — got {expected_joined!r}"
    )
    assert "FixedWidth" in actual_joined, (
        f"actual body must contain 'FixedWidth' — got {actual_joined!r}"
    )


# ---------------------------------------------------------------------------
# Test 3 — hunk targets a file that doesn't exist in the repo.
# ---------------------------------------------------------------------------


def test_hunk_targets_missing_file(tmp_path):
    """Hunk refers to a file that does not exist on disk → one
    ``HunkMismatch`` with ``kind='file_missing'``."""
    # Empty repo — no files at all.
    diff = (
        "diff --git a/pkg/ghost.py b/pkg/ghost.py\n"
        "--- a/pkg/ghost.py\n"
        "+++ b/pkg/ghost.py\n"
        "@@ -1,2 +1,2 @@\n"
        " line 1\n"
        "-line 2\n"
        "+line 2 modified\n"
    )
    mismatches = verify_diff_context(diff, tmp_path)
    assert len(mismatches) == 1
    assert mismatches[0].kind == "file_missing"
    assert mismatches[0].file == "pkg/ghost.py"


# ---------------------------------------------------------------------------
# Test 4 — whitespace-only drift, both sides of the fuzzy threshold.
# ---------------------------------------------------------------------------


def test_whitespace_drift_above_threshold_is_clean(tmp_path):
    """Minor whitespace drift that still scores >=0.95 on
    SequenceMatcher is accepted as clean (no mismatch)."""
    # The file has a single trailing space on one line; the diff
    # doesn't. SequenceMatcher ratio stays very high.
    _write(
        tmp_path,
        "pkg/ws.py",
        "def foo():\n    return 1  \n    return 2\n",
    )
    diff = (
        "diff --git a/pkg/ws.py b/pkg/ws.py\n"
        "--- a/pkg/ws.py\n"
        "+++ b/pkg/ws.py\n"
        "@@ -1,3 +1,3 @@\n"
        " def foo():\n"
        "     return 1\n"
        "-    return 2\n"
        "+    return 20\n"
    )
    mismatches = verify_diff_context(diff, tmp_path)
    assert mismatches == [], (
        f"whitespace drift on a single trailing space should still fuzz "
        f"above 0.95; got {mismatches!r}"
    )


def test_whitespace_drift_below_threshold_flagged_as_fuzzy(tmp_path):
    """Heavy whitespace drift (different indentation throughout) drops
    the ratio below 0.95. Kind should be ``fuzzy_below_threshold``
    because stripped lines still match (purely whitespace difference).
    """
    # File uses tabs; patch uses 8-space indent throughout. Every
    # content line disagrees on whitespace but no tokens differ. Long
    # enough that SequenceMatcher ratio falls under 0.95.
    _write(
        tmp_path,
        "pkg/heavy_ws.py",
        (
            "def outer():\n"
            "\tx = 1\n"
            "\ty = 2\n"
            "\tz = 3\n"
            "\treturn x + y + z\n"
        ),
    )
    diff = (
        "diff --git a/pkg/heavy_ws.py b/pkg/heavy_ws.py\n"
        "--- a/pkg/heavy_ws.py\n"
        "+++ b/pkg/heavy_ws.py\n"
        "@@ -1,5 +1,5 @@\n"
        " def outer():\n"
        "         x = 1\n"
        "         y = 2\n"
        "         z = 3\n"
        "-        return x + y + z\n"
        "+        return x * y * z\n"
    )
    mismatches = verify_diff_context(diff, tmp_path)
    assert len(mismatches) == 1
    m = mismatches[0]
    assert m.kind == "fuzzy_below_threshold", m
    # Stripped forms match — that's why it's fuzzy_below, not
    # content_mismatch.
    stripped_expected = [l.strip() for l in m.expected]
    stripped_actual = [l.strip() for l in m.actual]
    assert stripped_expected == stripped_actual, (
        f"stripped should match for fuzzy_below_threshold; "
        f"expected={stripped_expected!r} actual={stripped_actual!r}"
    )
    assert m.match_ratio < 0.95


# ---------------------------------------------------------------------------
# Test 5 — multi-hunk diff: one mismatches, one is clean.
# ---------------------------------------------------------------------------


def test_multi_hunk_one_bad_one_good(tmp_path):
    """Two hunks in one diff; only one triggers a mismatch. The
    ``hunk_index`` on the returned mismatch must be the offending
    hunk's position (zero-indexed over all hunks in the diff)."""
    _write(
        tmp_path,
        "pkg/two.py",
        (
            "def alpha():\n"
            "    return 1\n"
            "\n"
            "def beta():\n"
            "    return 2\n"
            "\n"
            "def gamma():\n"
            "    return 3\n"
        ),
    )
    diff = (
        "diff --git a/pkg/two.py b/pkg/two.py\n"
        "--- a/pkg/two.py\n"
        "+++ b/pkg/two.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def alpha():\n"
        "-    return 1\n"
        "+    return 10\n"
        "@@ -7,2 +7,2 @@\n"
        " def gamma():\n"
        "-    return 99\n"
        "+    return 30\n"
    )
    mismatches = verify_diff_context(diff, tmp_path)
    assert len(mismatches) == 1
    m = mismatches[0]
    assert m.hunk_index == 1
    assert m.old_start == 7
    assert m.kind == "content_mismatch"


# ---------------------------------------------------------------------------
# Test 6 — malformed diff (no file headers) — defer to try_repair_patch.
# ---------------------------------------------------------------------------


def test_malformed_diff_no_file_headers_returns_empty(tmp_path):
    """A string with no ``diff --git`` / no hunk headers — the verifier
    returns ``[]`` and defers to ``try_repair_patch`` for structural
    repair. Its scope is content verification, not parsing recovery."""
    _write(tmp_path, "pkg/x.py", "something\n")
    mismatches = verify_diff_context(
        "I think we should change line 1 to say something else.\n",
        tmp_path,
    )
    assert mismatches == []

    # Another flavour of malformed — hunk-like but no file header.
    mismatches = verify_diff_context(
        "@@ -1,1 +1,1 @@\n-foo\n+bar\n",
        tmp_path,
    )
    assert mismatches == []


# ---------------------------------------------------------------------------
# Test 6b — observability-smoke regression (2026-04-23 b9b25c0):
# models often emit `--- a/... / +++ b/... / @@ ... @@` WITHOUT the
# `diff --git a/... b/...` header. `git apply` accepts both shapes;
# the verifier MUST accept both too. The django-10914 observe run
# exposed a parser gate that early-returned `[]` on missing
# `diff --git` — a false-negative on patches that were clearly
# content-mismatched against file bytes.
# ---------------------------------------------------------------------------


def test_headerless_unified_diff_is_parsed(tmp_path):
    """A unified diff without ``diff --git`` header MUST still be
    verified — ``git apply`` accepts that shape and models emit it."""
    # Fixture file where line 379 has content that does NOT match the
    # hunk body — the verifier should surface the mismatch.
    file_content = "\n".join(
        [f"# line {i}" for i in range(1, 400)]
        + ["DATETIME_INPUT_FORMATS = ["]          # line 400
    ) + "\n"
    _write(tmp_path, "pkg/settings.py", file_content)

    # NOTE: no ``diff --git`` prefix — just --- / +++ / @@.
    patch = (
        "--- a/pkg/settings.py\n"
        "+++ b/pkg/settings.py\n"
        "@@ -379,2 +379,2 @@\n"
        " # Default permissions for uploaded files.\n"
        "-FILE_UPLOAD_PERMISSIONS = None\n"
        "+FILE_UPLOAD_PERMISSIONS = 0o644\n"
    )
    mismatches = verify_diff_context(patch, tmp_path)
    assert len(mismatches) == 1, (
        f"Headerless unified diff should verify; got {mismatches!r}"
    )
    assert mismatches[0].kind == "content_mismatch"
    assert mismatches[0].file == "pkg/settings.py"
    assert mismatches[0].old_start == 379


# ---------------------------------------------------------------------------
# Test 7 — hunk-header line count mismatch → scope of counts-repair.
# ---------------------------------------------------------------------------


def test_hunk_header_count_mismatch_not_our_scope(tmp_path):
    """Hunk header says ``@@ -35,6 +35,7 @@`` but only 5 context/removed
    lines follow. This is a structural problem that belongs to
    ``swebench_patch_repair.repair_hunk_counts``. Verifier's scope is
    content-vs-file, not header-vs-body. Return empty list."""
    _write(
        tmp_path,
        "pkg/counts.py",
        "def foo():\n    return 1\n    return 2\n    return 3\n",
    )
    # Header claims -1,4 (4 old-side lines) but body has 3 ctx/del lines.
    diff = (
        "diff --git a/pkg/counts.py b/pkg/counts.py\n"
        "--- a/pkg/counts.py\n"
        "+++ b/pkg/counts.py\n"
        "@@ -1,4 +1,4 @@\n"
        " def foo():\n"
        "     return 1\n"
        "-    return 2\n"
        "+    return 20\n"
    )
    mismatches = verify_diff_context(diff, tmp_path)
    assert mismatches == []


# ---------------------------------------------------------------------------
# Test 8 — SR-converted unified diff (open question #3).
# ---------------------------------------------------------------------------


def test_sr_converted_diff_passes_when_search_matches(tmp_path):
    """The SR emission path synthesises a unified diff via
    ``_blocks_to_unified_diff`` after validating the SEARCH text
    against file bytes. A synthesised diff whose body already matches
    the file passes the verifier cleanly. Answers spec open question
    #3: YES, the verifier runs on SR-converted diffs."""
    _write(tmp_path, "pkg/sr.py", "def foo(x):\n    return x\n")
    # Shape matches what _blocks_to_unified_diff emits (hunk header
    # followed by context/minus/plus lines).
    diff = (
        "diff --git a/pkg/sr.py b/pkg/sr.py\n"
        "--- a/pkg/sr.py\n"
        "+++ b/pkg/sr.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def foo(x):\n"
        "-    return x\n"
        "+    return x + 1\n"
    )
    assert verify_diff_context(diff, tmp_path) == []


def test_sr_converted_diff_with_structurally_broken_header_empty(tmp_path):
    """A synthesised unified diff with a structurally broken hunk
    header (hunk body count inconsistent with header count) falls under
    the scope of ``try_repair_patch``, not the verifier. Empty list —
    the verifier doesn't repair structural issues."""
    _write(tmp_path, "pkg/sr2.py", "def foo(x):\n    return x\n")
    # Header claims -1,3 but only 2 ctx/minus lines follow.
    diff = (
        "diff --git a/pkg/sr2.py b/pkg/sr2.py\n"
        "--- a/pkg/sr2.py\n"
        "+++ b/pkg/sr2.py\n"
        "@@ -1,3 +1,3 @@\n"
        " def foo(x):\n"
        "-    return x\n"
        "+    return x + 1\n"
    )
    assert verify_diff_context(diff, tmp_path) == []


# ---------------------------------------------------------------------------
# Dataclass contract tests.
# ---------------------------------------------------------------------------


def test_hunkmismatch_is_frozen():
    """Per spec, ``HunkMismatch`` is ``@dataclass(frozen=True)``."""
    m = HunkMismatch(
        file="x",
        hunk_index=0,
        old_start=1,
        old_count=1,
        expected=["a"],
        actual=["b"],
        kind="content_mismatch",
        match_ratio=0.0,
    )
    with pytest.raises((AttributeError, Exception)):
        m.file = "y"  # type: ignore[misc]
