"""Reason/outcome tests for the SWE-bench diff verifier."""
from __future__ import annotations

from pathlib import Path

from sage.bench.swebench_diff_verifier import (
    verify_diff_context,
    verify_diff_context_with_reasons,
)


def _write(repo: Path, relpath: str, content: str) -> None:
    target = repo / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def test_diff_verifier_reason_clean(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/mod.py", "old\n")
    diff = (
        "--- a/pkg/mod.py\n"
        "+++ b/pkg/mod.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )

    result = verify_diff_context_with_reasons(diff, tmp_path)

    assert result.outcome == "clean"
    assert result.reasons == ["clean"]
    assert result.mismatches == []


def test_diff_verifier_reason_content_mismatch(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/mod.py", "def foo():\n    return 99\n")
    diff = (
        "--- a/pkg/mod.py\n"
        "+++ b/pkg/mod.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def foo():\n"
        "-    return 1\n"
        "+    return 2\n"
    )

    result = verify_diff_context_with_reasons(diff, tmp_path)

    assert result.outcome == "content_mismatch"
    assert "content_mismatch" in result.reasons
    assert result.mismatches[0].kind == "content_mismatch"


def test_diff_verifier_reason_fuzzy_below_threshold(tmp_path: Path) -> None:
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

    result = verify_diff_context_with_reasons(diff, tmp_path)

    assert result.outcome == "fuzzy_below_threshold"
    assert "fuzzy_below_threshold" in result.reasons
    assert result.mismatches[0].kind == "fuzzy_below_threshold"


def test_diff_verifier_reason_file_missing(tmp_path: Path) -> None:
    diff = (
        "--- a/pkg/ghost.py\n"
        "+++ b/pkg/ghost.py\n"
        "@@ -1,2 +1,2 @@\n"
        " line 1\n"
        "-line 2\n"
        "+line 2 modified\n"
    )

    result = verify_diff_context_with_reasons(diff, tmp_path)

    assert result.outcome == "file_missing"
    assert "file_missing" in result.reasons
    assert result.mismatches[0].kind == "file_missing"


def test_diff_verifier_reason_malformed_hunk_header(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/mod.py", "x\n")
    diff = (
        "--- a/pkg/mod.py\n"
        "+++ b/pkg/mod.py\n"
        "@@ -not-a-range +1,1 @@\n"
        "-x\n"
        "+y\n"
    )

    result = verify_diff_context_with_reasons(diff, tmp_path)

    assert result.outcome == "malformed_hunk_header"
    assert result.reasons == ["malformed_hunk_header"]
    assert result.mismatches == []


def test_diff_verifier_reason_hunk_body_count_mismatch(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/mod.py", "a\nb\n")
    diff = (
        "--- a/pkg/mod.py\n"
        "+++ b/pkg/mod.py\n"
        "@@ -1,3 +1,1 @@\n"
        " a\n"
        "-b\n"
        "+x\n"
    )

    result = verify_diff_context_with_reasons(diff, tmp_path)

    assert result.outcome == "hunk_body_count_mismatch"
    assert "hunk_body_count_mismatch" in result.reasons
    assert result.mismatches == []


def test_diff_verifier_reason_file_creation_or_deletion(tmp_path: Path) -> None:
    diff = (
        "--- /dev/null\n"
        "+++ b/pkg/new.py\n"
        "@@ -0,0 +1,1 @@\n"
        '+print("new")\n'
    )

    result = verify_diff_context_with_reasons(diff, tmp_path)

    assert result.outcome == "file_creation_or_deletion"
    assert "file_creation_or_deletion" in result.reasons
    assert result.mismatches == []


def test_diff_verifier_reason_not_unified_diff(tmp_path: Path) -> None:
    diff = "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE\n"

    result = verify_diff_context_with_reasons(diff, tmp_path)

    assert result.outcome == "not_unified_diff"
    assert result.reasons == ["not_unified_diff"]
    assert result.mismatches == []


def test_diff_verifier_reason_unsupported_no_opinion_for_hunk_without_file_header(
    tmp_path: Path,
) -> None:
    diff = "@@ -1,1 +1,1 @@\n-old\n+new\n"

    result = verify_diff_context_with_reasons(diff, tmp_path)

    assert result.outcome == "unsupported_no_opinion"
    assert "unsupported_no_opinion" in result.reasons
    assert result.mismatches == []


def test_verify_diff_context_legacy_wrapper_returns_only_mismatches(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "pkg/mod.py", "def foo():\n    return 99\n")
    diff = (
        "--- a/pkg/mod.py\n"
        "+++ b/pkg/mod.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def foo():\n"
        "-    return 1\n"
        "+    return 2\n"
    )

    assert verify_diff_context(diff, tmp_path) == verify_diff_context_with_reasons(
        diff, tmp_path
    ).mismatches


def test_diff_verifier_mixed_reasons_roll_up_to_strongest_outcome(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "pkg/mod.py", "a\nb\nc\n")
    diff = (
        "--- a/pkg/mod.py\n"
        "+++ b/pkg/mod.py\n"
        "@@ -1,1 +1,1 @@\n"
        "-a\n"
        "+x\n"
        "@@ -2,3 +2,1 @@\n"
        " b\n"
        "-c\n"
        "+y\n"
    )

    result = verify_diff_context_with_reasons(diff, tmp_path)

    assert "clean" in result.reasons
    assert "hunk_body_count_mismatch" in result.reasons
    assert result.outcome == "hunk_body_count_mismatch"
