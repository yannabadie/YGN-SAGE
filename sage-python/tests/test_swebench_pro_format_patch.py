"""Cycle-13 E Tier 2.0 NO-API canary — Pro patch format adapter tests.

Per cgpro DESIGN E trap Q5 (2026-05-05, conv `cgpro_pi_mono_pivot_20260505`):
  SWE-bench Pro grader expects `{instance_id, patch, prefix?}` JSON list,
  NOT SWE-bench Lite's `{instance_id, model_name_or_path, model_patch}`
  shape. Validate the adapter produces the right shape BEFORE any API
  spend on Tier 2.1 arm D smoke.

Tests cover:
  - validate_record() rejects malformed dicts.
  - format_patch() produces the canonical Pro shape.
  - write_predictions() writes a JSON list with LF line endings.
  - Empty patch ("agent gave up") is a valid input (grader treats as
    non-resolution).
  - prefix is optional but, when present, must be str.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


# Load the script as a module since it lives under scripts/, not src/.
_SCRIPT_PATH = (
    Path(__file__).parent.parent
    / "scripts"
    / "swebench_pro_format_patch.py"
).resolve()


@pytest.fixture(scope="module")
def fmt_module():
    spec = importlib.util.spec_from_file_location(
        "swebench_pro_format_patch", _SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["swebench_pro_format_patch"] = mod
    spec.loader.exec_module(mod)
    return mod


# ── validate_record() ────────────────────────────────────────────────────────


def test_validate_record_accepts_minimal_shape(fmt_module):
    fmt_module.validate_record({"instance_id": "task-1", "patch": ""})


def test_validate_record_accepts_with_prefix(fmt_module):
    fmt_module.validate_record(
        {"instance_id": "task-1", "patch": "diff --git", "prefix": "run-a"}
    )


def test_validate_record_rejects_missing_instance_id(fmt_module):
    with pytest.raises(ValueError, match="missing required keys"):
        fmt_module.validate_record({"patch": ""})


def test_validate_record_rejects_missing_patch(fmt_module):
    with pytest.raises(ValueError, match="missing required keys"):
        fmt_module.validate_record({"instance_id": "task-1"})


def test_validate_record_rejects_lite_shape(fmt_module):
    """The trap this whole module exists to prevent: SWE-bench Lite
    keys (model_name_or_path, model_patch) silently passed to the
    Pro grader."""
    lite_record = {
        "instance_id": "task-1",
        "model_name_or_path": "claude-opus-4-7",
        "model_patch": "diff --git",
    }
    with pytest.raises(ValueError, match="unexpected keys"):
        fmt_module.validate_record(lite_record)


def test_validate_record_rejects_extra_keys(fmt_module):
    with pytest.raises(ValueError, match="unexpected keys"):
        fmt_module.validate_record(
            {"instance_id": "task-1", "patch": "", "extra": "noise"}
        )


def test_validate_record_rejects_non_str_instance_id(fmt_module):
    with pytest.raises(ValueError, match="instance_id must be non-empty str"):
        fmt_module.validate_record({"instance_id": 42, "patch": ""})


def test_validate_record_rejects_empty_instance_id(fmt_module):
    with pytest.raises(ValueError, match="instance_id must be non-empty str"):
        fmt_module.validate_record({"instance_id": "", "patch": ""})


def test_validate_record_rejects_non_str_patch(fmt_module):
    with pytest.raises(ValueError, match="patch must be str"):
        fmt_module.validate_record({"instance_id": "task-1", "patch": 123})


def test_validate_record_accepts_empty_patch(fmt_module):
    """Empty patch represents 'agent gave up' — valid Pro input.

    Per grader docstring: empty patch is treated as non-resolution.
    Important: the formatter MUST NOT reject these so we can record
    'agent failed' results in the same predictions.json as successful
    ones.
    """
    fmt_module.validate_record({"instance_id": "task-1", "patch": ""})


def test_validate_record_rejects_non_dict(fmt_module):
    with pytest.raises(ValueError, match="must be dict"):
        fmt_module.validate_record(["instance_id", "task-1"])


def test_validate_record_rejects_non_str_prefix(fmt_module):
    with pytest.raises(ValueError, match="prefix must be str"):
        fmt_module.validate_record(
            {"instance_id": "task-1", "patch": "", "prefix": 99}
        )


# ── format_patch() ───────────────────────────────────────────────────────────


def test_format_patch_minimal(fmt_module):
    record = fmt_module.format_patch("instance_xyz", "diff --git a b\n")
    assert record == {"instance_id": "instance_xyz", "patch": "diff --git a b\n"}
    assert "prefix" not in record


def test_format_patch_with_prefix(fmt_module):
    record = fmt_module.format_patch(
        "instance_xyz", "diff --git a b\n", prefix="ygn-sage-arm-d-smoke-001"
    )
    assert record["instance_id"] == "instance_xyz"
    assert record["patch"] == "diff --git a b\n"
    assert record["prefix"] == "ygn-sage-arm-d-smoke-001"


def test_format_patch_empty_patch_string(fmt_module):
    """Canary case: agent produced no patch — grader accepts."""
    record = fmt_module.format_patch("instance_xyz", "")
    assert record["patch"] == ""


# ── write_predictions() ──────────────────────────────────────────────────────


def test_write_predictions_writes_json_list(fmt_module, tmp_path: Path):
    output = tmp_path / "predictions.json"
    records = [
        fmt_module.format_patch("task-1", "diff a\n", prefix="run-a"),
        fmt_module.format_patch("task-2", "", prefix="run-a"),
    ]
    fmt_module.write_predictions(records, output)

    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert parsed[0]["instance_id"] == "task-1"
    assert parsed[0]["patch"] == "diff a\n"
    assert parsed[0]["prefix"] == "run-a"
    assert parsed[1]["patch"] == ""


def test_write_predictions_lf_only_line_endings(fmt_module, tmp_path: Path):
    """Per cgpro DESIGN E trap Q6: LF-only delimiter, NOT CRLF.

    The Pro grader runs in Modal/Docker where LF is canonical.
    Windows hosts must explicitly force LF.
    """
    output = tmp_path / "predictions.json"
    records = [fmt_module.format_patch("task-1", "diff\n")]
    fmt_module.write_predictions(records, output)
    raw = output.read_bytes()
    # The file must contain LF (0x0A), MUST NOT contain CRLF (0x0D 0x0A).
    assert b"\n" in raw, "no LF found — file may have CR-only line endings"
    assert b"\r\n" not in raw, "CRLF found — Pro grader requires LF only"


def test_write_predictions_round_trips_through_json(fmt_module, tmp_path: Path):
    """Sanity: writing then reading recovers the original records."""
    output = tmp_path / "predictions.json"
    records = [
        fmt_module.format_patch("a", "diff a\n"),
        fmt_module.format_patch("b", "diff b\n", prefix="x"),
    ]
    fmt_module.write_predictions(records, output)

    parsed = json.loads(output.read_text(encoding="utf-8"))
    # Re-wrap parsed dicts as records to check shape post-roundtrip.
    for raw in parsed:
        fmt_module.validate_record(raw)


def test_write_predictions_rejects_invalid_record_in_list(
    fmt_module, tmp_path: Path
):
    output = tmp_path / "predictions.json"
    invalid_record = {"instance_id": "task-1"}  # missing patch
    with pytest.raises(ValueError, match="missing required keys"):
        fmt_module.write_predictions([invalid_record], output)
    # File MUST NOT have been created (validation runs before write).
    assert not output.exists()


def test_write_predictions_handles_unicode(fmt_module, tmp_path: Path):
    """Pro accepts utf-8; ensure no escape conversion mangles non-ASCII
    patches (filenames with unicode, multi-byte chars in commit msgs).
    """
    output = tmp_path / "predictions.json"
    record = fmt_module.format_patch("task-1", "diff --git a/résumé.py b/résumé.py\n")
    fmt_module.write_predictions([record], output)

    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed[0]["patch"] == "diff --git a/résumé.py b/résumé.py\n"


# ── End-to-end synthetic canary (NO API, NO Docker) ──────────────────────────


def test_canary_synthetic_pipeline_pro_shape(fmt_module, tmp_path: Path):
    """The whole point of Tier 2.0: prove a synthetic patch flows
    through the formatter and produces a valid predictions.json that
    matches the Pro grader's expected schema (per its docstring).

    No grader is invoked — that requires Docker. This test verifies
    the SHAPE the grader will accept.
    """
    output = tmp_path / "predictions.json"

    # Simulate 3 tasks: 1 successful patch, 1 empty (gave up), 1 with
    # prefix override.
    records = [
        fmt_module.format_patch(
            "django__django-12345",
            "diff --git a/foo.py b/foo.py\nindex 1..2 100644\n--- a/foo.py\n+++ b/foo.py\n@@ -1,1 +1,1 @@\n-old\n+new\n",
            prefix="ygn-sage-arm-d-smoke-canary",
        ),
        fmt_module.format_patch(
            "astropy__astropy-67890",
            "",
            prefix="ygn-sage-arm-d-smoke-canary",
        ),
        fmt_module.format_patch(
            "numpy__numpy-11111",
            "diff --git a/bar.py b/bar.py\n+ x = 1\n",
        ),
    ]

    fmt_module.write_predictions(records, output)

    # Re-parse via canonical Python json — same library the grader uses.
    parsed = json.loads(output.read_text(encoding="utf-8"))

    # Per Pro grader docstring: list of records with instance_id +
    # patch (+ optional prefix).
    assert isinstance(parsed, list)
    assert len(parsed) == 3
    for record in parsed:
        assert "instance_id" in record
        assert "patch" in record
        # Per grader: prefix is optional, but when present must be str.
        if "prefix" in record:
            assert isinstance(record["prefix"], str)
