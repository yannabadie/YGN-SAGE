"""Byte-identity regression tests for C2a — SWE-bench TaskInput migration.

Spec: docs/superpowers/specs/2026-04-21-universal-input-adapter-design.md §C2a.

The previous inline `_TASK_TEMPLATE` / `_build_task_prompt` pair in
`sage.bench.swebench_bench` now routes through
`sage.input.swebench.normalize_swebench` +
`sage.input.swebench.render_swebench_prompt`. This commit is a pure
refactor: the prompt text seen by the model MUST be byte-for-byte
identical to what the bench emitted before the swap. Fixtures in
`tests/fixtures/swebench_prompt_*.txt` capture the expected output
for a canonical instance and are regenerated deliberately only when
content changes ship (C2b will be the first such commit).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sage.bench.swebench_bench import _build_task_prompt
from sage.input import (
    ResponseFormat,
    SWEBENCH_SYSTEM_TEMPLATE,
    TaskInput,
    normalize_swebench,
    render_swebench_prompt,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Canonical instance used by the byte-identity tests
# ---------------------------------------------------------------------------

CANONICAL_INSTANCE = {
    "instance_id": "django__django-10914",
    "repo": "django/django",
    "version": "3.0",
    "base_commit": "e7fd69d051eaa67cb17f172a39b57253e9cb831a",
    "problem_statement": (
        "Set default FILE_UPLOAD_PERMISSION to 0o644.\n"
        "Hello, the default setting of FILE_UPLOAD_PERMISSION (None) can "
        "lead to surprising behaviour."
    ),
    "hints_text": "I would expect some extra hints from issue comments.",
}


def _read_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# normalize_swebench — field mapping
# ---------------------------------------------------------------------------


def test_normalize_swebench_returns_task_input():
    ti = normalize_swebench(CANONICAL_INSTANCE)
    assert isinstance(ti, TaskInput)


def test_normalize_swebench_prompt_is_problem_statement():
    """The NL request the model reads is the issue description.
    Repo / commit / version are context, not the request."""
    ti = normalize_swebench(CANONICAL_INSTANCE)
    assert ti.prompt == CANONICAL_INSTANCE["problem_statement"]


def test_normalize_swebench_response_format_is_patch():
    """SWE-bench graders consume unified diffs — enforce the format."""
    ti = normalize_swebench(CANONICAL_INSTANCE)
    assert ti.response_format == ResponseFormat.PATCH


def test_normalize_swebench_source_tag():
    assert normalize_swebench(CANONICAL_INSTANCE).source == "swebench"


def test_normalize_swebench_hints_carry_context():
    ti = normalize_swebench(CANONICAL_INSTANCE)
    assert ti.hints["repo"] == "django/django"
    assert ti.hints["version"] == "3.0"
    assert ti.hints["base_commit"].startswith("e7fd69d0")
    assert ti.hints["hints_text"] == CANONICAL_INSTANCE["hints_text"]


def test_normalize_swebench_version_defaults_to_unknown():
    """The SWE-bench Lite split occasionally ships instances without a
    `version` key. The renderer needs *some* string to drop into the
    template — falling back to "unknown" preserves prior behavior."""
    stripped = dict(CANONICAL_INSTANCE)
    stripped.pop("version")
    assert normalize_swebench(stripped).hints["version"] == "unknown"


def test_normalize_swebench_missing_hints_text_becomes_empty_string():
    """A missing or None `hints_text` must normalize to "" so the
    renderer's truthiness check produces the same empty hints_section
    as the pre-C2a path."""
    for missing in ({"instance_id": "x"}, {"hints_text": None}, {"hints_text": ""}):
        stripped = {**CANONICAL_INSTANCE, **missing}
        # hints_text=None variant:
        stripped.setdefault("hints_text", None)
        stripped = dict(CANONICAL_INSTANCE)
        stripped.pop("hints_text", None)
        assert normalize_swebench(stripped).hints["hints_text"] == ""


def test_normalize_swebench_instructions_is_full_template():
    """C4 will render the template through a generic layered builder.
    Until then we still want the template text accessible on the
    TaskInput for downstream observability (dumping prompts to
    traces, etc.)."""
    ti = normalize_swebench(CANONICAL_INSTANCE)
    assert ti.instructions == SWEBENCH_SYSTEM_TEMPLATE


# ---------------------------------------------------------------------------
# render_swebench_prompt — byte-identity
# ---------------------------------------------------------------------------


def test_render_swebench_prompt_matches_fixture_with_hints():
    """Byte-for-byte match against the pre-C2a snapshot. Any drift
    means the refactor leaked a content change — C2a is meant to be
    a pure refactor; content changes belong in C2b."""
    ti = normalize_swebench(CANONICAL_INSTANCE)
    actual = render_swebench_prompt(ti)
    expected = _read_fixture("swebench_prompt_with_hints.txt")
    assert actual == expected, "C2a byte-identity broke — fixture drift detected"


def test_render_swebench_prompt_matches_fixture_without_hints():
    """Same byte-identity guarantee on the no-hints branch."""
    no_hints_instance = dict(CANONICAL_INSTANCE, hints_text="")
    ti = normalize_swebench(no_hints_instance)
    actual = render_swebench_prompt(ti)
    expected = _read_fixture("swebench_prompt_no_hints.txt")
    assert actual == expected


def test_build_task_prompt_delegates_to_input_layer():
    """The bench's top-level entry point produces the same string as
    the explicit normalize + render path — confirms the call-site
    swap in swebench_bench.py landed cleanly."""
    via_bench = _build_task_prompt(CANONICAL_INSTANCE)
    via_layer = render_swebench_prompt(normalize_swebench(CANONICAL_INSTANCE))
    assert via_bench == via_layer


# ---------------------------------------------------------------------------
# Content guards — these tests intentionally lock specific phrasing so
# that C2b's softening becomes a visible fixture edit, not silent drift.
# ---------------------------------------------------------------------------


def test_c2a_preserves_must_bash_clause():
    """C2a is a pure refactor — the "MUST make at least THREE
    distinct execute_bash calls" anti-affordance is locked until
    C2b rewrites it. If this test fails in a commit that isn't
    labeled C2b, the refactor leaked content."""
    ti = normalize_swebench(CANONICAL_INSTANCE)
    prompt = render_swebench_prompt(ti)
    assert "MUST make at least THREE distinct execute_bash calls" in prompt


def test_c2a_preserves_patch_format_strict_block():
    """The grader-facing patch format contract stays locked through
    C2a. Weakening the "Hard requirements" bullet list would break
    the Docker eval's patch validator."""
    prompt = render_swebench_prompt(normalize_swebench(CANONICAL_INSTANCE))
    assert "## Patch Format — Strict" in prompt
    assert "diff --git" in prompt
    assert "MUST be correct" in prompt  # in the hunk-ranges bullet


def test_hints_section_absent_when_hints_text_empty():
    ti = normalize_swebench(dict(CANONICAL_INSTANCE, hints_text=""))
    prompt = render_swebench_prompt(ti)
    assert "## Hints (from the issue comments)" not in prompt


def test_hints_section_present_when_hints_text_provided():
    prompt = render_swebench_prompt(normalize_swebench(CANONICAL_INSTANCE))
    assert "## Hints (from the issue comments)" in prompt
    assert CANONICAL_INSTANCE["hints_text"] in prompt


def test_hints_section_strips_whitespace_padded_hints():
    """A `hints_text` of pure whitespace must NOT trigger the
    section (matches the pre-C2a `hints.strip()` check)."""
    ti = normalize_swebench(dict(CANONICAL_INSTANCE, hints_text="   \n\n  \t"))
    prompt = render_swebench_prompt(ti)
    assert "## Hints (from the issue comments)" not in prompt


# ---------------------------------------------------------------------------
# Required fields — fail loudly on malformed input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing_key", ["problem_statement", "repo", "base_commit"])
def test_normalize_swebench_raises_on_missing_required_key(missing_key):
    """Missing a truly required field is a bug in the dataset loader,
    not something to paper over with a default. The pre-C2a path
    KeyError'd on these; we preserve that loudness."""
    broken = dict(CANONICAL_INSTANCE)
    broken.pop(missing_key)
    with pytest.raises(KeyError):
        normalize_swebench(broken)
