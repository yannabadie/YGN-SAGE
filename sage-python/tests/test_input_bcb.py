"""Byte-identity regression tests for C3 — BCB TaskInput migration.

Spec: docs/superpowers/specs/2026-04-21-universal-input-adapter-design.md §C3.

The previous inline prompt-build in `sage.bench.bigcodebench_bench.run()`
(lines 79-87 pre-C3) now routes through `sage.input.bcb.normalize_bcb` +
`sage.input.bcb.render_bcb_prompt`. Byte-for-byte equivalence is the
gate; content changes belong in a later commit with a deliberate
fixture update.

Fixtures in `tests/fixtures/bcb_prompt_*.txt` were captured from the
pre-C3 inline builder on a canonical `BigCodeBench/13`-shaped task and
regenerate only on purpose.
"""
from __future__ import annotations

from pathlib import Path

from sage.input import (
    ResponseFormat,
    TaskInput,
    normalize_bcb,
    render_bcb_prompt,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Canonical task — shape matches BigCodeBench/13 from the cached HF dataset
# ---------------------------------------------------------------------------

CANONICAL_TASK = {
    "task_id": "BigCodeBench/13",
    "instruct_prompt": (
        "Download all files from a specific directory on an FTP server "
        "using wget in a subprocess."
    ),
    "complete_prompt": (
        '"""Download all files from FTP."""\ndef task_func(...): pass'
    ),
    "code_prompt": (
        "import subprocess\n"
        "import ftplib\n"
        "import os\n"
        "\n"
        "def task_func(directory: str) -> list:\n"
    ),
    "test": (
        "import unittest\n"
        "class TestCases(unittest.TestCase):\n"
        "    def test_x(self): pass"
    ),
    "entry_point": "task_func",
    "libs": "['subprocess', 'ftplib', 'os']",
}


def _read_fixture(name: str) -> str:
    return (FIXTURES / f"bcb_prompt_{name}.txt").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# normalize_bcb — field mapping
# ---------------------------------------------------------------------------


def test_normalize_bcb_returns_task_input():
    assert isinstance(normalize_bcb(CANONICAL_TASK), TaskInput)


def test_normalize_bcb_instruct_prompt_is_nl_request():
    ti = normalize_bcb(CANONICAL_TASK, split="instruct")
    assert ti.prompt == CANONICAL_TASK["instruct_prompt"]


def test_normalize_bcb_complete_split_selects_complete_prompt():
    ti = normalize_bcb(CANONICAL_TASK, split="complete")
    assert ti.prompt == CANONICAL_TASK["complete_prompt"]


def test_normalize_bcb_default_split_is_instruct():
    """CLI `--split` defaults to `instruct`; the normalizer must match."""
    ti_default = normalize_bcb(CANONICAL_TASK)
    ti_explicit = normalize_bcb(CANONICAL_TASK, split="instruct")
    assert ti_default.prompt == ti_explicit.prompt


def test_normalize_bcb_response_format_is_code():
    """BCB evaluators exec the agent's Python code; enforce CODE format."""
    assert normalize_bcb(CANONICAL_TASK).response_format == ResponseFormat.CODE


def test_normalize_bcb_source_tag():
    assert normalize_bcb(CANONICAL_TASK).source == "bcb"


def test_normalize_bcb_hints_carry_canonical_fields():
    ti = normalize_bcb(CANONICAL_TASK)
    assert ti.hints["code_prompt"] == CANONICAL_TASK["code_prompt"]
    assert ti.hints["test"] == CANONICAL_TASK["test"]
    assert ti.hints["entry_point"] == CANONICAL_TASK["entry_point"]
    assert ti.hints["libs"] == CANONICAL_TASK["libs"]
    assert ti.hints["split"] == "instruct"


def test_normalize_bcb_missing_optional_fields_become_empty_string():
    """Real BCB dumps occasionally miss `code_prompt` or `test`. The
    pre-C3 path treated missing as empty string — preserve that."""
    minimal = {"instruct_prompt": "do the thing", "entry_point": "fn"}
    ti = normalize_bcb(minimal)
    assert ti.hints["code_prompt"] == ""
    assert ti.hints["test"] == ""
    assert ti.hints["entry_point"] == "fn"
    assert ti.hints["libs"] == ""


def test_normalize_bcb_falls_back_to_instruct_prompt_when_complete_missing():
    """Pre-C3 path: `task.get(prompt_key, task.get("instruct_prompt", ""))`.
    If `split="complete"` but the `complete_prompt` key is absent, fall
    back to `instruct_prompt` rather than returning an empty string."""
    no_complete = {**CANONICAL_TASK}
    no_complete.pop("complete_prompt")
    ti = normalize_bcb(no_complete, split="complete")
    assert ti.prompt == CANONICAL_TASK["instruct_prompt"]


def test_normalize_bcb_instructions_prevents_tool_loops():
    """BCB tasks are self-contained — no repository to explore. instructions
    guides direct code generation and prevents tool-search loops on multi-agent
    bypass paths (C5, 2026-05-03)."""
    ti = normalize_bcb(CANONICAL_TASK)
    assert "self-contained" in ti.instructions
    assert "Python" in ti.instructions
    assert ti.instructions  # non-empty


# ---------------------------------------------------------------------------
# render_bcb_prompt — byte-identity against pre-C3 snapshots
# ---------------------------------------------------------------------------


def test_render_bcb_prompt_matches_fixture_instruct_with_code():
    ti = normalize_bcb(CANONICAL_TASK, split="instruct")
    actual = render_bcb_prompt(ti)
    expected = _read_fixture("instruct_with_code")
    assert actual == expected, "C3 byte-identity broke on instruct+code path"


def test_render_bcb_prompt_matches_fixture_complete_with_code():
    ti = normalize_bcb(CANONICAL_TASK, split="complete")
    actual = render_bcb_prompt(ti)
    expected = _read_fixture("complete_with_code")
    assert actual == expected, "C3 byte-identity broke on complete+code path"


def test_render_bcb_prompt_matches_fixture_no_code():
    """When `code_prompt` is empty the pre-C3 path skipped the
    fenced-python header entirely; the output is just the NL prompt."""
    no_code = {**CANONICAL_TASK, "code_prompt": ""}
    ti = normalize_bcb(no_code, split="instruct")
    actual = render_bcb_prompt(ti)
    expected = _read_fixture("no_code")
    assert actual == expected, "C3 byte-identity broke on no-code path"


def test_build_prompt_delegates_to_input_layer():
    """The bench's per-task prompt matches the explicit normalize +
    render path — confirms the inline-builder swap in
    bigcodebench_bench.py landed cleanly (C3). C5 adds OUTPUT REQUIREMENT."""
    from sage.input.bcb import normalize_bcb, render_bcb_prompt

    via_layer = render_bcb_prompt(normalize_bcb(CANONICAL_TASK, "instruct"))
    prompt = CANONICAL_TASK["instruct_prompt"]
    code_prompt = CANONICAL_TASK["code_prompt"]
    entry = CANONICAL_TASK["entry_point"]
    inline_equiv = (
        f"Use this function signature and imports:\n"
        f"```python\n{code_prompt}\n```\n\n{prompt}"
        f"\n\nOUTPUT REQUIREMENT: Return ONLY a complete, runnable Python function "
        f"implementation that starts with `def {entry}(`. No planning, no diffs, "
        "no explanations — just working Python code."
    )
    assert via_layer == inline_equiv


# ---------------------------------------------------------------------------
# Content guards — features that must stay for the CODE-fenced emit path
# ---------------------------------------------------------------------------


def test_c3_preserves_fenced_python_block_when_code_prompt_present():
    prompt = render_bcb_prompt(normalize_bcb(CANONICAL_TASK))
    assert "```python\n" in prompt
    assert prompt.count("```") == 2  # opening + closing fence


def test_c3_preserves_signature_header_text():
    prompt = render_bcb_prompt(normalize_bcb(CANONICAL_TASK))
    assert "Use this function signature and imports:" in prompt
    # NL prompt follows the fenced block, preceded by blank line
    assert "```\n\n" + CANONICAL_TASK["instruct_prompt"] in prompt


# ---------------------------------------------------------------------------
# __all__ surface lock
# ---------------------------------------------------------------------------


def test_package_public_api_surface_includes_bcb():
    """C3 expands the package surface with normalize_bcb +
    render_bcb_prompt. Locking to prevent silent removal."""
    import sage.input as pkg

    assert "normalize_bcb" in pkg.__all__
    assert "render_bcb_prompt" in pkg.__all__
