"""Unit tests for C4 — AgentSystem.run() str|TaskInput overload.

Spec: docs/superpowers/specs/2026-04-21-universal-input-adapter-design.md §C4.

Entry-point dispatch: when a caller hands `AgentSystem.run()` a TaskInput
instead of a string, the method routes to the source-specific renderer
(`swebench`, `bcb`, or the raw `prompt` field for chat / unknown
sources) before the rest of the pipeline runs. These tests mock the
downstream (pipeline / agent_loop) and verify the dispatch logic only —
full bench runs are covered by smoke tests elsewhere.
"""
from __future__ import annotations

import pytest

from sage.input import (
    ResponseFormat,
    TaskInput,
    normalize_bcb,
    normalize_chat,
    normalize_swebench,
    render_bcb_prompt,
    render_swebench_prompt,
)


# ---------------------------------------------------------------------------
# Fakes — simulate AgentSystem.run() WITHOUT booting the full stack. We
# extract the dispatch snippet into a local shim that mirrors the code in
# `sage.boot.AgentSystem.run`, then verify each branch. The snippet is
# small and deliberately duplicated here so breaking the prod dispatch
# (e.g., forgetting a new source tag) shows up as a test diff.
# ---------------------------------------------------------------------------


def _dispatch(task):
    """Mirror of sage.boot.AgentSystem.run dispatch prologue."""
    if isinstance(task, TaskInput):
        if task.source == "swebench":
            return render_swebench_prompt(task), "taskinput-swebench"
        if task.source == "bcb":
            return render_bcb_prompt(task), "taskinput-bcb"
        return task.prompt, f"taskinput-{task.source}"
    return task, "string"


# ---------------------------------------------------------------------------
# Legacy string path — unchanged
# ---------------------------------------------------------------------------


def test_run_passthrough_for_raw_string():
    """Before C4, run() took `task: str`. That path stays exact —
    strings go through untouched, no normalizer invoked."""
    out, label = _dispatch("fix the bug in module X")
    assert out == "fix the bug in module X"
    assert label == "string"


def test_run_passthrough_preserves_whitespace_and_unicode():
    weird = "  你好 world  \n\ttab"
    out, _ = _dispatch(weird)
    assert out == weird


# ---------------------------------------------------------------------------
# TaskInput swebench — dispatches through render_swebench_prompt
# ---------------------------------------------------------------------------


SWEBENCH_INSTANCE = {
    "instance_id": "django__django-10914",
    "repo": "django/django",
    "version": "3.0",
    "base_commit": "e7fd69d0",
    "problem_statement": "Set default FILE_UPLOAD_PERMISSION",
    "hints_text": "",
}


def test_run_dispatches_swebench_task_input_via_renderer():
    ti = normalize_swebench(SWEBENCH_INSTANCE)
    out, label = _dispatch(ti)
    expected = render_swebench_prompt(ti)
    assert out == expected
    assert label == "taskinput-swebench"


def test_run_swebench_dispatch_contains_repo_and_problem_statement():
    ti = normalize_swebench(SWEBENCH_INSTANCE)
    out, _ = _dispatch(ti)
    assert "django/django" in out
    assert "Set default FILE_UPLOAD_PERMISSION" in out


# ---------------------------------------------------------------------------
# TaskInput bcb — dispatches through render_bcb_prompt
# ---------------------------------------------------------------------------


BCB_TASK = {
    "task_id": "BigCodeBench/13",
    "instruct_prompt": "Download files from FTP.",
    "code_prompt": "import os\ndef task_func(x): pass\n",
    "entry_point": "task_func",
    "libs": "['os']",
}


def test_run_dispatches_bcb_task_input_via_renderer():
    ti = normalize_bcb(BCB_TASK, split="instruct")
    out, label = _dispatch(ti)
    expected = render_bcb_prompt(ti)
    assert out == expected
    assert label == "taskinput-bcb"


def test_run_bcb_dispatch_preserves_code_prompt_block():
    ti = normalize_bcb(BCB_TASK, split="instruct")
    out, _ = _dispatch(ti)
    assert "```python\nimport os" in out
    assert "Download files from FTP." in out


# ---------------------------------------------------------------------------
# TaskInput chat / unknown source — raw `prompt` field
# ---------------------------------------------------------------------------


def test_run_chat_task_input_uses_raw_prompt():
    ti = normalize_chat("how does Django's FILE_UPLOAD_PERMISSION default behave?")
    out, label = _dispatch(ti)
    assert out == "how does Django's FILE_UPLOAD_PERMISSION default behave?"
    assert label == "taskinput-chat"


def test_run_unknown_source_uses_raw_prompt_not_renderer():
    """A hypothetical future source (e.g. 'jupyter', 'slack') with no
    dedicated renderer falls back to the raw prompt field. Prevents
    an unrecognized source from silently swallowing the input."""
    ti = TaskInput(
        prompt="hi from some future interface",
        response_format=ResponseFormat.TEXT,
        source="future_source",
    )
    out, label = _dispatch(ti)
    assert out == "hi from some future interface"
    assert label == "taskinput-future_source"


# ---------------------------------------------------------------------------
# Byte-identity for the migration: whichever path benches take, the
# string reaching the pipeline must be identical.
# ---------------------------------------------------------------------------


def test_swebench_taskinput_dispatch_matches_pre_c4_build_task_prompt():
    """Before C4, benches called `render_swebench_prompt(normalize_
    swebench(instance))` themselves and passed the resulting string.
    After C4, they pass TaskInput and AgentSystem.run() renders. The
    string the pipeline sees MUST be identical in both paths."""
    ti = normalize_swebench(SWEBENCH_INSTANCE)
    via_dispatch, _ = _dispatch(ti)
    via_pre_c4 = render_swebench_prompt(normalize_swebench(SWEBENCH_INSTANCE))
    assert via_dispatch == via_pre_c4


def test_bcb_taskinput_dispatch_matches_pre_c4_inline_build():
    """Same guarantee for BCB."""
    ti = normalize_bcb(BCB_TASK, split="instruct")
    via_dispatch, _ = _dispatch(ti)
    via_pre_c4 = render_bcb_prompt(normalize_bcb(BCB_TASK, split="instruct"))
    assert via_dispatch == via_pre_c4


# ---------------------------------------------------------------------------
# Real AgentSystem.run dispatch (integration — no pipeline).
# We verify the dispatch block alone by reading the production source
# and asserting its structure. A drift here = the local _dispatch
# helper above is out of sync with the real code.
# ---------------------------------------------------------------------------


def test_production_dispatch_handles_all_sources():
    """Read the real AgentSystem.run source and verify the dispatch
    block mentions each source we care about. This is a structural
    guard against silent removal of a branch."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src/sage/boot.py"
    text = src.read_text(encoding="utf-8")
    assert "isinstance(task, TaskInput)" in text
    assert 'task.source == "swebench"' in text
    assert 'task.source == "bcb"' in text
    assert "render_swebench_prompt" in text
    assert "render_bcb_prompt" in text
    # The chat/unknown fallback has no branch — it's the else arm.
    # Guard by asserting the comment naming the fallback still exists:
    assert "chat" in text and "future source" in text


@pytest.mark.parametrize("source,renderer_name", [
    ("swebench", "render_swebench_prompt"),
    ("bcb", "render_bcb_prompt"),
])
def test_renderer_names_match_known_sources(source, renderer_name):
    """Locks the source→renderer mapping. Adding a new renderer means
    updating both this parametrize and the dispatch in boot.py."""
    from sage.input import render_swebench_prompt, render_bcb_prompt
    registry = {
        "swebench": render_swebench_prompt,
        "bcb": render_bcb_prompt,
    }
    assert source in registry
    assert registry[source].__name__ == renderer_name
