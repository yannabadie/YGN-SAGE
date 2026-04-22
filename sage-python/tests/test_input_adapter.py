"""Unit tests for the universal input adapter (C1).

Spec: docs/superpowers/specs/2026-04-21-universal-input-adapter-design.md
Scope: new `sage.input` package only — no bench / pipeline / perceive
changes yet. This commit is user-visible-change-free and safe to ship
alongside a running smoke.
"""
from __future__ import annotations

import pytest

from sage.input import CHAT_DEFAULT_TOOLS, ResponseFormat, TaskInput, normalize_chat
from sage.input.chat import _BASH_OPT_IN_ENV


# ---------------------------------------------------------------------------
# ResponseFormat enum
# ---------------------------------------------------------------------------


def test_response_format_has_all_five_members():
    """Every format the spec calls out must exist."""
    expected = {"TEXT", "CODE", "PATCH", "JSON", "SEARCH_REPLACE"}
    assert {m.name for m in ResponseFormat} == expected


def test_response_format_values_are_lowercase_strings():
    """Values are the wire-format string (used in logs, JSON payloads)."""
    assert ResponseFormat.TEXT.value == "text"
    assert ResponseFormat.CODE.value == "code"
    assert ResponseFormat.PATCH.value == "patch"
    assert ResponseFormat.JSON.value == "json"
    assert ResponseFormat.SEARCH_REPLACE.value == "search_replace"


def test_response_format_is_str_compatible():
    """Subclassing `str` means `ResponseFormat.TEXT == "text"` holds, which
    simplifies JSON serialization and equality checks in benches."""
    assert ResponseFormat.TEXT == "text"
    assert ResponseFormat.PATCH == "patch"


# ---------------------------------------------------------------------------
# TaskInput dataclass
# ---------------------------------------------------------------------------


def test_task_input_minimum_construction():
    """Only `prompt` is required; everything else has a safe default."""
    ti = TaskInput(prompt="hello")
    assert ti.prompt == "hello"


def test_task_input_defaults_match_spec():
    ti = TaskInput(prompt="x")
    assert ti.response_format == ResponseFormat.TEXT
    assert ti.hints == {}
    assert ti.instructions == ""
    assert ti.tools_filter is None
    assert ti.expected_length_hint == 0
    assert ti.source == "chat"


def test_task_input_hints_default_not_shared_across_instances():
    """Classic dataclass footgun: a mutable default shared by reference
    would leak mutations between instances. `field(default_factory=dict)`
    prevents this; this test locks the invariant."""
    a = TaskInput(prompt="a")
    b = TaskInput(prompt="b")
    a.hints["x"] = 1
    assert "x" not in b.hints


def test_task_input_all_fields_overridable():
    ti = TaskInput(
        prompt="fix the bug",
        response_format=ResponseFormat.PATCH,
        hints={"repo": "foo/bar", "base_commit": "abc123"},
        instructions="MUST produce a unified diff.",
        tools_filter=["execute_bash", "search_exocortex"],
        expected_length_hint=50,
        source="swebench",
    )
    assert ti.prompt == "fix the bug"
    assert ti.response_format == ResponseFormat.PATCH
    assert ti.hints["repo"] == "foo/bar"
    assert ti.instructions.startswith("MUST")
    assert ti.tools_filter == ["execute_bash", "search_exocortex"]
    assert ti.expected_length_hint == 50
    assert ti.source == "swebench"


def test_task_input_tools_filter_none_vs_empty_list():
    """`None` means "no filter, all tools"; `[]` means "no tools at all".
    The distinction matters for benches that want to disable tools."""
    no_filter = TaskInput(prompt="x")
    empty_filter = TaskInput(prompt="x", tools_filter=[])
    assert no_filter.tools_filter is None
    assert empty_filter.tools_filter == []
    assert no_filter.tools_filter != empty_filter.tools_filter


# ---------------------------------------------------------------------------
# normalize_chat — Q1 (bash OFF by default) + Q3 (no format enforcement)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_bash_env(monkeypatch):
    """Every test starts with SAGE_CHAT_ALLOW_BASH unset so that chat
    defaults are exercised unless a test explicitly opts in."""
    monkeypatch.delenv(_BASH_OPT_IN_ENV, raising=False)


def test_normalize_chat_returns_task_input():
    ti = normalize_chat("hello world")
    assert isinstance(ti, TaskInput)


def test_normalize_chat_preserves_prompt_verbatim():
    """No trimming, no normalization — whatever the user typed is what
    lands in `prompt`. Uppercase, whitespace, unicode all round-trip."""
    message = "  Hello, SAGE! Can you 你好 fix bug #42?  "
    ti = normalize_chat(message)
    assert ti.prompt == message


def test_normalize_chat_response_format_is_text():
    """Q3 — chat mode never forces a format."""
    assert normalize_chat("anything").response_format == ResponseFormat.TEXT


def test_normalize_chat_source_is_chat():
    assert normalize_chat("hi").source == "chat"


def test_normalize_chat_tools_filter_excludes_bash_by_default():
    """Q1 — bash must NOT leak into chat sessions without explicit
    opt-in. This is the core security/safety property of chat mode."""
    ti = normalize_chat("ls")
    assert ti.tools_filter is not None
    assert "bash" not in ti.tools_filter
    assert "execute_bash" not in ti.tools_filter


def test_normalize_chat_tools_filter_excludes_mutation_tools():
    """Default allowlist is read-only. Mutation tools (store/update/
    delete memory, create_agent) must also be excluded by default."""
    ti = normalize_chat("remember this")
    assert ti.tools_filter is not None
    forbidden = {
        "store_memory",
        "update_memory",
        "delete_memory",
        "create_agent",
        "call_agent",
        "create_python_tool",
        "create_bash_tool",
        "sage_recurse",
    }
    assert forbidden.isdisjoint(ti.tools_filter)


def test_normalize_chat_tools_filter_includes_search_exocortex():
    """The one tool every chat session should have — per the audit that
    motivated this spec (2026-04-21 ExoCortex SWE-bench finding)."""
    ti = normalize_chat("what papers cover MAP-Elites + RL?")
    assert ti.tools_filter is not None
    assert "search_exocortex" in ti.tools_filter


def test_normalize_chat_tools_filter_includes_lookup_library_docs():
    """Post-C2c pivot (2026-04-22): `lookup_library_docs` ships on the
    chat allowlist. Chat users asking 'how does Django / requests /
    astropy X behave' get the tool without a benchmark-proven lift —
    zero-code-cost deployment while keeping the door open for a
    variance-controlled benchmark validation later."""
    ti = normalize_chat("how does requests.Response.json handle empty body?")
    assert ti.tools_filter is not None
    assert "lookup_library_docs" in ti.tools_filter


def test_normalize_chat_bash_opt_in_env_var(monkeypatch):
    """SAGE_CHAT_ALLOW_BASH=1 → no tool filter applied (all tools
    available). Matches the spec Q1 escape hatch."""
    monkeypatch.setenv(_BASH_OPT_IN_ENV, "1")
    ti = normalize_chat("run ls")
    assert ti.tools_filter is None


@pytest.mark.parametrize("truthy", ["1", "true", "True", "yes", "on", "TRUE"])
def test_normalize_chat_bash_opt_in_accepts_common_truthy_values(monkeypatch, truthy):
    monkeypatch.setenv(_BASH_OPT_IN_ENV, truthy)
    assert normalize_chat("x").tools_filter is None


@pytest.mark.parametrize("falsy", ["", "0", "false", "no", "off", "anything-else"])
def test_normalize_chat_bash_opt_in_rejects_falsy_and_unknown_values(monkeypatch, falsy):
    """Only explicit truthy values flip the switch. An unrecognized
    string is treated as "not opted in" — fail-safe."""
    monkeypatch.setenv(_BASH_OPT_IN_ENV, falsy)
    ti = normalize_chat("x")
    assert ti.tools_filter is not None
    assert "bash" not in ti.tools_filter


def test_normalize_chat_tools_filter_is_per_call_copy():
    """Each call returns a fresh list so that mutating one TaskInput's
    `tools_filter` never bleeds into the module-level constant or into
    the next returned TaskInput."""
    a = normalize_chat("one")
    b = normalize_chat("two")
    a.tools_filter.append("bash")  # mutate in place
    assert "bash" not in b.tools_filter
    assert "bash" not in CHAT_DEFAULT_TOOLS


def test_normalize_chat_empty_string_is_valid():
    """Empty chat messages shouldn't crash the adapter; the pipeline
    downstream may reject them, but the adapter is pure normalization."""
    ti = normalize_chat("")
    assert ti.prompt == ""
    assert ti.response_format == ResponseFormat.TEXT


def test_normalize_chat_no_hints_no_instructions_no_length_hint():
    """Chat mode starts bare — no hints, no workflow block, no length
    constraint. Benches layer these on; chat does not."""
    ti = normalize_chat("something")
    assert ti.hints == {}
    assert ti.instructions == ""
    assert ti.expected_length_hint == 0


def test_package_public_api_surface():
    """Lock the __all__ exports so the package surface grows only on
    purpose. Expanded in C2a to include the SWE-bench normalizer."""
    import sage.input as pkg

    assert set(pkg.__all__) == {
        "CHAT_DEFAULT_TOOLS",
        "ResponseFormat",
        "SWEBENCH_SYSTEM_TEMPLATE",
        "TaskInput",
        "normalize_chat",
        "normalize_swebench",
        "render_swebench_prompt",
    }
