"""Unit tests for C4b — perceive() consuming TaskInput directly.

Spec: docs/superpowers/specs/2026-04-21-universal-input-adapter-design.md §C4 / C4b.

Pre-C4b `perceive()` took a raw `task: str`. Post-C4b it accepts
`str | TaskInput`. String callers keep byte-identical behavior (the
input is wrapped in a zero-field TaskInput). TaskInput callers unlock:
  * `instructions` → "## Workflow" section injected into the system
    prompt after the tool-affordance block.
  * `tools_filter` → overrides `loop.config.tools` for both the
    affordance block and the tool_defs passed to the LLM.

These tests use a minimal fake `AgentLoop` built by hand — full boot
is unnecessary and would pull network / provider init. We verify the
system-prompt composition + tool-filter precedence + memory-of-task-
str behavior directly.
"""
from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from sage.input.types import ResponseFormat, TaskInput
from sage.phases.perceive import _coerce_to_task_input, perceive


# ---------------------------------------------------------------------------
# Minimal AgentLoop fake — enough surface for perceive() to run without
# touching a real pipeline. Anything perceive reads gets stubbed; anything
# it writes (working_memory events, emit, record_success) gets captured
# for assertion.
# ---------------------------------------------------------------------------


@dataclass
class _StubConfig:
    name: str = "test-agent"
    system_prompt: str = "base system prompt"
    tools: list[str] | None = None
    validation_level: int = 1


class _StubWorkingMemory:
    def __init__(self):
        self.events: list[tuple[str, str]] = []

    def add_event(self, kind: str, payload: str) -> None:
        self.events.append((kind, payload))


class _StubTools:
    """Records the `names` filter every call receives so tests can
    assert precedence (TaskInput.tools_filter > loop.config.tools)."""

    def __init__(self, affordance: str = "## Available Tools\n- bash\n"):
        self.affordance = affordance
        self.last_describe_names: list[str] | None = "UNSET"  # sentinel
        self.last_get_names: list[str] | None = "UNSET"

    def describe_for_prompt(self, names):
        self.last_describe_names = names
        return self.affordance

    def get_tool_defs(self, names):
        self.last_get_names = names
        return []


class _StubCircuit:
    def __init__(self):
        self.successes = 0
        self.failures = 0

    def should_skip(self) -> bool:
        return False

    def record_success(self) -> None:
        self.successes += 1

    def record_failure(self, _exc) -> None:
        self.failures += 1


class _StubRelevanceGate:
    def is_relevant(self, task, ctx) -> bool:
        return False  # skip semantic + smmu injection in these unit tests


def _make_loop(
    *,
    config: _StubConfig | None = None,
    affordance: str = "## Available Tools\n- bash\n",
) -> MagicMock:
    loop = MagicMock()
    loop.config = config or _StubConfig()
    loop.metacognition = None
    loop._skip_routing = True  # forces the "ablation_forced_s2" branch
    loop.guardrail_pipeline = None
    loop._skip_guardrails = True
    loop.semantic_memory = None
    loop._skip_memory = True
    loop._cb_semantic = _StubCircuit()
    loop._cb_smmu = _StubCircuit()
    loop._relevance_gate = _StubRelevanceGate()
    loop._emit = MagicMock()
    loop.working_memory = _StubWorkingMemory()
    loop._tools = _StubTools(affordance=affordance)
    return loop


# ---------------------------------------------------------------------------
# _coerce_to_task_input — unit tests
# ---------------------------------------------------------------------------


def test_coerce_wraps_raw_string_into_taskinput():
    ti = _coerce_to_task_input("hello world")
    assert isinstance(ti, TaskInput)
    assert ti.prompt == "hello world"
    assert ti.source == "direct"
    # Zero-value fields on the wrapper keep pre-C4b behavior:
    assert ti.instructions == ""
    assert ti.tools_filter is None
    assert ti.response_format == ResponseFormat.TEXT


def test_coerce_passes_through_existing_taskinput_identity():
    original = TaskInput(
        prompt="fix bug", instructions="step 1", source="swebench"
    )
    ti = _coerce_to_task_input(original)
    assert ti is original  # passthrough, no copy


# ---------------------------------------------------------------------------
# perceive() — string-path byte identity (pre-C4b parity)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_perceive_with_raw_string_matches_preC4b_shape():
    """String callers hit the same code paths they did pre-C4b:
    base system_prompt + tool-affordance, no Workflow section,
    one USER event in working_memory."""
    loop = _make_loop()
    result = await perceive("solve this", loop)

    assert result.blocked_reason is None
    assert result.system_prompt.startswith("base system prompt\n\n## Available Tools")
    # No Workflow section for string callers (instructions=="")
    assert "## Workflow" not in result.system_prompt
    # Messages
    assert len(result.messages) == 2
    assert result.messages[0].content == result.system_prompt
    assert result.messages[1].content == "solve this"
    # working_memory got the USER event
    assert ("USER", "solve this") in loop.working_memory.events


@pytest.mark.asyncio
async def test_perceive_with_raw_string_uses_config_tools_filter():
    """When no TaskInput is given, `loop.config.tools` drives the
    affordance + tool_defs filter (per-node topology filtering
    from agent_loop_factory stays intact)."""
    loop = _make_loop(config=_StubConfig(tools=["execute_bash", "stm_read"]))
    await perceive("go", loop)

    assert loop._tools.last_describe_names == ["execute_bash", "stm_read"]
    assert loop._tools.last_get_names == ["execute_bash", "stm_read"]


# ---------------------------------------------------------------------------
# perceive() — TaskInput path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_perceive_uses_taskinput_prompt_as_user_message():
    """TaskInput.prompt is the NL the model sees in the USER message
    (not the source tag, not the instructions, not the hints)."""
    ti = TaskInput(
        prompt="what is a decorator?",
        instructions="Long workflow block that is NOT the user question.",
        source="chat",
    )
    loop = _make_loop()
    result = await perceive(ti, loop)

    assert result.messages[1].content == "what is a decorator?"
    assert ("USER", "what is a decorator?") in loop.working_memory.events


@pytest.mark.asyncio
async def test_perceive_injects_instructions_as_workflow_section():
    """`instructions` (e.g. SWE-bench's Mandatory Workflow block)
    lands under a '## Workflow' heading after the tool-affordance
    block. Layered composition — not substitution."""
    ti = TaskInput(
        prompt="fix the bug",
        instructions="1. locate\n2. read\n3. test",
        source="swebench",
    )
    loop = _make_loop()
    result = await perceive(ti, loop)

    sp = result.system_prompt
    assert "base system prompt" in sp
    assert "## Available Tools" in sp
    assert "## Workflow" in sp
    assert "1. locate" in sp
    # Ordering: base → tools → workflow → (validation, empty here)
    base_idx = sp.index("base system prompt")
    tools_idx = sp.index("## Available Tools")
    workflow_idx = sp.index("## Workflow")
    assert base_idx < tools_idx < workflow_idx


@pytest.mark.asyncio
async def test_perceive_empty_instructions_no_workflow_section():
    """Chat TaskInputs (empty instructions) stay pre-C4b byte-
    identical to the string-callers path. No empty Workflow heading
    leaks in."""
    ti = TaskInput(prompt="hi", source="chat")
    loop = _make_loop()
    result = await perceive(ti, loop)
    assert "## Workflow" not in result.system_prompt


@pytest.mark.asyncio
async def test_perceive_taskinput_tools_filter_overrides_config_tools():
    """`task_input.tools_filter` beats `loop.config.tools` when both
    are present. This lets chat-mode's read-only allowlist take
    effect without mutating the agent's static config."""
    ti = TaskInput(
        prompt="search this",
        tools_filter=["search_exocortex"],
        source="chat",
    )
    # Agent has a broader static config — TaskInput must win.
    loop = _make_loop(config=_StubConfig(tools=["execute_bash", "stm_read"]))
    await perceive(ti, loop)

    assert loop._tools.last_describe_names == ["search_exocortex"]
    assert loop._tools.last_get_names == ["search_exocortex"]


@pytest.mark.asyncio
async def test_perceive_taskinput_with_none_filter_falls_back_to_config():
    """`tools_filter=None` on the TaskInput means 'no opinion' —
    fall back to `loop.config.tools`. This is what str callers get
    via the coercion wrapper, so string-path byte identity holds."""
    ti = TaskInput(prompt="x", tools_filter=None, source="chat")
    loop = _make_loop(config=_StubConfig(tools=["execute_bash"]))
    await perceive(ti, loop)

    assert loop._tools.last_describe_names == ["execute_bash"]


@pytest.mark.asyncio
async def test_perceive_taskinput_empty_filter_restricts_to_zero_tools():
    """`tools_filter=[]` is distinct from None: an explicit 'no
    tools available'. Benches that need to force the LLM off tools
    can pass `[]`."""
    ti = TaskInput(prompt="x", tools_filter=[], source="chat")
    loop = _make_loop(config=_StubConfig(tools=["execute_bash"]))
    await perceive(ti, loop)

    # Empty list means no tools — still the list, not None.
    assert loop._tools.last_describe_names == []
    assert loop._tools.last_get_names == []


@pytest.mark.asyncio
async def test_perceive_perceive_meta_records_task_str_not_full_taskinput():
    """The PERCEIVE event's `task` field gets the NL string (for
    telemetry / logs), not the full TaskInput object. Dumping a
    dataclass into a trace event would bloat the stream."""
    ti = TaskInput(prompt="what is x", instructions="very long", source="chat")
    loop = _make_loop()
    await perceive(ti, loop)

    # Inspect the emit call's kwargs
    assert loop._emit.called
    _, kwargs = loop._emit.call_args
    assert kwargs["task"] == "what is x"
    # Instructions should NOT leak into the trace payload
    assert "very long" not in str(kwargs.get("task", ""))
