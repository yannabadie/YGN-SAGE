"""F3 — forced final answer + emitter budget promotion (cgpro
DESIGN_LOCKED 2026-06-11, conv cgpro_emission_fixes_design, sequence 4).

Side-effect rule: the structured exhaustion (and legacy sentinel) is
emitted only when the forced non-tool turn STILL yields no content.

Loop tests drive think/act with fakes: a bare ToolRegistry already
yields tool_defs == [], so the forced turn is discriminated by the
_FORCED_FINAL_NUDGE message — the real injection signal — not by the
tool_defs value.
"""
from __future__ import annotations

import sys
import types

# sage_core mock (same pattern as test_agent_loop.py — must precede
# sage imports for environments without the Rust wheel's WorkingMemory).
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")
_mock_core = sys.modules["sage_core"]
if not hasattr(_mock_core, "WorkingMemory"):
    class _MockWorkingMemory:
        def __init__(self, agent_id, parent_id=None):
            self.agent_id = agent_id
            self._events: list = []

        def add_event(self, event_type, content):
            self._events.append((event_type, content))
            return f"evt-{len(self._events)}"

        def get_event(self, event_id):
            return None

        def recent_events(self, n):
            return []

        def event_count(self):
            return len(self._events)

        def add_child_agent(self, child_id):
            pass

        def child_agents(self):
            return []

        def compress_old_events(self, keep_recent, summary):
            pass

        def compact_to_arrow(self):
            return 0

        def compact_to_arrow_with_meta(self, keywords, embedding,
                                       parent_chunk_id):
            return 0

        def retrieve_relevant_chunks(self, active_chunk_id, max_hops,
                                     weights):
            return []

        def get_page_out_candidates(self, active_chunk_id, max_hops,
                                    budget):
            return []

        def smmu_chunk_count(self):
            return 0

        def get_latest_arrow_chunk(self):
            return None

    _mock_core.WorkingMemory = _MockWorkingMemory

import pytest  # noqa: E402

from sage.agent import AgentConfig  # noqa: E402
from sage.agent_loop import _FORCED_FINAL_NUDGE, AgentLoop  # noqa: E402
from sage.llm.base import LLMConfig  # noqa: E402
from sage.llm.mock import MockProvider  # noqa: E402
from sage.phases.act import _ActResult  # noqa: E402
from sage.phases.learn import EMPTY_STEP_SENTINEL  # noqa: E402
from sage.phases.think import _ThinkResult  # noqa: E402

DIFF = (
    "--- a/mod.py\n+++ b/mod.py\n@@ -1,1 +1,1 @@\n-a = 1\n+a = 2\n"
)


def _loop(max_steps: int = 3, **cfg) -> AgentLoop:
    config = AgentConfig(
        name="f3-test",
        llm=LLMConfig(provider="mock", model="mock"),
        max_steps=max_steps,
        validation_level=0,
        **cfg,
    )
    return AgentLoop(config=config, llm_provider=MockProvider(responses=["x"]))


def _nudged(messages) -> bool:
    return any(
        getattr(m, "content", "") == _FORCED_FINAL_NUDGE for m in messages
    )


def _install_fakes(monkeypatch, *, forced_reply: str):
    """think: tool-thrash until the nudge appears, then ``forced_reply``.
    act: break with content when present, else proceed (tool step).
    Records (nudged, tool_defs) per think call."""
    import sage.phases.act as act_mod
    import sage.phases.think as think_mod

    record: list[tuple[bool, list]] = []

    async def fake_think(task, messages, system_prompt, tool_defs, loop):
        nudged = _nudged(messages)
        record.append((nudged, tool_defs))
        if nudged:
            return _ThinkResult(content=forced_reply, response=None)
        fake_call = types.SimpleNamespace(
            name="grep", arguments={"q": len(record)}
        )
        response = types.SimpleNamespace(tool_calls=[fake_call])
        return _ThinkResult(content="", response=response)

    async def fake_act(task, content, response, brake, messages, loop):
        if content:
            return _ActResult(result_text=content, loop_action="break")
        return _ActResult(result_text="", loop_action="proceed",
                          has_tool_calls=True)

    monkeypatch.setattr(think_mod, "think", fake_think)
    monkeypatch.setattr(act_mod, "act", fake_act)
    return record


@pytest.mark.asyncio
async def test_penultimate_step_forces_non_tool_final_turn(monkeypatch) -> None:
    """Steps 1-2 tool-thrash; the LAST budgeted step must carry the
    emit-now nudge and run with tools disabled."""
    record = _install_fakes(monkeypatch, forced_reply="final answer: done")
    loop = _loop(max_steps=3)
    result = await loop.run("do the thing")
    assert loop._forced_final_attempted is True
    nudged_calls = [r for r in record if r[0]]
    assert len(nudged_calls) == 1
    assert nudged_calls[0][1] == []  # tools disabled on the forced turn
    assert "final answer" in result


@pytest.mark.asyncio
async def test_forced_final_content_prevents_sentinel_failure(monkeypatch) -> None:
    _install_fakes(
        monkeypatch, forced_reply="the patch:\n```diff\n" + DIFF + "```"
    )
    loop = _loop(max_steps=3)
    result = await loop.run("fix it")
    assert "exited after" not in result
    assert loop.last_exhaustion is None
    assert "+a = 2" in result


@pytest.mark.asyncio
async def test_forced_final_empty_emits_runtime_failure_reason_and_legacy_sentinel(
    monkeypatch,
) -> None:
    _install_fakes(monkeypatch, forced_reply="")
    loop = _loop(max_steps=3)
    result = await loop.run("fix it")
    assert result == EMPTY_STEP_SENTINEL.format(step_count=3)
    assert loop._forced_final_attempted is True
    assert loop.last_exhaustion is not None
    assert loop.last_exhaustion.reason == "budget_exhausted"


@pytest.mark.asyncio
async def test_stall_cap_forces_final_turn_once(monkeypatch) -> None:
    """Tool-thrash hits the stall cap mid-budget: instead of breaking
    with exhaustion, the loop redirects to ONE forced no-tool turn."""
    record = _install_fakes(monkeypatch, forced_reply="forced final content")
    loop = _loop(max_steps=10, stall_after_tool_steps=3)
    result = await loop.run("task")
    assert result == "forced final content"
    assert loop._forced_final_attempted is True
    assert sum(1 for r in record if r[0]) == 1  # exactly one forced turn
    assert loop.last_exhaustion is None


def test_patch_synthesizer_budget_promoted_s1_to_s2(monkeypatch) -> None:
    monkeypatch.setenv("SAGE_TASK_ARTIFACT_PROFILE", "unified_diff")
    from sage.agent_loop_factory import create_node_agent_loop
    from sage.tools.registry import ToolRegistry

    loop = create_node_agent_loop(
        node_role="synthesizer",
        node_name="synth-0",
        llm_provider=MockProvider(responses=["x"]),
        llm_config=LLMConfig(provider="mock", model="mock"),
        tool_registry=ToolRegistry(),
        system_prompt="emit",
        system_level=1,
    )
    assert loop.config.max_steps == 10


def test_non_patch_s1_budget_unchanged(monkeypatch) -> None:
    monkeypatch.delenv("SAGE_TASK_ARTIFACT_PROFILE", raising=False)
    from sage.agent_loop_factory import create_node_agent_loop
    from sage.tools.registry import ToolRegistry

    loop = create_node_agent_loop(
        node_role="synthesizer",
        node_name="synth-0",
        llm_provider=MockProvider(responses=["x"]),
        llm_config=LLMConfig(provider="mock", model="mock"),
        tool_registry=ToolRegistry(),
        system_prompt="emit",
        system_level=1,
    )
    assert loop.config.max_steps == 5


# ── F4: bypass task-profile specialization (DESIGN sequence 5) ──────────────


def _bypass(task_profile=None):
    from sage.agent_loop_factory import create_bypass_agent_loop

    singleton = _loop(max_steps=5)
    loop = create_bypass_agent_loop(
        singleton=singleton,
        llm_provider=MockProvider(responses=["x"]),
        llm_config=LLMConfig(provider="mock", model="mock"),
        system_level=1,
        task_profile=task_profile,
    )
    return singleton, loop


def test_bypass_patch_profile_uses_patch_focused_prompt() -> None:
    _, loop = _bypass(task_profile="unified_diff")
    assert "unified diff" in loop.config.system_prompt
    assert "byte-for-byte" in loop.config.system_prompt
    assert loop.config.max_steps == 10  # S2 floor on patch profile


def test_bypass_no_profile_keeps_generic_prompt() -> None:
    singleton, loop = _bypass(task_profile=None)
    assert loop.config.system_prompt == singleton.config.system_prompt
    assert loop.config.max_steps == 5  # S1 unchanged


def test_bypass_domain_and_task_profile_forwarded() -> None:
    """The factory accepts the explicit profile kwarg (backward-compat:
    omitting it behaves exactly as before)."""
    import inspect

    from sage.agent_loop_factory import create_bypass_agent_loop

    sig = inspect.signature(create_bypass_agent_loop)
    assert "task_profile" in sig.parameters
    assert sig.parameters["task_profile"].default is None


def test_repo_tools_remain_registered_for_patch_bypass() -> None:
    singleton, loop = _bypass(task_profile="unified_diff")
    assert loop._tools is singleton._tools  # shared registry, not stripped
