"""Tests for ToolRegistry.describe_for_prompt() + perceive auto-injection.

Motivated by docs/audits/2026-04-21-exocortex-swebench-usage.md: every
tool registered at boot should be discoverable from the agent's
prompt, not just from the structured tool-call schema. Each bench no
longer needs to hand-roll its tool list in its task template.
"""
from __future__ import annotations

import pytest

from sage.llm.base import ToolDef
from sage.tools.base import Tool
from sage.tools.registry import ToolRegistry


def _make_tool(name: str, description: str) -> Tool:
    async def _noop(**_kwargs):
        return "ok"
    return Tool(spec=ToolDef(name=name, description=description, parameters={}), handler=_noop)


# -- describe_for_prompt ---------------------------------------------------


def test_describe_empty_registry_returns_empty_string():
    reg = ToolRegistry()
    assert reg.describe_for_prompt() == ""


def test_describe_single_tool_renders_header_and_bullet():
    reg = ToolRegistry()
    reg.register(_make_tool("execute_bash", "Run any shell command (cat, grep, ...)"))
    out = reg.describe_for_prompt()
    assert out.startswith("## Available Tools")
    assert "- **execute_bash** — Run any shell command" in out
    assert out.endswith("\n")


def test_describe_sorts_tools_alphabetically():
    reg = ToolRegistry()
    reg.register(_make_tool("search_exocortex", "Query research KB."))
    reg.register(_make_tool("execute_bash", "Run shell."))
    reg.register(_make_tool("another_tool", "..."))
    out = reg.describe_for_prompt()
    # `another_tool` appears before `execute_bash` before `search_exocortex`
    assert out.index("another_tool") < out.index("execute_bash") < out.index("search_exocortex")


def test_describe_truncates_long_descriptions():
    reg = ToolRegistry()
    long = "x" * 500
    reg.register(_make_tool("big", long))
    out = reg.describe_for_prompt(max_desc_chars=100)
    # The truncated body ends with "..."
    assert "..." in out
    # No single tool line exceeds a reasonable budget.
    tool_line = next(line for line in out.split("\n") if line.startswith("- **big**"))
    assert len(tool_line) < 150


def test_describe_filters_by_names():
    reg = ToolRegistry()
    reg.register(_make_tool("a", "tool a"))
    reg.register(_make_tool("b", "tool b"))
    reg.register(_make_tool("c", "tool c"))
    out = reg.describe_for_prompt(names=["a", "c"])
    assert "tool a" in out
    assert "tool b" not in out
    assert "tool c" in out


def test_describe_collapses_newlines_in_description():
    """Multi-line descriptions render as a single-line bullet for Markdown
    consistency (agents don't need the structure, tokens matter)."""
    reg = ToolRegistry()
    reg.register(_make_tool("x", "first line\n\nsecond line"))
    out = reg.describe_for_prompt()
    assert "first line" in out
    assert "second line" in out
    # The bullet should be a single line — no literal \n inside the description block.
    bullet_line = next(line for line in out.split("\n") if line.startswith("- **x**"))
    assert "\n" not in bullet_line
    assert "first line  second line" in bullet_line or "first line second line" in bullet_line


# -- perceive auto-injection integration -----------------------------------


@pytest.mark.asyncio
async def test_perceive_injects_tool_block_into_system_prompt(monkeypatch):
    """Integration: perceive() must append the tool block to system_prompt
    so the LLM sees what tools are available without each bench having to
    list them manually."""
    from sage.agent_loop import AgentLoop, AgentConfig
    from sage.providers.pydantic_ai_provider import PydanticAIProvider  # noqa: F401 import side-effect
    from sage.phases.perceive import perceive

    # Minimal AgentLoop: bypass real LLM / working memory tool-hooks by
    # constructing a bare instance with just what perceive touches.
    class _StubLLM:
        async def generate(self, **_kwargs):
            raise AssertionError("perceive should not call the LLM directly")

    reg = ToolRegistry()
    reg.register(_make_tool("execute_bash", "Run shell."))
    reg.register(_make_tool("search_exocortex", "Query research KB for API contracts."))

    from sage.llm.base import LLMConfig as _LLMConfig
    cfg = AgentConfig(
        name="test-agent",
        llm=_LLMConfig(provider="test", model="dummy"),
        system_prompt="You are a test agent.",
        tools=[],  # None means "all registered" in perceive
    )
    loop = AgentLoop(config=cfg, llm_provider=_StubLLM(), tool_registry=reg)

    # Run the perceive step.
    result = await perceive("do a thing", loop)

    # The assembled system_prompt must contain both the original text
    # AND the auto-injected ``## Available Tools`` block with both tools.
    assert "You are a test agent." in result.system_prompt
    assert "## Available Tools" in result.system_prompt
    assert "execute_bash" in result.system_prompt
    assert "search_exocortex" in result.system_prompt


@pytest.mark.asyncio
async def test_perceive_noop_when_no_tools_registered():
    """Edge: empty tool registry → system_prompt is unchanged (no dangling
    ``## Available Tools`` header with zero bullets)."""
    from sage.agent_loop import AgentLoop, AgentConfig
    from sage.phases.perceive import perceive

    class _StubLLM:
        async def generate(self, **_kwargs):
            raise AssertionError("perceive should not call the LLM directly")

    from sage.llm.base import LLMConfig as _LLMConfig
    cfg = AgentConfig(
        name="test",
        llm=_LLMConfig(provider="test", model="dummy"),
        system_prompt="bare system prompt",
    )
    loop = AgentLoop(config=cfg, llm_provider=_StubLLM(), tool_registry=ToolRegistry())

    result = await perceive("task", loop)
    assert "## Available Tools" not in result.system_prompt
    assert result.system_prompt.startswith("bare system prompt")
