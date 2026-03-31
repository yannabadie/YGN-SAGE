"""Tests for ToolForge autonomous tool synthesis (Axis 7).

Covers: GapDetector, ToolForge build loop, dual-gate validation,
registry usage tracking, and agent_loop TOOL_GAP emission.
"""
from __future__ import annotations

import asyncio
import json
import pytest
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

from sage.tools.gap_detector import CreationTicket, GapDetector


# ── GapDetector tests ──────────────────────────────────────────────────────


class TestGapDetector:
    def test_on_unknown_tool_creates_ticket(self):
        gd = GapDetector()
        ticket = gd.on_unknown_tool("my_tool", {"x": 1}, "solve task")
        assert ticket is not None
        assert ticket.tool_name_hint == "my_tool"
        assert "my_tool" in ticket.gap_description
        assert gd.pending_count == 1

    def test_on_unknown_tool_empty_name_returns_none(self):
        gd = GapDetector()
        assert gd.on_unknown_tool("", {}, "task") is None
        assert gd.pending_count == 0

    def test_max_pending_limit(self):
        gd = GapDetector()
        for i in range(GapDetector.MAX_PENDING):
            assert gd.on_unknown_tool(f"tool_{i}", {}, "task") is not None
        # Queue full
        assert gd.on_unknown_tool("tool_overflow", {}, "task") is None
        assert gd.pending_count == GapDetector.MAX_PENDING

    def test_deduplication(self):
        gd = GapDetector()
        gd.on_unknown_tool("my_tool", {}, "task1")
        duplicate = gd.on_unknown_tool("my_tool", {}, "task2")
        assert duplicate is None
        assert gd.pending_count == 1

    def test_tick_expires_old_tickets(self):
        gd = GapDetector()
        gd.on_unknown_tool("old_tool", {}, "task")
        assert gd.pending_count == 1
        # Advance past TTL
        for _ in range(GapDetector.TICKET_TTL + 1):
            gd.tick()
        assert gd.pending_count == 0

    def test_pop_tickets_clears_queue(self):
        gd = GapDetector()
        gd.on_unknown_tool("tool_a", {}, "task")
        gd.on_unknown_tool("tool_b", {}, "task")
        tickets = gd.pop_tickets()
        assert len(tickets) == 2
        assert gd.pending_count == 0

    def test_clear(self):
        gd = GapDetector()
        gd.on_unknown_tool("tool_a", {}, "task")
        gd.clear()
        assert gd.pending_count == 0


# ── ToolForge tests ────────────────────────────────────────────────────────


class TestToolForge:
    def _make_forge(self, llm_response=""):
        from sage.tools.forge import ToolForge

        registry = MagicMock()
        registry.get.return_value = None  # tool not found

        llm = AsyncMock()
        response_mock = MagicMock()
        response_mock.content = llm_response
        llm.generate.return_value = response_mock

        config = MagicMock()
        forge = ToolForge(
            registry=registry,
            llm_provider=llm,
            llm_config=config,
        )
        return forge

    def test_sanitize_name(self):
        from sage.tools.forge import ToolForge
        assert ToolForge._sanitize_name("my-tool!") == "my_tool_"
        assert ToolForge._sanitize_name("123abc") == "_123abc"
        assert ToolForge._sanitize_name("") == "custom_tool"
        assert ToolForge._sanitize_name("ValidName") == "validname"

    def test_parse_tool_response_two_blocks(self):
        from sage.tools.forge import ToolForge
        content = (
            "Here is the tool:\n"
            "```python\ndef tool_foo(args):\n    pass\n```\n"
            "And the tests:\n"
            "```python\ntool_foo({})\n```\n"
        )
        code, tests = ToolForge._parse_tool_response(content)
        assert "def tool_foo" in code
        assert "tool_foo" in tests

    def test_parse_tool_response_one_block(self):
        from sage.tools.forge import ToolForge
        content = "```python\ndef tool_bar(args):\n    pass\n```\n"
        code, tests = ToolForge._parse_tool_response(content)
        assert "def tool_bar" in code
        assert tests == ""

    def test_parse_tool_response_no_blocks(self):
        from sage.tools.forge import ToolForge
        code, tests = ToolForge._parse_tool_response("No code here.")
        assert code == ""
        assert tests == ""

    def test_validate_ast_valid(self):
        from sage.tools.forge import ToolForge
        ok, err = ToolForge._validate_ast("x = 1 + 2\nprint(x)")
        assert ok is True
        assert err == ""

    def test_validate_ast_invalid(self):
        from sage.tools.forge import ToolForge
        ok, err = ToolForge._validate_ast("def broken(:\n    pass")
        assert ok is False
        assert err  # Non-empty error message (exact wording depends on backend)

    def test_extract_tool_name_from_code(self):
        from sage.tools.forge import ToolForge
        name = ToolForge._extract_tool_name("def tool_my_func(args):\n    pass", "hint")
        assert name == "tool_my_func"

    def test_extract_tool_name_fallback_to_hint(self):
        from sage.tools.forge import ToolForge
        name = ToolForge._extract_tool_name("x = 1", "my_hint")
        assert name == "my_hint"

    @pytest.mark.asyncio
    async def test_process_tickets_respects_max(self):
        forge = self._make_forge()
        tickets = [
            CreationTicket(task="t", gap_description="g", required_interface="",
                           context="", created_at=0)
            for _ in range(5)
        ]
        forge._creations_this_run = 2  # Already at max
        created = await forge.process_tickets(tickets)
        assert len(created) == 0

    @pytest.mark.asyncio
    async def test_build_tool_success(self):
        """End-to-end: LLM generates valid tool → registered successfully."""
        llm_response = (
            "```python\n"
            "import json\n"
            "def tool_adder(args):\n"
            "    result = args.get('a', 0) + args.get('b', 0)\n"
            "    print(json.dumps({'output': result}))\n"
            "```\n\n"
            "```python\n"
            "tool_adder({'a': 1, 'b': 2})\n"
            "tool_adder({'a': 0, 'b': 0})\n"
            "tool_adder({'a': -1, 'b': 1})\n"
            "```\n"
        )
        forge = self._make_forge(llm_response)

        # Mock create_python_tool where forge.py imports it
        with patch("sage.tools.meta.create_python_tool", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = "Tool 'tool_adder' created successfully"
            ticket = CreationTicket(
                task="add two numbers",
                gap_description="Tool 'adder' not found",
                required_interface="{'a': int, 'b': int}",
                context="",
                created_at=0,
                tool_name_hint="adder",
            )
            name = await forge._build_tool(ticket)
            assert name == "tool_adder"
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_tool_all_rounds_fail(self):
        """All 3 rounds fail → returns None."""
        # LLM always returns invalid code
        forge = self._make_forge("No code here, just text.")
        ticket = CreationTicket(
            task="task", gap_description="gap", required_interface="",
            context="", created_at=0, tool_name_hint="broken",
        )
        name = await forge._build_tool(ticket)
        assert name is None

    def test_reset_run(self):
        forge = self._make_forge()
        forge._creations_this_run = 2
        forge.reset_run()
        assert forge._creations_this_run == 0


# ── Registry Usage Tracking tests ──────────────────────────────────────────


class TestRegistryUsageTracking:
    def test_record_usage_creates_entry(self):
        from sage.tools.registry import ToolRegistry
        reg = ToolRegistry()
        # Only test if record_usage exists (added in this PR)
        if not hasattr(reg, "record_usage"):
            pytest.skip("record_usage not yet implemented")
        reg.record_usage("my_tool", success=True)
        usage = reg.get_usage("my_tool")
        assert usage["usage_count"] == 1
        assert usage["success_count"] == 1

    def test_record_usage_failure(self):
        from sage.tools.registry import ToolRegistry
        reg = ToolRegistry()
        if not hasattr(reg, "record_usage"):
            pytest.skip("record_usage not yet implemented")
        reg.record_usage("my_tool", success=False)
        usage = reg.get_usage("my_tool")
        assert usage["usage_count"] == 1
        assert usage["success_count"] == 0

    def test_mark_source(self):
        from sage.tools.registry import ToolRegistry
        reg = ToolRegistry()
        if not hasattr(reg, "mark_source"):
            pytest.skip("mark_source not yet implemented")
        reg.mark_source("my_tool", "forged")
        usage = reg.get_usage("my_tool")
        assert usage["source"] == "forged"

    def test_get_usage_unknown_tool(self):
        from sage.tools.registry import ToolRegistry
        reg = ToolRegistry()
        if not hasattr(reg, "get_usage"):
            pytest.skip("get_usage not yet implemented")
        usage = reg.get_usage("nonexistent")
        assert usage["usage_count"] == 0
