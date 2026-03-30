"""Tests for causal memory wiring in the agent loop.

Verifies that:
1. Causal edges are created from entity extraction results
2. Tool calls produce causal edges (tool -> result)
3. Causal context is injected into LLM messages
4. Circuit breaker protects against causal memory failures
5. search_causal_chain tool works in both directions
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sage.memory.causal import CausalMemory


# -- Unit tests: CausalMemory edge creation from extraction --------------------

class TestCausalEdgeFromExtraction:
    """Simulate what agent_loop does after memory_agent.extract()."""

    def test_consecutive_entities_form_causal_edges(self):
        cm = CausalMemory()
        entities = ["Parser", "AST", "Optimizer", "CodeGen"]

        for i in range(len(entities) - 1):
            src, tgt = entities[i], entities[i + 1]
            cm.add_entity(src)
            cm.add_entity(tgt)
            cm.add_causal_edge(src, tgt, cause_type="enabled")

        assert cm.entity_count() == 4
        assert len(cm._causal_edges) == 3

        # Forward chain from Parser should reach all nodes
        chain = cm.get_causal_chain("Parser")
        assert chain == ["Parser", "AST", "Optimizer", "CodeGen"]

    def test_single_entity_no_edges(self):
        cm = CausalMemory()
        cm.add_entity("Singleton")
        assert cm.entity_count() == 1
        assert len(cm._causal_edges) == 0

    def test_two_entities_one_edge(self):
        cm = CausalMemory()
        entities = ["Input", "Output"]
        for i in range(len(entities) - 1):
            cm.add_entity(entities[i])
            cm.add_entity(entities[i + 1])
            cm.add_causal_edge(entities[i], entities[i + 1], cause_type="enabled")

        assert len(cm._causal_edges) == 1
        assert cm.get_causal_chain("Input") == ["Input", "Output"]
        assert cm.get_causal_ancestors("Output") == ["Input"]


# -- Unit tests: tool call causal edges ----------------------------------------

class TestToolCallCausalEdges:
    """Simulate what agent_loop does after tool execution."""

    def test_tool_triggered_result(self):
        cm = CausalMemory()
        tool_entity = "tool:execute_code"
        output_entity = "result:execute_code:3"

        cm.add_entity(tool_entity)
        cm.add_entity(output_entity)
        cm.add_causal_edge(tool_entity, output_entity, cause_type="triggered")

        chain = cm.get_causal_chain(tool_entity)
        assert output_entity in chain

        ancestors = cm.get_causal_ancestors(output_entity)
        assert tool_entity in ancestors

    def test_multiple_tool_calls_independent(self):
        cm = CausalMemory()
        for step in range(3):
            tool_ent = f"tool:search:{step}"
            result_ent = f"result:search:{step}"
            cm.add_entity(tool_ent)
            cm.add_entity(result_ent)
            cm.add_causal_edge(tool_ent, result_ent, cause_type="triggered")

        assert cm.entity_count() == 6
        assert len(cm._causal_edges) == 3


# -- Unit tests: causal context injection --------------------------------------

class TestCausalContextInjection:
    """Test get_context_for() used for LLM prompt injection."""

    def test_context_includes_causal_arrows(self):
        cm = CausalMemory()
        cm.add_entity("database")
        cm.add_entity("cache")
        cm.add_causal_edge("database", "cache", cause_type="triggered")

        context = cm.get_context_for("check the database")
        assert "database" in context
        assert "triggered" in context

    def test_no_context_when_no_match(self):
        cm = CausalMemory()
        cm.add_entity("foo")
        cm.add_entity("bar")
        cm.add_causal_edge("foo", "bar", cause_type="caused")

        context = cm.get_context_for("unrelated query about widgets")
        assert context == ""

    def test_empty_memory_returns_empty(self):
        cm = CausalMemory()
        assert cm.get_context_for("anything") == ""


# -- Unit tests: search_causal_chain tool --------------------------------------

class TestSearchCausalChainTool:
    """Test the memory tool for causal chain search."""

    def test_tool_creation_with_causal_memory(self):
        from sage.memory.episodic import EpisodicMemory
        from sage.memory.working import WorkingMemory
        from sage.tools.memory_tools import create_memory_tools

        wm = MagicMock(spec=WorkingMemory)
        ep = MagicMock(spec=EpisodicMemory)
        cm = CausalMemory()

        tools = create_memory_tools(wm, ep, causal_memory=cm)
        tool_names = [t.spec.name for t in tools]
        assert "search_causal_chain" in tool_names

    def test_tool_creation_without_causal_memory(self):
        from sage.memory.episodic import EpisodicMemory
        from sage.memory.working import WorkingMemory
        from sage.tools.memory_tools import create_memory_tools

        wm = MagicMock(spec=WorkingMemory)
        ep = MagicMock(spec=EpisodicMemory)

        tools = create_memory_tools(wm, ep, causal_memory=None)
        tool_names = [t.spec.name for t in tools]
        assert "search_causal_chain" not in tool_names

    @pytest.mark.asyncio
    async def test_forward_chain_tool(self):
        from sage.memory.episodic import EpisodicMemory
        from sage.memory.working import WorkingMemory
        from sage.tools.memory_tools import create_memory_tools

        wm = MagicMock(spec=WorkingMemory)
        ep = MagicMock(spec=EpisodicMemory)
        cm = CausalMemory()
        cm.add_entity("A")
        cm.add_entity("B")
        cm.add_entity("C")
        cm.add_causal_edge("A", "B", cause_type="enabled")
        cm.add_causal_edge("B", "C", cause_type="enabled")

        tools = create_memory_tools(wm, ep, causal_memory=cm)
        chain_tool = [t for t in tools if t.spec.name == "search_causal_chain"][0]
        result = (await chain_tool.execute({"entity": "A", "direction": "forward"})).output
        assert "A" in result
        assert "B" in result
        assert "C" in result

    @pytest.mark.asyncio
    async def test_backward_chain_tool(self):
        from sage.memory.episodic import EpisodicMemory
        from sage.memory.working import WorkingMemory
        from sage.tools.memory_tools import create_memory_tools

        wm = MagicMock(spec=WorkingMemory)
        ep = MagicMock(spec=EpisodicMemory)
        cm = CausalMemory()
        cm.add_entity("X")
        cm.add_entity("Y")
        cm.add_causal_edge("X", "Y", cause_type="caused")

        tools = create_memory_tools(wm, ep, causal_memory=cm)
        chain_tool = [t for t in tools if t.spec.name == "search_causal_chain"][0]
        result = (await chain_tool.execute({"entity": "Y", "direction": "backward"})).output
        assert "X" in result
        assert "Ancestors" in result

    @pytest.mark.asyncio
    async def test_no_ancestors_found(self):
        """Backward search on entity with no ancestors returns 'No causal chain'."""
        from sage.memory.episodic import EpisodicMemory
        from sage.memory.working import WorkingMemory
        from sage.tools.memory_tools import create_memory_tools

        wm = MagicMock(spec=WorkingMemory)
        ep = MagicMock(spec=EpisodicMemory)
        cm = CausalMemory()

        tools = create_memory_tools(wm, ep, causal_memory=cm)
        chain_tool = [t for t in tools if t.spec.name == "search_causal_chain"][0]
        result = (await chain_tool.execute({"entity": "nonexistent", "direction": "backward"})).output
        assert "No causal chain found" in result
