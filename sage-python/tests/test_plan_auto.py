"""Tests for TaskPlanner.plan_auto() — LLM-driven decomposition."""
from __future__ import annotations
import asyncio
import pytest
from unittest.mock import AsyncMock
from sage.llm.base import LLMResponse
from sage.contracts.planner import TaskPlanner


def test_plan_auto_decomposes_task():
    planner = TaskPlanner()
    mock_provider = AsyncMock()
    mock_provider.generate.return_value = LLMResponse(
        content='[{"id": "a", "description": "step a"}, {"id": "b", "description": "step b", "depends_on": ["a"]}]',
        model="test",
    )
    result = asyncio.run(planner.plan_auto("build a web app", mock_provider))
    assert result.node_count == 2
    assert result.edge_count == 1


def test_plan_auto_caps_at_6_steps():
    planner = TaskPlanner()
    mock_provider = AsyncMock()
    import json
    steps = [{"id": str(i), "description": f"step {i}"} for i in range(10)]
    mock_provider.generate.return_value = LLMResponse(content=json.dumps(steps), model="test")
    result = asyncio.run(planner.plan_auto("complex task", mock_provider))
    assert result.node_count <= 6


def test_plan_auto_fallback_on_parse_error():
    planner = TaskPlanner()
    mock_provider = AsyncMock()
    mock_provider.generate.return_value = LLMResponse(content="not json at all", model="test")
    result = asyncio.run(planner.plan_auto("some task", mock_provider))
    assert result.node_count == 1  # fallback single-node DAG


def test_plan_auto_fallback_on_provider_error():
    planner = TaskPlanner()
    mock_provider = AsyncMock()
    mock_provider.generate.side_effect = Exception("API error")
    result = asyncio.run(planner.plan_auto("some task", mock_provider))
    assert result.node_count == 1  # fallback


def test_plan_auto_handles_markdown_fenced_json():
    planner = TaskPlanner()
    mock_provider = AsyncMock()
    mock_provider.generate.return_value = LLMResponse(
        content='```json\n[{"id": "x", "description": "do x"}]\n```',
        model="test",
    )
    result = asyncio.run(planner.plan_auto("task", mock_provider))
    assert result.node_count == 1
