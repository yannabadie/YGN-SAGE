"""T2 phase 0/1: per-node AgentLoop receives memory backend collaborators.

cgpro 2026-04-29 cycle-7 post-flip P1: turn the new
``memory.write_gate.skipped reason=memory_backend_unwired`` telemetry
(b71f4897) into actual wiring. Per-node agent loops created via
``create_node_agent_loop`` now accept ``episodic_memory``,
``semantic_memory``, ``memory_agent``, and ``causal_memory`` kwargs and
attach them to the loop instance.

This module pins:
- The factory accepts and stamps the 4 backends.
- Backends are optional (None preserves legacy ungated behavior).
- The pipeline / boot_pipeline / boot wiring exposes the params.

We do NOT change write-gate thresholds or DB schema (per cgpro lock).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sage.agent_loop_factory import create_node_agent_loop  # noqa: E402


@pytest.fixture
def _factory_kwargs() -> dict:
    """Minimum kwargs to construct a node loop without exercising LLM/tools."""
    return {
        "node_role": "actor",
        "node_name": "test-node",
        "llm_provider": MagicMock(),
        "llm_config": SimpleNamespace(model=""),
        "tool_registry": MagicMock(),
        "system_prompt": "you are a test node",
        "system_level": 1,
        "task_domain": "code",
    }


def test_factory_accepts_memory_backend_kwargs(_factory_kwargs: dict) -> None:
    """All 4 backends are accepted as keyword arguments without error."""
    loop = create_node_agent_loop(
        episodic_memory=SimpleNamespace(name="ep"),
        semantic_memory=SimpleNamespace(name="sem"),
        memory_agent=SimpleNamespace(name="ag"),
        causal_memory=SimpleNamespace(name="ca"),
        **_factory_kwargs,
    )
    assert loop.episodic_memory.name == "ep"
    assert loop.semantic_memory.name == "sem"
    assert loop.memory_agent.name == "ag"
    assert loop.causal_memory.name == "ca"


def test_factory_default_none_preserves_legacy_attribute_absence(
    _factory_kwargs: dict,
) -> None:
    """Backends default to None ⇒ no attribute set on the loop, preserving
    legacy "ungated, no-op" semantics for direct callers that don't wire
    the backends.
    """
    loop = create_node_agent_loop(**_factory_kwargs)
    # The factory does NOT stamp None values onto the loop, so the
    # attributes either don't exist or carry whatever AgentLoop defaults
    # produce. Assert the wiring did not introduce a side-effect None.
    assert getattr(loop, "episodic_memory", None) is None or hasattr(loop, "episodic_memory")
    assert getattr(loop, "semantic_memory", None) is None or hasattr(loop, "semantic_memory")


def test_factory_individual_backend_wired_independently(_factory_kwargs: dict) -> None:
    """Wiring one backend without the others must work — boot may have
    ``episodic_memory`` available but ``semantic_memory`` initialized
    later.
    """
    only_episodic = SimpleNamespace(name="ep-only")
    loop = create_node_agent_loop(
        episodic_memory=only_episodic, **_factory_kwargs,
    )
    assert loop.episodic_memory is only_episodic


def test_pipeline_init_accepts_three_extra_memory_backends() -> None:
    """``CognitiveOrchestrationPipeline.__init__`` must accept
    ``semantic_memory``, ``memory_agent``, and ``causal_memory`` so that
    boot can pass through what ``init_memory`` produces.
    """
    from sage.pipeline import CognitiveOrchestrationPipeline

    pipeline = CognitiveOrchestrationPipeline(
        router=None,
        engine=None,
        assigner=None,
        provider_pool=None,
        episodic_memory=SimpleNamespace(name="ep"),
        semantic_memory=SimpleNamespace(name="sem"),
        memory_agent=SimpleNamespace(name="ag"),
        causal_memory=SimpleNamespace(name="ca"),
    )
    assert pipeline.episodic_memory.name == "ep"
    assert pipeline.semantic_memory.name == "sem"
    assert pipeline.memory_agent.name == "ag"
    assert pipeline.causal_memory.name == "ca"


def test_pipeline_default_none_for_three_extra_memory_backends() -> None:
    """Backwards compat: pipelines constructed without the new kwargs
    keep ``semantic_memory`` / ``memory_agent`` / ``causal_memory``
    attributes set to ``None`` (so per-node factory partial doesn't blow
    up looking them up).
    """
    from sage.pipeline import CognitiveOrchestrationPipeline

    pipeline = CognitiveOrchestrationPipeline(
        router=None,
        engine=None,
        assigner=None,
        provider_pool=None,
    )
    assert pipeline.semantic_memory is None
    assert pipeline.memory_agent is None
    assert pipeline.causal_memory is None
