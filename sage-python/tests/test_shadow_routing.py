"""Tests for shadow routing — dual Rust/Python routing with divergence tracking."""
from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sage.routing.shadow import ShadowRouter


# ---------------------------------------------------------------------------
# Mock decision types (mimic Rust and Python routing outputs)
# ---------------------------------------------------------------------------

@dataclass
class MockRustDecision:
    """Mimics sage_core routing decision."""
    system: int
    model_id: str
    confidence: float
    estimated_cost: float


@dataclass
class MockPythonDecision:
    """Mimics sage.strategy.metacognition.RoutingDecision."""
    system: int
    llm_tier: str
    max_tokens: int = 4096
    use_z3: bool = False
    validation_level: int = 1


@dataclass
class MockCognitiveProfile:
    """Mimics sage.strategy.metacognition.CognitiveProfile."""
    complexity: float
    uncertainty: float
    tool_required: bool
    reasoning: str = "mock"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rust_router(system: int = 2, model_id: str = "gemini-3-flash") -> MagicMock:
    """Create a mock Rust SystemRouter."""
    router = MagicMock()
    router.route.return_value = MockRustDecision(
        system=system, model_id=model_id,
        confidence=0.85, estimated_cost=0.001,
    )
    return router


def _make_python_metacognition(
    system: int = 2, llm_tier: str = "mutator",
    complexity: float = 0.5, uncertainty: float = 0.3,
) -> MagicMock:
    """Create a mock Python AdaptiveRouter / ComplexityRouter."""
    meta = MagicMock()
    profile = MockCognitiveProfile(
        complexity=complexity, uncertainty=uncertainty, tool_required=False,
    )
    meta.assess_complexity_async = AsyncMock(return_value=profile)
    meta.route.return_value = MockPythonDecision(system=system, llm_tier=llm_tier)
    return meta


# ---------------------------------------------------------------------------
# Tests: both routers present (shadow mode active)
# ---------------------------------------------------------------------------

class TestShadowBothRouters:
    """Shadow mode with both Rust and Python routers."""

    @pytest.fixture
    def shadow(self, tmp_path: Path) -> ShadowRouter:
        rust = _make_rust_router(system=2, model_id="gemini-3-flash")
        python = _make_python_metacognition(system=2, llm_tier="mutator")
        trace_path = tmp_path / "shadow_traces.jsonl"
        return ShadowRouter(
            rust_router=rust,
            python_metacognition=python,
            trace_path=trace_path,
        )

    @pytest.mark.asyncio
    async def test_returns_rust_decision(self, shadow: ShadowRouter) -> None:
        """Primary decision must come from Rust router."""
        decision = await shadow.route("Write a sorting algorithm", budget=10.0)
        assert int(decision.system) == 2
        assert decision.model_id == "gemini-3-flash"

    @pytest.mark.asyncio
    async def test_both_routers_called(self, shadow: ShadowRouter) -> None:
        """Both routers must be invoked on every call."""
        await shadow.route("Simple task", budget=10.0)
        shadow._rust_router.route.assert_called_once_with("Simple task", 10.0)
        shadow._python_metacognition.assess_complexity_async.assert_awaited_once_with("Simple task")
        shadow._python_metacognition.route.assert_called_once()

    @pytest.mark.asyncio
    async def test_match_logged(self, shadow: ShadowRouter, tmp_path: Path) -> None:
        """When both routers agree, match=true in trace."""
        await shadow.route("Hello world", budget=5.0)
        trace_path = tmp_path / "shadow_traces.jsonl"
        lines = trace_path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["match"] is True
        assert record["rust_system"] == 2
        assert record["python_system"] == 2

    @pytest.mark.asyncio
    async def test_divergence_logged(self, tmp_path: Path) -> None:
        """When routers disagree, match=false and divergence counted."""
        rust = _make_rust_router(system=2, model_id="gemini-3-flash")
        python = _make_python_metacognition(system=1, llm_tier="fast")
        trace_path = tmp_path / "shadow_traces.jsonl"
        shadow = ShadowRouter(
            rust_router=rust, python_metacognition=python,
            trace_path=trace_path,
        )
        await shadow.route("What is 2+2?", budget=10.0)

        lines = trace_path.read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert record["match"] is False
        assert record["rust_system"] == 2
        assert record["python_system"] == 1
        assert shadow.stats["system_mismatches"] == 1

    @pytest.mark.asyncio
    async def test_trace_jsonl_format(self, shadow: ShadowRouter, tmp_path: Path) -> None:
        """Each trace line must contain expected fields."""
        await shadow.route("Explain quantum entanglement", budget=10.0)
        trace_path = tmp_path / "shadow_traces.jsonl"
        record = json.loads(trace_path.read_text().strip())
        required_keys = {
            "timestamp", "task_preview", "rust_system", "python_system",
            "rust_model", "python_tier", "match",
        }
        assert required_keys.issubset(record.keys())
        assert record["task_preview"] == "Explain quantum entanglement"

    @pytest.mark.asyncio
    async def test_task_preview_truncated(self, shadow: ShadowRouter, tmp_path: Path) -> None:
        """Task preview must be truncated to 60 characters."""
        long_task = "A" * 200
        await shadow.route(long_task, budget=10.0)
        trace_path = tmp_path / "shadow_traces.jsonl"
        record = json.loads(trace_path.read_text().strip())
        assert len(record["task_preview"]) == 60


# ---------------------------------------------------------------------------
# Tests: only Python router (zero-overhead passthrough)
# ---------------------------------------------------------------------------

class TestShadowPythonOnly:
    """When Rust router is absent, Python path runs directly."""

    @pytest.mark.asyncio
    async def test_passthrough_returns_python_decision(self, tmp_path: Path) -> None:
        """With no Rust router, returns Python decision directly."""
        python = _make_python_metacognition(system=1, llm_tier="fast")
        shadow = ShadowRouter(
            rust_router=None,
            python_metacognition=python,
            trace_path=tmp_path / "traces.jsonl",
        )
        decision = await shadow.route("Hello", budget=10.0)
        assert decision.system == 1
        assert decision.llm_tier == "fast"

    @pytest.mark.asyncio
    async def test_no_trace_written(self, tmp_path: Path) -> None:
        """No trace file written when only one router exists."""
        python = _make_python_metacognition(system=1, llm_tier="fast")
        trace_path = tmp_path / "traces.jsonl"
        shadow = ShadowRouter(
            rust_router=None, python_metacognition=python,
            trace_path=trace_path,
        )
        await shadow.route("Hello", budget=10.0)
        assert not trace_path.exists()

    @pytest.mark.asyncio
    async def test_no_comparisons_tracked(self, tmp_path: Path) -> None:
        """Comparison stats not incremented in passthrough mode."""
        python = _make_python_metacognition(system=1, llm_tier="fast")
        shadow = ShadowRouter(
            rust_router=None, python_metacognition=python,
            trace_path=tmp_path / "traces.jsonl",
        )
        await shadow.route("Hello", budget=10.0)
        assert shadow.stats["total_comparisons"] == 0


# ---------------------------------------------------------------------------
# Tests: only Rust router (no Python comparison)
# ---------------------------------------------------------------------------

class TestShadowRustOnly:
    """When Python metacognition is absent, Rust runs alone."""

    @pytest.mark.asyncio
    async def test_rust_only_returns_decision(self, tmp_path: Path) -> None:
        """With no Python router, returns Rust decision directly."""
        rust = _make_rust_router(system=3, model_id="gpt-5.3-codex")
        shadow = ShadowRouter(
            rust_router=rust, python_metacognition=None,
            trace_path=tmp_path / "traces.jsonl",
        )
        decision = await shadow.route("Prove P != NP", budget=10.0)
        assert int(decision.system) == 3
        assert decision.model_id == "gpt-5.3-codex"

    @pytest.mark.asyncio
    async def test_no_trace_written(self, tmp_path: Path) -> None:
        """No trace file written when only one router exists."""
        rust = _make_rust_router(system=2, model_id="gemini-3-flash")
        trace_path = tmp_path / "traces.jsonl"
        shadow = ShadowRouter(
            rust_router=rust, python_metacognition=None,
            trace_path=trace_path,
        )
        await shadow.route("Hello", budget=10.0)
        assert not trace_path.exists()


# ---------------------------------------------------------------------------
# Tests: divergence tracking and Phase 5 gate
# ---------------------------------------------------------------------------

class TestDivergenceTracking:
    """Divergence rate calculation and Phase 5 readiness gate."""

    @pytest.mark.asyncio
    async def test_divergence_rate_zero_on_match(self, tmp_path: Path) -> None:
        """Divergence rate is 0.0 when all routing agrees."""
        rust = _make_rust_router(system=2, model_id="gemini-3-flash")
        python = _make_python_metacognition(system=2, llm_tier="mutator")
        shadow = ShadowRouter(
            rust_router=rust, python_metacognition=python,
            trace_path=tmp_path / "traces.jsonl",
        )
        for _ in range(10):
            await shadow.route("Test task", budget=10.0)
        assert shadow.divergence_rate() == 0.0

    @pytest.mark.asyncio
    async def test_divergence_rate_with_mismatches(self, tmp_path: Path) -> None:
        """Divergence rate reflects actual mismatch proportion."""
        rust = _make_rust_router(system=2, model_id="gemini-3-flash")
        python = _make_python_metacognition(system=2, llm_tier="mutator")
        shadow = ShadowRouter(
            rust_router=rust, python_metacognition=python,
            trace_path=tmp_path / "traces.jsonl",
        )
        # 8 matches
        for _ in range(8):
            await shadow.route("Matching task", budget=10.0)

        # 2 mismatches
        python.route.return_value = MockPythonDecision(system=1, llm_tier="fast")
        for _ in range(2):
            await shadow.route("Divergent task", budget=10.0)

        assert shadow.divergence_rate() == pytest.approx(0.2)

    def test_divergence_rate_no_comparisons(self, tmp_path: Path) -> None:
        """Divergence rate is 0.0 when no comparisons have been made."""
        shadow = ShadowRouter(
            rust_router=None, python_metacognition=None,
            trace_path=tmp_path / "traces.jsonl",
        )
        assert shadow.divergence_rate() == 0.0

    @pytest.mark.asyncio
    async def test_phase5_not_ready_few_traces(self, tmp_path: Path) -> None:
        """Phase 5 requires >= 1000 comparisons."""
        rust = _make_rust_router(system=2, model_id="gemini-3-flash")
        python = _make_python_metacognition(system=2, llm_tier="mutator")
        shadow = ShadowRouter(
            rust_router=rust, python_metacognition=python,
            trace_path=tmp_path / "traces.jsonl",
        )
        for _ in range(100):
            await shadow.route("Test", budget=10.0)
        assert not shadow.is_phase5_ready()

    @pytest.mark.asyncio
    async def test_phase5_not_ready_high_divergence(self, tmp_path: Path) -> None:
        """Phase 5 requires < 5% divergence rate."""
        rust = _make_rust_router(system=2, model_id="gemini-3-flash")
        python = _make_python_metacognition(system=1, llm_tier="fast")
        shadow = ShadowRouter(
            rust_router=rust, python_metacognition=python,
            trace_path=tmp_path / "traces.jsonl",
        )
        # All 1000 are mismatches -- 100% divergence
        for _ in range(1000):
            await shadow.route("Test", budget=10.0)
        assert shadow.stats["total_comparisons"] == 1000
        assert not shadow.is_phase5_ready()

    def test_phase5_ready_manual_stats(self, tmp_path: Path) -> None:
        """Phase 5 gate passes with sufficient traces and low divergence."""
        shadow = ShadowRouter(
            rust_router=MagicMock(), python_metacognition=MagicMock(),
            trace_path=tmp_path / "traces.jsonl",
        )
        # Manually set stats to simulate 1000 traces with 4% divergence
        shadow.stats["total_comparisons"] = 1000
        shadow.stats["system_mismatches"] = 40
        assert shadow.divergence_rate() == pytest.approx(0.04)
        assert shadow.is_phase5_ready()

    def test_phase5_boundary_exactly_5_percent(self, tmp_path: Path) -> None:
        """Phase 5 gate fails at exactly 5% (must be strictly less than)."""
        shadow = ShadowRouter(
            rust_router=MagicMock(), python_metacognition=MagicMock(),
            trace_path=tmp_path / "traces.jsonl",
        )
        shadow.stats["total_comparisons"] = 1000
        shadow.stats["system_mismatches"] = 50  # exactly 5%
        assert not shadow.is_phase5_ready()


# ---------------------------------------------------------------------------
# Tests: Python router failure resilience
# ---------------------------------------------------------------------------

class TestShadowResilience:
    """Shadow comparison must not break the primary routing path."""

    @pytest.mark.asyncio
    async def test_python_failure_still_returns_rust(self, tmp_path: Path) -> None:
        """If Python router throws, Rust decision is still returned."""
        rust = _make_rust_router(system=2, model_id="gemini-3-flash")
        python = MagicMock()
        python.assess_complexity_async = AsyncMock(side_effect=RuntimeError("LLM down"))
        shadow = ShadowRouter(
            rust_router=rust, python_metacognition=python,
            trace_path=tmp_path / "traces.jsonl",
        )
        decision = await shadow.route("Test", budget=10.0)
        assert int(decision.system) == 2
        assert decision.model_id == "gemini-3-flash"

    @pytest.mark.asyncio
    async def test_python_failure_no_comparison_logged(self, tmp_path: Path) -> None:
        """If Python router throws, no comparison is tracked."""
        rust = _make_rust_router(system=2, model_id="gemini-3-flash")
        python = MagicMock()
        python.assess_complexity_async = AsyncMock(side_effect=RuntimeError("LLM down"))
        trace_path = tmp_path / "traces.jsonl"
        shadow = ShadowRouter(
            rust_router=rust, python_metacognition=python,
            trace_path=trace_path,
        )
        await shadow.route("Test", budget=10.0)
        assert shadow.stats["total_comparisons"] == 0
        assert not trace_path.exists()
