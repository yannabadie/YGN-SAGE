from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sage.phases.act import act  # noqa: E402


class _CircuitBreaker:
    def should_skip(self) -> bool:
        return False

    def record_success(self) -> None:
        pass

    def record_failure(self, _exc: BaseException) -> None:
        pass


class _WorkingMemory:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def add_event(self, role: str, content: str) -> None:
        self.events.append((role, content))


class _ObservableGate:
    def __init__(self, *, allowed: bool) -> None:
        self.allowed = allowed
        self.evaluations: list[dict[str, Any]] = []

    def evaluate(self, content: str, confidence: float, **kwargs: Any) -> Any:
        self.evaluations.append(
            {"content": content, "confidence": confidence, "kwargs": kwargs}
        )
        return SimpleNamespace(
            allowed=self.allowed,
            salience_score=1.0 if self.allowed else 0.0,
            signal_breakdown={
                "confidence": confidence,
                "novelty": 1.0,
                "reliability": 1.0,
                "recency": 1.0,
                "relevance": 1.0,
            },
            reason="Allowed" if self.allowed else "gate_rejected",
        )


class _EpisodicMemory:
    def __init__(self) -> None:
        self.stores: list[dict[str, Any]] = []

    async def store(self, **kwargs: Any) -> None:
        self.stores.append(kwargs)


class _MemoryAgent:
    def __init__(self) -> None:
        self.extracts: list[str] = []

    async def extract(self, content: str) -> Any:
        self.extracts.append(content)
        return SimpleNamespace(entities=["Alpha", "Beta"])


class _SemanticMemory:
    def __init__(self) -> None:
        self.extractions: list[Any] = []

    def add_extraction(self, extraction: Any) -> None:
        self.extractions.append(extraction)


class _Loop:
    def __init__(
        self,
        *,
        gate: _ObservableGate,
        episodic_memory: _EpisodicMemory | None,
        semantic_memory: _SemanticMemory | None,
        memory_agent: _MemoryAgent | None,
    ) -> None:
        self.config = SimpleNamespace(validation_level=0)
        self._skip_avr = False
        self.working_memory = _WorkingMemory()
        self.write_gate = gate
        self.gate_current_task = "remember Alpha and Beta"
        self.gate_source_tier = "unit"
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.memory_agent = memory_agent
        self.causal_memory = None
        self._cb_episodic = _CircuitBreaker()
        self._cb_entity = _CircuitBreaker()
        self._cb_causal = _CircuitBreaker()
        self._skip_memory = False
        self.step_count = 7
        self.tool_turn_count = 0
        self.tool_call_count = 0
        self.executed_commands: list[str] = []

    def _emit(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def _execute_tool_call(self, _tool_call: Any) -> str:
        return "tool ok"


def _response() -> Any:
    return SimpleNamespace(content="", tool_calls=[], thinking="")


def _qualifying_content() -> str:
    return (
        "Alpha and Beta are persistent collaborators in the T2 memory write "
        "path smoke. This content is deliberately long enough to qualify for "
        "both episodic storage and semantic extraction without changing any "
        "write-gate threshold."
    )


def _messages_with(
    caplog: pytest.LogCaptureFixture,
    marker: str,
) -> list[str]:
    return [record.getMessage() for record in caplog.records if marker in record.getMessage()]


@pytest.mark.asyncio
async def test_act_logs_memory_backend_unwired_before_backend_write_attempt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="sage.memory.write_gate")
    gate = _ObservableGate(allowed=True)
    loop = _Loop(
        gate=gate,
        episodic_memory=None,
        semantic_memory=None,
        memory_agent=None,
    )

    await act(
        "remember Alpha and Beta",
        _qualifying_content(),
        _response(),
        False,
        [],
        loop,  # type: ignore[arg-type]
    )

    skipped = _messages_with(caplog, "memory.write_gate.skipped")
    assert any("reason=memory_backend_unwired" in msg for msg in skipped)
    assert gate.evaluations == []
    assert not _messages_with(caplog, "memory.write_gate.fired")


@pytest.mark.asyncio
async def test_act_with_wired_backends_reaches_real_write_attempt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="sage.memory.write_gate")
    gate = _ObservableGate(allowed=True)
    episodic = _EpisodicMemory()
    semantic = _SemanticMemory()
    memory_agent = _MemoryAgent()
    loop = _Loop(
        gate=gate,
        episodic_memory=episodic,
        semantic_memory=semantic,
        memory_agent=memory_agent,
    )

    await act(
        "remember Alpha and Beta",
        _qualifying_content(),
        _response(),
        False,
        [],
        loop,  # type: ignore[arg-type]
    )

    assert gate.evaluations
    assert episodic.stores
    assert memory_agent.extracts
    assert semantic.extractions
    assert any(
        "decision=persist" in msg
        for msg in _messages_with(caplog, "memory.write_gate.fired")
    )
    assert not _messages_with(caplog, "memory.write_gate.skipped")


@pytest.mark.asyncio
async def test_gate_rejected_telemetry_is_distinct_from_content_too_short(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="sage.memory.write_gate")
    gate = _ObservableGate(allowed=False)
    episodic = _EpisodicMemory()
    semantic = _SemanticMemory()
    memory_agent = _MemoryAgent()
    loop = _Loop(
        gate=gate,
        episodic_memory=episodic,
        semantic_memory=semantic,
        memory_agent=memory_agent,
    )

    await act(
        "remember Alpha and Beta",
        _qualifying_content(),
        _response(),
        False,
        [],
        loop,  # type: ignore[arg-type]
    )

    fired = _messages_with(caplog, "memory.write_gate.fired")
    assert gate.evaluations
    assert any("decision=abstain" in msg and "gate_rejected" in msg for msg in fired)
    assert not episodic.stores
    assert not memory_agent.extracts
    assert not semantic.extractions
    skipped = _messages_with(caplog, "memory.write_gate.skipped")
    assert not any("reason=content_too_short" in msg for msg in skipped)


@pytest.mark.asyncio
async def test_content_too_short_skips_before_gate_evaluation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="sage.memory.write_gate")
    gate = _ObservableGate(allowed=False)
    episodic = _EpisodicMemory()
    semantic = _SemanticMemory()
    memory_agent = _MemoryAgent()
    loop = _Loop(
        gate=gate,
        episodic_memory=episodic,
        semantic_memory=semantic,
        memory_agent=memory_agent,
    )

    await act(
        "remember Alpha and Beta",
        "short content",
        _response(),
        False,
        [],
        loop,  # type: ignore[arg-type]
    )

    skipped = _messages_with(caplog, "memory.write_gate.skipped")
    assert any("reason=content_too_short" in msg for msg in skipped)
    assert gate.evaluations == []
    assert not episodic.stores
    assert not memory_agent.extracts
    assert not semantic.extractions
    assert not _messages_with(caplog, "memory.write_gate.fired")
