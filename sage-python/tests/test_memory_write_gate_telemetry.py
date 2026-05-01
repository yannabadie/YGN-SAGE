from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from sage.phases.act import act


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


class _AllowingGate:
    def evaluate(self, *_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            allowed=True,
            salience_score=1.0,
            signal_breakdown={},
            reason="",
        )


class _RejectingGate:
    def evaluate(self, *_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            allowed=False,
            salience_score=0.1,
            signal_breakdown={},
            reason="below_threshold",
        )


class _EpisodicMemory:
    def __init__(self) -> None:
        self.stored: list[dict[str, Any]] = []

    async def store(self, **kwargs: Any) -> None:
        self.stored.append(kwargs)


class _MemoryAgent:
    async def extract(self, _content: str) -> Any:
        return SimpleNamespace(entities=[])


class _SemanticMemory:
    def add_extraction(self, _extraction: Any) -> None:
        pass


class _Loop:
    def __init__(self) -> None:
        self.config = SimpleNamespace(validation_level=0)
        self._skip_avr = False
        self.working_memory = _WorkingMemory()
        self.write_gate = _AllowingGate()
        self.gate_current_task = "task"
        self.gate_source_tier = "unit"
        self.episodic_memory = _EpisodicMemory()
        self.semantic_memory = _SemanticMemory()
        self.memory_agent = _MemoryAgent()
        self.causal_memory = None
        self._cb_episodic = _CircuitBreaker()
        self._cb_entity = _CircuitBreaker()
        self._cb_causal = _CircuitBreaker()
        self._skip_memory = False
        self.step_count = 1
        self.tool_turn_count = 0
        self.tool_call_count = 0
        self.executed_commands: list[str] = []

    def _emit(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    async def _execute_tool_call(self, _tool_call: Any) -> str:
        return "tool ok"


@pytest.mark.asyncio
async def test_skip_reason_tool_only_empty_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="sage.memory.write_gate")
    loop = _Loop()
    response = SimpleNamespace(
        content="",
        tool_calls=[
            SimpleNamespace(id="tc-1", name="inspect_repo", arguments={"path": "."})
        ],
        thinking="",
    )

    await act(
        "task",
        "",
        response,
        False,
        [],
        loop,  # type: ignore[arg-type]
    )

    skip_logs = [
        record.getMessage()
        for record in caplog.records
        if "memory.write_gate.skipped" in record.getMessage()
    ]
    assert skip_logs
    msg = skip_logs[0]
    assert "reason=tool_only_empty_content" in msg
    assert "content_len=0" in msg
    assert "has_tool_calls=true" in msg
    assert "tool_call_count=1" in msg
    assert "episodic_wired=true" in msg
    assert "semantic_wired=true" in msg
    assert "memory_agent_wired=true" in msg
    assert "source_tier=unit" in msg


@pytest.mark.asyncio
async def test_gate_rejected_logs_skip_reason_without_persisting(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="sage.memory.write_gate")
    loop = _Loop()
    loop.write_gate = _RejectingGate()
    response = SimpleNamespace(
        content=(
            "This response is intentionally long enough to qualify for both "
            "episodic and semantic write paths, but the salience gate rejects it."
        ),
        tool_calls=[],
        thinking="",
    )

    await act(
        "task",
        response.content,
        response,
        False,
        [],
        loop,  # type: ignore[arg-type]
    )

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "memory.write_gate.fired decision=abstain" in message
        for message in messages
    )
    assert any(
        "memory.write_gate.skipped reason=gate_rejected" in message
        for message in messages
    )
    assert loop.episodic_memory.stored == []
