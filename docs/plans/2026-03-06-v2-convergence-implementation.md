# YGN-SAGE v2 "Convergence" Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform YGN-SAGE from research prototype to functional ADK surpassing Google ADK, OpenAI Agents SDK, and LangGraph.

**Architecture:** 8 chantiers in dependency order: Cleanup -> EventBus -> Multi-Agent + Guardrails + Memory (parallel) -> Dashboard -> Benchmarks -> Tests.

**Tech Stack:** Python 3.12+, aiosqlite, FastAPI, WebSocket, Tailwind CSS, Chart.js, Z3. No new frameworks.

---

## Task 0: Cleanup -- Remove Credibility-Damaging Artifacts

**Files:**
- Delete: `docs/plans/official_benchmark_proof.json`
- Delete: `docs/plans/benchmark_results.json`
- Delete: `docs/plans/benchmark_dashboard.html`
- Delete: `docs/plans/cybergym_benchmark_proof.json`
- Delete: `docs/plans/real_benchmark_proof.json`
- Delete: `sage-discover/official_benchmark_suite.py`
- Delete: `sage-discover/cybergym_benchmark_suite.py`
- Delete: `research_journal/` (entire directory, 86 files)
- Modify: `Cargo.toml:5` (root workspace)
- Create: `LICENSE`

**Step 1: Delete synthetic benchmarks and stale directories**

```bash
cd C:/Code/YGN-SAGE
git rm docs/plans/official_benchmark_proof.json
git rm docs/plans/benchmark_results.json
git rm docs/plans/benchmark_dashboard.html
git rm docs/plans/cybergym_benchmark_proof.json
git rm docs/plans/real_benchmark_proof.json
git rm sage-discover/official_benchmark_suite.py
git rm sage-discover/cybergym_benchmark_suite.py
git rm -r research_journal/
```

**Step 2: Fix license inconsistency**

In `Cargo.toml` (root workspace), change line 5:
```toml
license = "Proprietary"
```

**Step 3: Create LICENSE file**

```
Copyright (c) 2026 Yann Abadie. All rights reserved.

This software is proprietary and confidential.
Unauthorized copying, distribution, or use is strictly prohibited.
```

**Step 4: Triage debug/ directory**

Keep: `debug/run_ygn_sage_agent.py` (useful launcher)
Delete everything else in `debug/`:
```bash
git rm debug/debug_h96.py debug/test_signal.py debug/test_codex_fix.py
git rm debug/test_simd.py debug/test_ebpf_eval.py debug/test_ebpf_raw.py
git rm debug/test_cold_start.py debug/test_simd_retrieval.py debug/repro_system2.py
```

**Step 5: Commit**

```bash
git add -A
git commit -m "chore: remove synthetic benchmarks, stale artifacts, fix license"
```

---

## Task 1: EventBus -- Central Nervous System

**Files:**
- Create: `sage-python/src/sage/events/__init__.py`
- Create: `sage-python/src/sage/events/bus.py`
- Test: `sage-python/tests/test_event_bus.py`
- Modify: `sage-python/src/sage/agent_loop.py`
- Modify: `sage-python/src/sage/boot.py`

**Step 1: Write failing tests**

```python
# sage-python/tests/test_event_bus.py
import pytest
import asyncio
from sage.events.bus import EventBus
from sage.agent_loop import AgentEvent

def make_event(type_: str = "TEST", step: int = 1) -> AgentEvent:
    import time
    return AgentEvent(type=type_, step=step, timestamp=time.time())

def test_emit_and_query():
    bus = EventBus()
    evt = make_event("THINK", 1)
    bus.emit(evt)
    assert len(bus.query()) == 1
    assert bus.query()[0].type == "THINK"

def test_query_filters_by_phase():
    bus = EventBus()
    bus.emit(make_event("THINK", 1))
    bus.emit(make_event("ACT", 2))
    bus.emit(make_event("THINK", 3))
    assert len(bus.query(phase="THINK")) == 2
    assert len(bus.query(phase="ACT")) == 1

def test_query_last_n():
    bus = EventBus()
    for i in range(20):
        bus.emit(make_event("STEP", i))
    assert len(bus.query(last_n=5)) == 5
    assert bus.query(last_n=5)[-1].step == 19

def test_subscribe_receives_events():
    bus = EventBus()
    received = []
    bus.subscribe(lambda e: received.append(e))
    bus.emit(make_event("ROUTING", 0))
    assert len(received) == 1

def test_unsubscribe():
    bus = EventBus()
    received = []
    sub_id = bus.subscribe(lambda e: received.append(e))
    bus.emit(make_event("A"))
    bus.unsubscribe(sub_id)
    bus.emit(make_event("B"))
    assert len(received) == 1

@pytest.mark.asyncio
async def test_stream_yields_events():
    bus = EventBus()
    results = []

    async def consumer():
        async for evt in bus.stream():
            results.append(evt)
            if len(results) >= 3:
                break

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.05)
    for i in range(3):
        bus.emit(make_event("S", i))
        await asyncio.sleep(0.01)
    await asyncio.wait_for(task, timeout=2.0)
    assert len(results) == 3

def test_buffer_max_size():
    bus = EventBus(max_buffer=100)
    for i in range(200):
        bus.emit(make_event("X", i))
    assert len(bus.query(last_n=200)) == 100
```

**Step 2: Run tests to verify they fail**

```bash
cd sage-python && python -m pytest tests/test_event_bus.py -v
```
Expected: FAIL (ImportError: cannot import EventBus)

**Step 3: Implement EventBus**

```python
# sage-python/src/sage/events/__init__.py
from sage.events.bus import EventBus
__all__ = ["EventBus"]
```

```python
# sage-python/src/sage/events/bus.py
"""In-process event bus. Zero dependencies. Thread-safe."""
from __future__ import annotations

import asyncio
import threading
import uuid
from collections import deque
from typing import Any, AsyncIterator, Callable

from sage.agent_loop import AgentEvent


class EventBus:
    """Central event bus for YGN-SAGE. All components emit here."""

    def __init__(self, max_buffer: int = 5000):
        self._buffer: deque[AgentEvent] = deque(maxlen=max_buffer)
        self._subscribers: dict[str, Callable[[AgentEvent], None]] = {}
        self._async_queues: list[asyncio.Queue] = []
        self._lock = threading.Lock()

    def emit(self, event: AgentEvent) -> None:
        with self._lock:
            self._buffer.append(event)
            for cb in self._subscribers.values():
                try:
                    cb(event)
                except Exception:
                    pass
            for q in self._async_queues:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass

    def subscribe(self, callback: Callable[[AgentEvent], None]) -> str:
        sub_id = str(uuid.uuid4())
        with self._lock:
            self._subscribers[sub_id] = callback
        return sub_id

    def unsubscribe(self, sub_id: str) -> None:
        with self._lock:
            self._subscribers.pop(sub_id, None)

    async def stream(self) -> AsyncIterator[AgentEvent]:
        q: asyncio.Queue[AgentEvent] = asyncio.Queue(maxsize=1000)
        with self._lock:
            self._async_queues.append(q)
        try:
            while True:
                event = await q.get()
                yield event
        finally:
            with self._lock:
                self._async_queues.remove(q)

    def query(self, phase: str | None = None, last_n: int = 50) -> list[AgentEvent]:
        with self._lock:
            events = list(self._buffer)
        if phase:
            events = [e for e in events if e.type == phase]
        return events[-last_n:]
```

**Step 4: Run tests to verify they pass**

```bash
cd sage-python && python -m pytest tests/test_event_bus.py -v
```
Expected: 8 passed

**Step 5: Wire EventBus into boot.py and AgentLoop**

In `sage-python/src/sage/boot.py`:
- Add `from sage.events.bus import EventBus` to imports
- Add `event_bus: EventBus | None = None` parameter to `boot_agent_system()`
- Create `event_bus = event_bus or EventBus()` in function body
- Pass `on_event=event_bus.emit` to `AgentLoop()`
- Store `event_bus` on `AgentSystem`

Add `event_bus: EventBus` field to `AgentSystem` dataclass.

In `sage-python/src/sage/agent_loop.py`:
- Remove `STREAM_FILE` constant and `_default_event_handler` method
- The default handler becomes a no-op logger. EventBus is the primary handler.

**Step 6: Run all existing tests to verify no regressions**

```bash
cd sage-python && python -m pytest tests/ -v
```
Expected: 200+ passed

**Step 7: Commit**

```bash
git add -A
git commit -m "feat(events): add EventBus central event system, wire to boot + agent loop"
```

---

## Task 2: Multi-Agent Composition Patterns

**Files:**
- Create: `sage-python/src/sage/agents/__init__.py`
- Create: `sage-python/src/sage/agents/sequential.py`
- Create: `sage-python/src/sage/agents/parallel.py`
- Create: `sage-python/src/sage/agents/loop_agent.py`
- Create: `sage-python/src/sage/agents/handoff.py`
- Test: `sage-python/tests/test_agents_composition.py`

**Step 1: Write failing tests**

```python
# sage-python/tests/test_agents_composition.py
import pytest
from unittest.mock import AsyncMock
from sage.agents.sequential import SequentialAgent
from sage.agents.parallel import ParallelAgent
from sage.agents.loop_agent import LoopAgent
from sage.agents.handoff import Handoff, HandoffResult


class MockRunnable:
    """Minimal agent-like object for testing."""
    def __init__(self, name: str, response: str):
        self.name = name
        self._response = response
        self.last_input = None

    async def run(self, task: str) -> str:
        self.last_input = task
        return self._response


@pytest.mark.asyncio
async def test_sequential_chains_output():
    a = MockRunnable("a", "step1")
    b = MockRunnable("b", "step2")
    seq = SequentialAgent(name="seq", agents=[a, b])
    result = await seq.run("start")
    assert result == "step2"
    assert a.last_input == "start"
    assert b.last_input == "step1"

@pytest.mark.asyncio
async def test_parallel_runs_concurrently():
    a = MockRunnable("a", "result_a")
    b = MockRunnable("b", "result_b")
    par = ParallelAgent(name="par", agents=[a, b])
    result = await par.run("task")
    assert "result_a" in result
    assert "result_b" in result

@pytest.mark.asyncio
async def test_parallel_with_custom_aggregator():
    a = MockRunnable("a", "3")
    b = MockRunnable("b", "7")
    par = ParallelAgent(
        name="par", agents=[a, b],
        aggregator=lambda results: str(sum(int(r) for r in results.values()))
    )
    result = await par.run("compute")
    assert result == "10"

@pytest.mark.asyncio
async def test_loop_agent_runs_until_condition():
    counter = {"n": 0}
    class CountAgent:
        name = "counter"
        async def run(self, task):
            counter["n"] += 1
            return f"count={counter['n']}"

    loop = LoopAgent(
        name="loop",
        agent=CountAgent(),
        max_iterations=10,
        exit_condition=lambda result: "count=3" in result,
    )
    result = await loop.run("count up")
    assert counter["n"] == 3
    assert "count=3" in result

@pytest.mark.asyncio
async def test_loop_agent_respects_max_iterations():
    class NeverDone:
        name = "never"
        async def run(self, task):
            return "nope"

    loop = LoopAgent(
        name="loop",
        agent=NeverDone(),
        max_iterations=5,
        exit_condition=lambda r: False,
    )
    result = await loop.run("go")
    assert result == "nope"  # Last result

@pytest.mark.asyncio
async def test_handoff_transfers_to_target():
    target = MockRunnable("specialist", "handled by specialist")
    handoff = Handoff(target=target, description="For code tasks")
    result = await handoff.execute("fix this bug")
    assert result.output == "handled by specialist"
    assert result.target_name == "specialist"

@pytest.mark.asyncio
async def test_handoff_with_input_filter():
    target = MockRunnable("t", "done")
    handoff = Handoff(
        target=target,
        description="test",
        input_filter=lambda task: task.upper(),
    )
    result = await handoff.execute("hello")
    assert target.last_input == "HELLO"

@pytest.mark.asyncio
async def test_handoff_callback():
    target = MockRunnable("t", "done")
    callback_calls = []
    handoff = Handoff(
        target=target,
        description="test",
        on_handoff=lambda name, task: callback_calls.append((name, task)),
    )
    await handoff.execute("task1")
    assert len(callback_calls) == 1
    assert callback_calls[0] == ("t", "task1")
```

**Step 2: Run tests to verify they fail**

```bash
cd sage-python && python -m pytest tests/test_agents_composition.py -v
```
Expected: FAIL (ImportError)

**Step 3: Implement composition patterns**

```python
# sage-python/src/sage/agents/__init__.py
from sage.agents.sequential import SequentialAgent
from sage.agents.parallel import ParallelAgent
from sage.agents.loop_agent import LoopAgent
from sage.agents.handoff import Handoff, HandoffResult
__all__ = ["SequentialAgent", "ParallelAgent", "LoopAgent", "Handoff", "HandoffResult"]
```

```python
# sage-python/src/sage/agents/sequential.py
"""Sequential agent composition: execute agents in series."""
from __future__ import annotations
from typing import Any


class SequentialAgent:
    def __init__(self, name: str, agents: list[Any], shared_state: dict | None = None):
        self.name = name
        self.agents = agents
        self.shared_state = shared_state or {}

    async def run(self, task: str) -> str:
        current_input = task
        for agent in self.agents:
            current_input = await agent.run(current_input)
        return current_input
```

```python
# sage-python/src/sage/agents/parallel.py
"""Parallel agent composition: execute agents concurrently."""
from __future__ import annotations
import asyncio
from typing import Any, Callable


class ParallelAgent:
    def __init__(self, name: str, agents: list[Any],
                 aggregator: Callable[[dict[str, str]], str] | None = None):
        self.name = name
        self.agents = agents
        self.aggregator = aggregator

    async def run(self, task: str) -> str:
        tasks = {agent.name: agent.run(task) for agent in self.agents}
        results = {}
        for name, coro in tasks.items():
            results[name] = await coro
        if self.aggregator:
            return self.aggregator(results)
        return "\n\n".join(f"[{name}]: {result}" for name, result in results.items())
```

```python
# sage-python/src/sage/agents/loop_agent.py
"""Loop agent: execute an agent repeatedly until exit condition."""
from __future__ import annotations
from typing import Any, Callable


class LoopAgent:
    def __init__(self, name: str, agent: Any, max_iterations: int = 10,
                 exit_condition: Callable[[str], bool] | None = None):
        self.name = name
        self.agent = agent
        self.max_iterations = max_iterations
        self.exit_condition = exit_condition or (lambda _: False)

    async def run(self, task: str) -> str:
        current_input = task
        result = ""
        for _ in range(self.max_iterations):
            result = await self.agent.run(current_input)
            if self.exit_condition(result):
                break
            current_input = result
        return result
```

```python
# sage-python/src/sage/agents/handoff.py
"""Agent handoff: transfer control to a specialist agent."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class HandoffResult:
    output: str
    target_name: str


class Handoff:
    def __init__(self, target: Any, description: str,
                 input_filter: Callable[[str], str] | None = None,
                 on_handoff: Callable[[str, str], None] | None = None):
        self.target = target
        self.description = description
        self.input_filter = input_filter
        self.on_handoff = on_handoff

    async def execute(self, task: str) -> HandoffResult:
        if self.on_handoff:
            self.on_handoff(self.target.name, task)
        filtered_task = self.input_filter(task) if self.input_filter else task
        output = await self.target.run(filtered_task)
        return HandoffResult(output=output, target_name=self.target.name)
```

**Step 4: Run tests**

```bash
cd sage-python && python -m pytest tests/test_agents_composition.py -v
```
Expected: 8 passed

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(agents): add Sequential/Parallel/Loop composition + Handoff"
```

---

## Task 3: Guardrails Framework

**Files:**
- Create: `sage-python/src/sage/guardrails/__init__.py`
- Create: `sage-python/src/sage/guardrails/base.py`
- Create: `sage-python/src/sage/guardrails/builtin.py`
- Test: `sage-python/tests/test_guardrails.py`

**Step 1: Write failing tests**

```python
# sage-python/tests/test_guardrails.py
import pytest
from sage.guardrails.base import GuardrailResult, GuardrailPipeline
from sage.guardrails.builtin import CostGuardrail, SchemaGuardrail


def test_guardrail_result_passed():
    r = GuardrailResult(passed=True)
    assert r.passed
    assert r.severity == "info"

def test_guardrail_result_blocked():
    r = GuardrailResult(passed=False, reason="over budget", severity="block")
    assert not r.passed
    assert r.reason == "over budget"

@pytest.mark.asyncio
async def test_cost_guardrail_passes_under_budget():
    guard = CostGuardrail(max_usd=1.0)
    result = await guard.check(context={"cost_usd": 0.5})
    assert result.passed

@pytest.mark.asyncio
async def test_cost_guardrail_blocks_over_budget():
    guard = CostGuardrail(max_usd=0.10)
    result = await guard.check(context={"cost_usd": 0.15})
    assert not result.passed
    assert "budget" in result.reason.lower()

@pytest.mark.asyncio
async def test_schema_guardrail_passes_valid():
    guard = SchemaGuardrail(required_fields=["answer"])
    result = await guard.check(output='{"answer": "42"}')
    assert result.passed

@pytest.mark.asyncio
async def test_schema_guardrail_fails_missing_field():
    guard = SchemaGuardrail(required_fields=["answer", "reasoning"])
    result = await guard.check(output='{"answer": "42"}')
    assert not result.passed

@pytest.mark.asyncio
async def test_pipeline_runs_all_guards():
    g1 = CostGuardrail(max_usd=10.0)
    g2 = SchemaGuardrail(required_fields=["answer"])
    pipeline = GuardrailPipeline([g1, g2])
    results = await pipeline.check_all(
        context={"cost_usd": 0.5},
        output='{"answer": "yes"}',
    )
    assert all(r.passed for r in results)

@pytest.mark.asyncio
async def test_pipeline_reports_first_failure():
    g1 = CostGuardrail(max_usd=0.01)  # Will fail
    g2 = SchemaGuardrail(required_fields=["answer"])
    pipeline = GuardrailPipeline([g1, g2])
    results = await pipeline.check_all(
        context={"cost_usd": 1.0},
        output='{"answer": "yes"}',
    )
    assert not results[0].passed
    assert results[1].passed
```

**Step 2: Run tests to verify they fail**

```bash
cd sage-python && python -m pytest tests/test_guardrails.py -v
```
Expected: FAIL (ImportError)

**Step 3: Implement guardrails**

```python
# sage-python/src/sage/guardrails/__init__.py
from sage.guardrails.base import GuardrailResult, GuardrailPipeline
from sage.guardrails.builtin import CostGuardrail, SchemaGuardrail
__all__ = ["GuardrailResult", "GuardrailPipeline", "CostGuardrail", "SchemaGuardrail"]
```

```python
# sage-python/src/sage/guardrails/base.py
"""Guardrail base classes."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class GuardrailResult:
    passed: bool
    reason: str = ""
    severity: str = "info"  # info, warning, block


class Guardrail:
    """Base class for all guardrails."""
    name: str = "base"

    async def check(self, input: str = "", output: str = "",
                    context: dict[str, Any] | None = None) -> GuardrailResult:
        return GuardrailResult(passed=True)


class GuardrailPipeline:
    """Run multiple guardrails and collect results."""

    def __init__(self, guardrails: list[Guardrail]):
        self.guardrails = guardrails

    async def check_all(self, input: str = "", output: str = "",
                        context: dict[str, Any] | None = None) -> list[GuardrailResult]:
        results = []
        for g in self.guardrails:
            r = await g.check(input=input, output=output, context=context or {})
            results.append(r)
        return results

    def any_blocked(self, results: list[GuardrailResult]) -> bool:
        return any(not r.passed and r.severity == "block" for r in results)
```

```python
# sage-python/src/sage/guardrails/builtin.py
"""Built-in guardrails."""
from __future__ import annotations
import json
from typing import Any
from sage.guardrails.base import Guardrail, GuardrailResult


class CostGuardrail(Guardrail):
    name = "cost"

    def __init__(self, max_usd: float = 1.0):
        self.max_usd = max_usd

    async def check(self, input: str = "", output: str = "",
                    context: dict[str, Any] | None = None) -> GuardrailResult:
        ctx = context or {}
        current = ctx.get("cost_usd", 0.0)
        if current > self.max_usd:
            return GuardrailResult(
                passed=False,
                reason=f"Budget exceeded: ${current:.4f} > ${self.max_usd:.4f}",
                severity="block",
            )
        return GuardrailResult(passed=True)


class SchemaGuardrail(Guardrail):
    name = "schema"

    def __init__(self, required_fields: list[str] | None = None):
        self.required_fields = required_fields or []

    async def check(self, input: str = "", output: str = "",
                    context: dict[str, Any] | None = None) -> GuardrailResult:
        if not self.required_fields:
            return GuardrailResult(passed=True)
        try:
            data = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            return GuardrailResult(
                passed=False, reason="Output is not valid JSON", severity="block"
            )
        missing = [f for f in self.required_fields if f not in data]
        if missing:
            return GuardrailResult(
                passed=False,
                reason=f"Missing required fields: {missing}",
                severity="block",
            )
        return GuardrailResult(passed=True)
```

**Step 4: Run tests**

```bash
cd sage-python && python -m pytest tests/test_guardrails.py -v
```
Expected: 8 passed

**Step 5: Commit**

```bash
git add -A
git commit -m "feat(guardrails): add GuardrailPipeline + CostGuardrail + SchemaGuardrail"
```

---

## Task 4: Memory v2 -- Episodic Persistence + Semantic Memory

**Files:**
- Modify: `sage-python/src/sage/memory/episodic.py`
- Create: `sage-python/src/sage/memory/semantic.py`
- Test: `sage-python/tests/test_memory_v2.py`
- Modify: `sage-python/pyproject.toml` (add aiosqlite)

**Step 1: Add aiosqlite dependency**

In `sage-python/pyproject.toml`, add `"aiosqlite>=0.20"` to the dependencies list.

**Step 2: Write failing tests**

```python
# sage-python/tests/test_memory_v2.py
import pytest
from pathlib import Path
from sage.memory.episodic import EpisodicMemory
from sage.memory.semantic import SemanticMemory
from sage.memory.memory_agent import ExtractionResult


@pytest.mark.asyncio
async def test_episodic_sqlite_store_and_search(tmp_path):
    db = str(tmp_path / "test.db")
    mem = EpisodicMemory(db_path=db)
    await mem.initialize()
    await mem.store("k1", "Python was created by Guido in 1991", {"domain": "history"})
    results = await mem.search("Python")
    assert len(results) >= 1
    assert "1991" in results[0]["content"]

@pytest.mark.asyncio
async def test_episodic_sqlite_persistence(tmp_path):
    db = str(tmp_path / "test.db")
    mem1 = EpisodicMemory(db_path=db)
    await mem1.initialize()
    await mem1.store("k1", "persistent data", {})
    del mem1

    mem2 = EpisodicMemory(db_path=db)
    await mem2.initialize()
    results = await mem2.search("persistent")
    assert len(results) == 1

@pytest.mark.asyncio
async def test_episodic_count(tmp_path):
    db = str(tmp_path / "test.db")
    mem = EpisodicMemory(db_path=db)
    await mem.initialize()
    await mem.store("a", "data a", {})
    await mem.store("b", "data b", {})
    assert await mem.count() == 2

@pytest.mark.asyncio
async def test_episodic_delete(tmp_path):
    db = str(tmp_path / "test.db")
    mem = EpisodicMemory(db_path=db)
    await mem.initialize()
    await mem.store("k1", "to delete", {})
    await mem.delete("k1")
    assert await mem.count() == 0

@pytest.mark.asyncio
async def test_episodic_update(tmp_path):
    db = str(tmp_path / "test.db")
    mem = EpisodicMemory(db_path=db)
    await mem.initialize()
    await mem.store("k1", "old content", {})
    await mem.update("k1", "new content", {})
    results = await mem.search("new")
    assert len(results) == 1
    assert "new content" in results[0]["content"]

@pytest.mark.asyncio
async def test_episodic_in_memory_fallback():
    """When no db_path, use in-memory mode (backward compatible)."""
    mem = EpisodicMemory()
    await mem.initialize()
    await mem.store("k1", "in memory", {})
    results = await mem.search("memory")
    assert len(results) == 1

def test_semantic_memory_add_and_query():
    sem = SemanticMemory()
    extraction = ExtractionResult(
        entities=["Python", "Guido"],
        relationships=[("Python", "created_by", "Guido")],
        summary="Python was created by Guido",
    )
    sem.add_extraction(extraction)
    assert sem.entity_count() == 2
    rels = sem.query_entities("Python")
    assert len(rels) >= 1
    assert rels[0] == ("Python", "created_by", "Guido")

def test_semantic_memory_context_for_task():
    sem = SemanticMemory()
    sem.add_extraction(ExtractionResult(
        entities=["Z3", "SMT", "Solver"],
        relationships=[("Z3", "is_a", "SMT"), ("SMT", "type_of", "Solver")],
        summary="Z3 is an SMT solver",
    ))
    context = sem.get_context_for("Tell me about Z3")
    assert "Z3" in context
    assert "SMT" in context

def test_semantic_memory_empty_query():
    sem = SemanticMemory()
    assert sem.entity_count() == 0
    assert sem.query_entities("nothing") == []
    assert sem.get_context_for("anything") == ""
```

**Step 3: Run tests to verify they fail**

```bash
cd sage-python && pip install aiosqlite && python -m pytest tests/test_memory_v2.py -v
```
Expected: FAIL

**Step 4: Rewrite EpisodicMemory with SQLite backend**

Replace `sage-python/src/sage/memory/episodic.py` entirely:

```python
"""Episodic memory: keyword-searchable store with optional SQLite persistence."""
from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)


class EpisodicMemory:
    """CRUD episodic store. SQLite-backed if db_path provided, else in-memory."""

    def __init__(self, db_path: str | None = None):
        self._db_path = db_path
        self._conn = None
        self._in_memory: list[dict[str, Any]] = []  # Fallback

    async def initialize(self) -> None:
        if self._db_path:
            import aiosqlite
            self._conn = await aiosqlite.connect(self._db_path)
            await self._conn.execute(
                "CREATE TABLE IF NOT EXISTS episodes ("
                "  key TEXT PRIMARY KEY,"
                "  content TEXT NOT NULL,"
                "  metadata TEXT DEFAULT '{}',"
                "  created_at REAL DEFAULT (julianday('now'))"
                ")"
            )
            await self._conn.commit()

    async def store(self, key: str, content: str, metadata: dict[str, Any]) -> None:
        if self._conn:
            await self._conn.execute(
                "INSERT OR REPLACE INTO episodes (key, content, metadata) VALUES (?, ?, ?)",
                (key, content, json.dumps(metadata)),
            )
            await self._conn.commit()
        else:
            self._in_memory = [e for e in self._in_memory if e["key"] != key]
            self._in_memory.append({"key": key, "content": content, "metadata": metadata})

    async def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        if self._conn:
            cursor = await self._conn.execute(
                "SELECT key, content, metadata FROM episodes WHERE content LIKE ? LIMIT ?",
                (f"%{query}%", limit),
            )
            rows = await cursor.fetchall()
            return [{"key": r[0], "content": r[1], "metadata": json.loads(r[2])} for r in rows]
        return [e for e in self._in_memory if query.lower() in e["content"].lower()][:limit]

    async def update(self, key: str, content: str, metadata: dict[str, Any]) -> None:
        await self.store(key, content, metadata)  # UPSERT

    async def delete(self, key: str) -> None:
        if self._conn:
            await self._conn.execute("DELETE FROM episodes WHERE key = ?", (key,))
            await self._conn.commit()
        else:
            self._in_memory = [e for e in self._in_memory if e["key"] != key]

    async def count(self) -> int:
        if self._conn:
            cursor = await self._conn.execute("SELECT COUNT(*) FROM episodes")
            row = await cursor.fetchone()
            return row[0] if row else 0
        return len(self._in_memory)

    async def list_all(self, limit: int = 100) -> list[dict[str, Any]]:
        if self._conn:
            cursor = await self._conn.execute(
                "SELECT key, content, metadata FROM episodes ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
            return [{"key": r[0], "content": r[1], "metadata": json.loads(r[2])} for r in rows]
        return self._in_memory[-limit:]
```

**Step 5: Create SemanticMemory**

```python
# sage-python/src/sage/memory/semantic.py
"""Semantic memory: in-memory entity-relation graph built by MemoryAgent."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from sage.memory.memory_agent import ExtractionResult


class SemanticMemory:
    """In-memory entity graph. Populated by MemoryAgent extractions."""

    def __init__(self):
        self._entities: set[str] = set()
        self._relations: list[tuple[str, str, str]] = []
        self._entity_index: dict[str, list[int]] = defaultdict(list)

    def add_extraction(self, result: ExtractionResult) -> None:
        for entity in result.entities:
            self._entities.add(entity)
        for rel in result.relationships:
            idx = len(self._relations)
            self._relations.append(rel)
            if len(rel) >= 3:
                self._entity_index[rel[0]].append(idx)
                self._entity_index[rel[2]].append(idx)

    def entity_count(self) -> int:
        return len(self._entities)

    def query_entities(self, entity: str, hops: int = 1) -> list[tuple[str, str, str]]:
        indices = self._entity_index.get(entity, [])
        return [self._relations[i] for i in indices]

    def get_context_for(self, task: str) -> str:
        if not self._entities:
            return ""
        task_lower = task.lower()
        relevant = [e for e in self._entities if e.lower() in task_lower]
        if not relevant:
            return ""
        lines = []
        for entity in relevant:
            rels = self.query_entities(entity)
            for s, p, o in rels:
                lines.append(f"{s} -> {p} -> {o}")
        return "\n".join(lines)
```

**Step 6: Run tests**

```bash
cd sage-python && python -m pytest tests/test_memory_v2.py -v
```
Expected: 12 passed

**Step 7: Update boot.py to wire SemanticMemory + initialize episodic**

In `boot.py`:
- Add `from sage.memory.semantic import SemanticMemory` import
- Create `semantic_memory = SemanticMemory()`
- Pass `db_path=os.path.expanduser("~/.sage/episodic.db")` to `EpisodicMemory()`
- Attach `loop.semantic_memory = semantic_memory` and `loop.memory_agent = memory_agent`
- Call `await episodic_memory.initialize()` in `AgentSystem.run()` before first use

**Step 8: Run all tests**

```bash
cd sage-python && python -m pytest tests/ -v
```
Expected: 210+ passed

**Step 9: Commit**

```bash
git add -A
git commit -m "feat(memory): SQLite episodic persistence + SemanticMemory entity graph"
```

---

## Task 5: Dashboard -- Fully Wired Single-File HTML

**Files:**
- Rewrite: `ui/app.py`
- Rewrite: `ui/static/index.html`

**Step 1: Rewrite ui/app.py**

Replace the dashboard backend to use EventBus instead of JSONL file polling. Key changes:
- Import EventBus, boot_agent_system
- Global `event_bus = EventBus()`
- `WS /ws` endpoint: `async for event in event_bus.stream()` -> `websocket.send_json()`
- `POST /api/task`: boot system with event_bus, run in background task
- `POST /api/benchmark`: launch benchmark runner
- `GET /api/memory/stats`: return 4-tier memory counts
- `GET /api/topology`: return active agent list from pool
- Remove all JSONL file reading code

The backend must remain a single file under 300 lines.

**Step 2: Rewrite ui/static/index.html**

Complete single-file dashboard with:
- Tailwind CSS (CDN)
- Chart.js (CDN) for sparklines
- Native WebSocket connection to `/ws`
- All 8 sections from the design: Routing, Response, Memory, Topology, Evolution, Guardrails, Events, Benchmarks
- Color scheme: dark theme, S1=green, S2=amber, S3=red
- Responsive layout using CSS grid

The HTML must remain a single file. Target ~800-1000 lines.

**Step 3: Manual test**

```bash
cd C:/Code/YGN-SAGE && python ui/app.py
# Open http://localhost:8000
# Submit a task, verify WebSocket events appear in real-time
```

**Step 4: Commit**

```bash
git add ui/app.py ui/static/index.html
git commit -m "feat(dashboard): fully wired real-time dashboard via EventBus WebSocket"
```

---

## Task 6: Benchmark Pipeline

**Files:**
- Create: `sage-python/src/sage/bench/__init__.py`
- Create: `sage-python/src/sage/bench/runner.py`
- Create: `sage-python/src/sage/bench/humaneval.py`
- Create: `sage-python/src/sage/bench/routing.py`
- Create: `sage-python/src/sage/bench/__main__.py`
- Test: `sage-python/tests/test_bench.py`

**Step 1: Write failing tests**

```python
# sage-python/tests/test_bench.py
import sys, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest
from sage.bench.runner import BenchmarkRunner, BenchReport, TaskResult
from sage.bench.routing import RoutingAccuracyBench
from sage.events.bus import EventBus


def test_bench_report_creation():
    report = BenchReport(
        benchmark="test",
        total=10,
        passed=7,
        failed=3,
        errors=0,
        pass_rate=0.7,
        avg_latency_ms=150.0,
        avg_cost_usd=0.005,
        routing_breakdown={"S1": 3, "S2": 5, "S3": 2},
        results=[],
        model_config={},
    )
    assert report.pass_rate == 0.7
    assert report.avg_latency_ms == 150.0

def test_task_result_creation():
    r = TaskResult(
        task_id="test_001",
        passed=True,
        system_used=2,
        latency_ms=1234.5,
        cost_usd=0.003,
    )
    assert r.passed
    assert r.system_used == 2
    assert r.latency_ms == 1234.5

@pytest.mark.asyncio
async def test_routing_accuracy_bench():
    """Routing bench works with mock metacognition."""
    from sage.strategy.metacognition import MetacognitiveController
    mc = MetacognitiveController()
    bench = RoutingAccuracyBench(metacognition=mc)
    report = await bench.run()
    assert report.total > 0
    assert 0.0 <= report.pass_rate <= 1.0
    assert report.avg_latency_ms >= 0
    assert report.benchmark == "routing_accuracy"
```

**Step 2: Run tests to verify they fail**

```bash
cd sage-python && python -m pytest tests/test_bench.py -v
```
Expected: FAIL

**Step 3: Implement benchmark framework**

```python
# sage-python/src/sage/bench/__init__.py
from sage.bench.runner import BenchmarkRunner, BenchReport, TaskResult
__all__ = ["BenchmarkRunner", "BenchReport", "TaskResult"]
```

```python
# sage-python/src/sage/bench/runner.py
"""Benchmark runner: executes standardized benchmarks."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskResult:
    task_id: str
    passed: bool
    system_used: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    sandbox_executions: int = 0
    memory_events: int = 0
    escalations: int = 0
    z3_checks: int = 0
    tokens_used: int = 0
    error: str = ""


@dataclass
class BenchReport:
    benchmark: str
    total: int
    passed: int
    failed: int
    errors: int
    pass_rate: float
    avg_latency_ms: float
    avg_cost_usd: float
    routing_breakdown: dict[str, int] = field(default_factory=dict)
    results: list[TaskResult] = field(default_factory=list)
    model_config: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    @staticmethod
    def from_results(benchmark: str, results: list[TaskResult],
                     model_config: dict | None = None) -> BenchReport:
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed and not r.error)
        errors = sum(1 for r in results if r.error)
        avg_lat = sum(r.latency_ms for r in results) / max(total, 1)
        avg_cost = sum(r.cost_usd for r in results) / max(total, 1)
        breakdown = {}
        for r in results:
            key = f"S{r.system_used}"
            breakdown[key] = breakdown.get(key, 0) + 1
        return BenchReport(
            benchmark=benchmark,
            total=total,
            passed=passed,
            failed=failed,
            errors=errors,
            pass_rate=passed / max(total, 1),
            avg_latency_ms=round(avg_lat, 1),
            avg_cost_usd=round(avg_cost, 6),
            routing_breakdown=breakdown,
            results=results,
            model_config=model_config or {},
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
```

```python
# sage-python/src/sage/bench/routing.py
"""Routing accuracy benchmark: measures MetacognitiveController precision."""
from __future__ import annotations

import time
from sage.bench.runner import BenchReport, TaskResult


LABELED_TASKS = [
    {"task": "What is 2+2?", "expected": 1},
    {"task": "What is the capital of France?", "expected": 1},
    {"task": "Translate hello to Spanish", "expected": 1},
    {"task": "What color is the sky?", "expected": 1},
    {"task": "Define photosynthesis in one sentence", "expected": 1},
    {"task": "Name three prime numbers", "expected": 1},
    {"task": "What year was Python created?", "expected": 1},
    {"task": "Convert 100 Fahrenheit to Celsius", "expected": 1},
    {"task": "What is HTTP?", "expected": 1},
    {"task": "Is 17 a prime number?", "expected": 1},
    {"task": "Debug this Python function that calculates fibonacci", "expected": 2},
    {"task": "Fix the off-by-one error in this binary search", "expected": 2},
    {"task": "Optimize this sorting algorithm for large datasets", "expected": 2},
    {"task": "Write a function to parse CSV files with error handling", "expected": 2},
    {"task": "Refactor this class to use dependency injection", "expected": 2},
    {"task": "Create a REST API endpoint for user authentication", "expected": 2},
    {"task": "Write unit tests for this database connection module", "expected": 2},
    {"task": "Find and fix the memory leak in this server code", "expected": 2},
    {"task": "Implement a thread-safe cache with TTL expiration", "expected": 2},
    {"task": "Design a retry mechanism with exponential backoff", "expected": 2},
    {"task": "Prove that this merge sort implementation always terminates and produces a sorted output", "expected": 3},
    {"task": "Formally verify that this concurrent queue never deadlocks", "expected": 3},
    {"task": "Prove the correctness of this consensus algorithm", "expected": 3},
    {"task": "Verify that this memory allocator never has use-after-free", "expected": 3},
    {"task": "Prove that this type system is sound and complete", "expected": 3},
    {"task": "Formally verify the safety invariants of this smart contract", "expected": 3},
    {"task": "Prove termination of this recursive descent parser", "expected": 3},
    {"task": "Verify that this distributed lock satisfies mutual exclusion", "expected": 3},
    {"task": "Prove that this garbage collector preserves reachability", "expected": 3},
    {"task": "Formally verify this cryptographic protocol against known attacks", "expected": 3},
]


class RoutingAccuracyBench:
    def __init__(self, metacognition):
        self.metacognition = metacognition

    async def run(self) -> BenchReport:
        results = []
        for item in LABELED_TASKS:
            t0 = time.perf_counter()
            profile = self.metacognition.assess_complexity(item["task"])
            decision = self.metacognition.route(profile)
            latency = (time.perf_counter() - t0) * 1000
            results.append(TaskResult(
                task_id=item["task"][:40],
                passed=(decision.system == item["expected"]),
                system_used=decision.system,
                latency_ms=round(latency, 2),
            ))
        return BenchReport.from_results("routing_accuracy", results)
```

```python
# sage-python/src/sage/bench/__main__.py
"""CLI: python -m sage.bench --type humaneval|swebench|routing|all"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="YGN-SAGE Benchmark Runner")
    parser.add_argument("--type", choices=["humaneval", "swebench", "routing", "all"],
                        default="routing")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    async def run():
        if args.type in ("routing", "all"):
            from sage.strategy.metacognition import MetacognitiveController
            from sage.bench.routing import RoutingAccuracyBench
            mc = MetacognitiveController()
            bench = RoutingAccuracyBench(metacognition=mc)
            report = await bench.run()
            print(f"\nRouting Accuracy: {report.pass_rate:.1%} ({report.passed}/{report.total})")
            print(f"Avg Latency: {report.avg_latency_ms:.1f}ms")
            print(f"Breakdown: {report.routing_breakdown}")

            if args.output:
                out = Path(args.output)
            else:
                out = Path(f"docs/benchmarks/{report.timestamp[:10]}-routing.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            import dataclasses
            out.write_text(json.dumps(dataclasses.asdict(report), indent=2, default=str))
            print(f"Saved to {out}")

    asyncio.run(run())


if __name__ == "__main__":
    main()
```

**Step 4: Run tests**

```bash
cd sage-python && python -m pytest tests/test_bench.py -v
```
Expected: 3 passed

**Step 5: Run the routing benchmark for real**

```bash
cd sage-python && python -m sage.bench --type routing
```
Expected: Output showing accuracy %, latency, breakdown.

**Step 6: Commit**

```bash
git add -A
git commit -m "feat(bench): add benchmark framework + routing accuracy bench (30 labeled tasks)"
```

---

## Task 7: Integration Tests + E2E Tests

**Files:**
- Create: `sage-python/tests/test_integration_v2.py`
- Create: `sage-python/tests/test_e2e_real.py`

**Step 1: Write integration tests (MockProvider, no API key)**

```python
# sage-python/tests/test_integration_v2.py
import sys, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")
    # ... (same mock as test_integration.py)

import pytest
from sage.boot import boot_agent_system
from sage.events.bus import EventBus
from sage.agents.sequential import SequentialAgent
from sage.agents.parallel import ParallelAgent
from sage.agents.handoff import Handoff
from sage.guardrails.builtin import CostGuardrail, SchemaGuardrail
from sage.guardrails.base import GuardrailPipeline
from sage.memory.episodic import EpisodicMemory
from sage.memory.semantic import SemanticMemory
from sage.memory.memory_agent import ExtractionResult


@pytest.mark.asyncio
async def test_eventbus_receives_all_phases():
    bus = EventBus()
    system = boot_agent_system(use_mock_llm=True, event_bus=bus)
    await system.run("What is 2+2?")
    types_seen = {e.type for e in bus.query(last_n=100)}
    assert "PERCEIVE" in types_seen
    assert "THINK" in types_seen

@pytest.mark.asyncio
async def test_routing_decision_emitted():
    bus = EventBus()
    system = boot_agent_system(use_mock_llm=True, event_bus=bus)
    await system.run("Simple question")
    perceive_events = bus.query(phase="PERCEIVE")
    assert len(perceive_events) >= 1

@pytest.mark.asyncio
async def test_episodic_persistence_cross_session(tmp_path):
    db = str(tmp_path / "ep.db")
    mem1 = EpisodicMemory(db_path=db)
    await mem1.initialize()
    await mem1.store("fact1", "The speed of light is 299792458 m/s", {})
    del mem1

    mem2 = EpisodicMemory(db_path=db)
    await mem2.initialize()
    results = await mem2.search("light")
    assert len(results) == 1
    assert "299792458" in results[0]["content"]

def test_semantic_memory_accumulates():
    sem = SemanticMemory()
    sem.add_extraction(ExtractionResult(
        entities=["A", "B"], relationships=[("A", "links", "B")], summary=""
    ))
    sem.add_extraction(ExtractionResult(
        entities=["B", "C"], relationships=[("B", "links", "C")], summary=""
    ))
    assert sem.entity_count() == 3
    assert len(sem.query_entities("B")) == 2

@pytest.mark.asyncio
async def test_guardrail_pipeline_all_pass():
    pipeline = GuardrailPipeline([
        CostGuardrail(max_usd=10.0),
        SchemaGuardrail(required_fields=["answer"]),
    ])
    results = await pipeline.check_all(
        context={"cost_usd": 0.5},
        output='{"answer": "42"}',
    )
    assert all(r.passed for r in results)

@pytest.mark.asyncio
async def test_guardrail_pipeline_cost_blocks():
    pipeline = GuardrailPipeline([
        CostGuardrail(max_usd=0.01),
    ])
    results = await pipeline.check_all(context={"cost_usd": 1.0})
    assert not results[0].passed
    assert pipeline.any_blocked(results)
```

**Step 2: Write E2E tests (require API key, opt-in)**

```python
# sage-python/tests/test_e2e_real.py
"""E2E tests with real LLM. Require GOOGLE_API_KEY. Skip in CI."""
import os
import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.environ.get("GOOGLE_API_KEY"),
        reason="GOOGLE_API_KEY not set"
    ),
]

@pytest.mark.asyncio
async def test_s1_simple_question():
    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=False, llm_tier="fast")
    result = await system.run("What is the capital of France?")
    assert "paris" in result.lower()

@pytest.mark.asyncio
async def test_s2_code_generation():
    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=False, llm_tier="fast")
    result = await system.run("Write a Python function that checks if a number is prime.")
    assert "def " in result
```

**Step 3: Run integration tests**

```bash
cd sage-python && python -m pytest tests/test_integration_v2.py -v
```
Expected: 6 passed

**Step 4: Run all tests**

```bash
cd sage-python && python -m pytest tests/ -v --ignore=tests/test_e2e_real.py
```
Expected: 220+ passed

**Step 5: Commit**

```bash
git add -A
git commit -m "test: add integration tests v2 + E2E real tests (opt-in)"
```

---

## Summary

| Task | Chantier | Est. Complexity | Dependencies |
|------|----------|----------------|--------------|
| 0 | Cleanup | Low | None |
| 1 | EventBus | Medium | Task 0 |
| 2 | Multi-Agent | Medium | Task 0 |
| 3 | Guardrails | Medium | Task 0 |
| 4 | Memory v2 | Medium | Task 0 |
| 5 | Dashboard | High | Tasks 1-4 |
| 6 | Benchmarks | Medium | Tasks 1-4 |
| 7 | Tests | Medium | Tasks 1-6 |

Tasks 2, 3, 4 are parallelizable (no dependencies between them).
Task 5 depends on 1-4 (needs EventBus + all components emitting events).
Task 6 depends on the full system being wired.
Task 7 validates everything.
