# Unified Entry Point Phase 1: Pipeline Calls agent_loop for Bypass

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the pipeline's Stage 4 bypass call `agent_loop.run(task)` instead of `provider.generate()`, giving every single-agent task access to tools, S2/S3 validation, guardrails, and memory. Simplify `system.run()` to always use the pipeline (mock bypass as one-line exception).

**Architecture:** Pipeline orchestrates (routing, topology, model assignment), agent_loop executes (tools, validation, guardrails, memory). Phase 1 wires agent_loop into the pipeline's bypass path (topology=None). The 30-turn tool-calling loop in Stage 4 is replaced by agent_loop's full PERCEIVE->THINK->ACT->LEARN cycle. Mock mode stays as a tested exception to avoid breaking 2001 tests.

**Tech Stack:** Python 3.11, sage-python SDK, Rust sage-core (PyO3), pytest

**Spec:** `docs/superpowers/specs/2026-04-09-unified-entry-point-design.md`

**Hazards addressed:** H1 (double routing), H3 (double tool loop), H4 (triple topology), H9 (mock bypass)

**Codex migration steps covered:** 1 (mock bypass), 2 (_skip_routing), 4 (_current_topology=None), 7 (delete pipeline tool loop)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `sage-python/src/sage/pipeline.py` | Modify | Add `_agent_loop` param; replace Stage 4 bypass tool loop with `agent_loop.run()` |
| `sage-python/src/sage/boot_pipeline.py` | Modify | Pass `agent_loop` to pipeline constructor |
| `sage-python/src/sage/boot.py` | Modify | Simplify `AgentSystem.run()`: mock bypass + pipeline.run() only |
| `sage-python/tests/test_pipeline_bypass.py` | Create | Tests for agent_loop bypass path in pipeline |
| `sage-python/tests/test_execution_path.py` | Modify | Update mock path assertion from "legacy" to "mock" |

---

### Task 1: Wire agent_loop into pipeline constructor

**Files:**
- Modify: `sage-python/src/sage/pipeline.py:85-125` (`__init__`)
- Modify: `sage-python/src/sage/boot_pipeline.py:181-195` (`init_pipeline`)
- Test: `sage-python/tests/test_pipeline_bypass.py` (create)

- [ ] **Step 1: Write test that pipeline accepts agent_loop parameter**

Create `sage-python/tests/test_pipeline_bypass.py`:

```python
"""Tests for pipeline bypass path using agent_loop.run().

Phase 1 of unified entry point: single-agent bypass calls agent_loop
instead of provider.generate(), gaining tools + validation + guardrails.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext


def _make_pipeline(agent_loop=None):
    """Create a minimal pipeline for testing."""
    return CognitiveOrchestrationPipeline(
        router=MagicMock(),
        engine=None,
        assigner=MagicMock(),
        provider_pool=MagicMock(),
        llm_provider=MagicMock(),
        llm_config=MagicMock(),
        agent_loop=agent_loop,
    )


def test_pipeline_accepts_agent_loop_param():
    """Pipeline constructor should accept and store agent_loop."""
    mock_loop = MagicMock()
    pipeline = _make_pipeline(agent_loop=mock_loop)
    assert pipeline._agent_loop is mock_loop


def test_pipeline_agent_loop_defaults_none():
    """Pipeline without agent_loop should default to None."""
    pipeline = _make_pipeline()
    assert pipeline._agent_loop is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_pipeline_bypass.py::test_pipeline_accepts_agent_loop_param -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'agent_loop'`

- [ ] **Step 3: Add agent_loop parameter to pipeline constructor**

In `sage-python/src/sage/pipeline.py`, add `agent_loop` parameter to `__init__`:

```python
    def __init__(
        self,
        router: Any,
        engine: Any,
        assigner: Any,
        provider_pool: Any,
        bandit: Any = None,
        quality_estimator: Any = None,
        event_bus: Any = None,
        llm_provider: Any = None,
        llm_config: Any = None,
        prm: Any = None,
        controller: Any = None,
        smmu: Any = None,
        consolidator: Any = None,
        working_memory: Any = None,
        episodic_memory: Any = None,
        tool_forge: Any = None,
        tool_registry: Any = None,
        harness_config: Any = None,
        agent_loop: Any = None,
    ) -> None:
```

Add at the end of `__init__` body (after `self._task_count = 0`):

```python
        self._agent_loop = agent_loop
```

- [ ] **Step 4: Wire agent_loop from boot into pipeline**

In `sage-python/src/sage/boot_pipeline.py`, modify the `init_pipeline` function signature to accept `agent_loop` parameter:

```python
def init_pipeline(
    router: Any,
    engine: Any,
    provider: Any,
    llm_config: Any,
    bandit: Any,
    rust_registry: Any,
    py_model_registry: Any,
    registry: Any,
    event_bus: Any,
    use_mock_llm: bool,
    consolidator: Any,
    working_memory: Any,
    episodic_memory: Any,
    tool_registry: Any,
    memory_compressor: Any,
    rust_router: Any = None,
    agent_loop: Any = None,
) -> dict[str, Any]:
```

In the pipeline construction call (line ~181), add the `agent_loop` parameter:

```python
            _pipeline = CognitiveOrchestrationPipeline(
                router=router,
                engine=engine,
                assigner=model_assigner,
                provider_pool=_provider_pool,
                bandit=bandit,
                quality_estimator=None,
                event_bus=event_bus,
                llm_provider=provider,
                llm_config=llm_config,
                consolidator=consolidator,
                working_memory=working_memory,
                episodic_memory=episodic_memory,
                tool_registry=tool_registry,
                agent_loop=agent_loop,
            )
```

In `sage-python/src/sage/boot.py`, pass `agent_loop=loop` to `init_pipeline` (line ~668):

```python
    pipe = init_pipeline(
        router=metacognition,
        engine=rust_topology_engine,
        provider=provider,
        llm_config=llm_config,
        bandit=rust_bandit,
        rust_registry=rust_registry,
        py_model_registry=py_model_registry,
        registry=registry,
        event_bus=event_bus,
        use_mock_llm=use_mock_llm,
        consolidator=consolidator,
        working_memory=loop.working_memory,
        episodic_memory=episodic_memory,
        tool_registry=tool_registry,
        memory_compressor=memory_compressor,
        rust_router=rust_router,
        agent_loop=loop,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd sage-python && python -m pytest tests/test_pipeline_bypass.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add sage-python/src/sage/pipeline.py sage-python/src/sage/boot_pipeline.py sage-python/src/sage/boot.py sage-python/tests/test_pipeline_bypass.py
git commit -m "feat: wire agent_loop into pipeline constructor (Phase 1 infra)"
```

---

### Task 2: Pipeline Stage 4 bypass calls agent_loop.run()

**Files:**
- Modify: `sage-python/src/sage/pipeline.py:874-966` (`_stage_execute` bypass section)
- Test: `sage-python/tests/test_pipeline_bypass.py` (add tests)

- [ ] **Step 1: Write test that bypass path calls agent_loop.run()**

Add to `sage-python/tests/test_pipeline_bypass.py`:

```python
@pytest.mark.asyncio
async def test_bypass_calls_agent_loop_run():
    """Stage 4 bypass should call agent_loop.run() instead of provider.generate()."""
    mock_loop = MagicMock()
    mock_loop.run = AsyncMock(return_value="agent_loop result")
    mock_loop.total_cost_usd = 0.001
    mock_loop._skip_routing = False
    mock_loop._current_topology = None
    mock_loop.config = MagicMock()
    mock_loop.config.validation_level = 1
    mock_loop.sandbox_manager = None

    pipeline = _make_pipeline(agent_loop=mock_loop)

    ctx = PipelineContext(task="Write hello world", system=2)
    ctx.topology = None  # bypass mode

    result_ctx = await pipeline._stage_execute(ctx)

    mock_loop.run.assert_called_once_with("Write hello world")
    assert result_ctx.result == "agent_loop result"
    assert result_ctx.cost == 0.001


@pytest.mark.asyncio
async def test_bypass_sets_skip_routing():
    """H1 fix: _skip_routing must be True during agent_loop.run(), restored after."""
    captured_skip = {}

    async def _capture_run(task):
        captured_skip["during"] = mock_loop._skip_routing
        return "result"

    mock_loop = MagicMock()
    mock_loop.run = _capture_run
    mock_loop.total_cost_usd = 0.0
    mock_loop._skip_routing = False
    mock_loop._current_topology = None
    mock_loop.config = MagicMock()
    mock_loop.config.validation_level = 1
    mock_loop.sandbox_manager = None

    pipeline = _make_pipeline(agent_loop=mock_loop)
    ctx = PipelineContext(task="test", system=1)
    ctx.topology = None

    await pipeline._stage_execute(ctx)

    assert captured_skip["during"] is True, "skip_routing must be True during run"
    assert mock_loop._skip_routing is False, "skip_routing must be restored after run"


@pytest.mark.asyncio
async def test_bypass_clears_topology():
    """H4 fix: _current_topology must be None during agent_loop.run()."""
    captured_topo = {}

    async def _capture_run(task):
        captured_topo["during"] = mock_loop._current_topology
        return "result"

    mock_loop = MagicMock()
    mock_loop.run = _capture_run
    mock_loop.total_cost_usd = 0.0
    mock_loop._skip_routing = False
    mock_loop._current_topology = "stale_topology"
    mock_loop.config = MagicMock()
    mock_loop.config.validation_level = 1
    mock_loop.sandbox_manager = None

    pipeline = _make_pipeline(agent_loop=mock_loop)
    ctx = PipelineContext(task="test", system=2)
    ctx.topology = None

    await pipeline._stage_execute(ctx)

    assert captured_topo["during"] is None, "topology must be cleared during run"


@pytest.mark.asyncio
async def test_bypass_sets_validation_level():
    """Validation level should match system classification from routing."""
    mock_loop = MagicMock()
    mock_loop.run = AsyncMock(return_value="result")
    mock_loop.total_cost_usd = 0.0
    mock_loop._skip_routing = False
    mock_loop._current_topology = None
    mock_loop.config = MagicMock()
    mock_loop.config.validation_level = 1
    mock_loop.sandbox_manager = MagicMock()  # sandbox available

    pipeline = _make_pipeline(agent_loop=mock_loop)
    ctx = PipelineContext(task="test", system=3)
    ctx.topology = None

    await pipeline._stage_execute(ctx)

    assert mock_loop.config.validation_level == 3


@pytest.mark.asyncio
async def test_bypass_without_agent_loop_uses_provider_loop():
    """When agent_loop is None, bypass should fall back to provider.generate() loop."""
    mock_provider = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "provider result"
    mock_response.tool_calls = None  # No tool calls = exit loop on first iteration
    mock_provider.generate = AsyncMock(return_value=mock_response)

    pipeline = _make_pipeline(agent_loop=None)
    pipeline.llm_provider = mock_provider

    ctx = PipelineContext(task="simple question", system=1)
    ctx.topology = None

    result_ctx = await pipeline._stage_execute(ctx)

    mock_provider.generate.assert_called_once()
    assert result_ctx.result == "provider result"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd sage-python && python -m pytest tests/test_pipeline_bypass.py -v`
Expected: 5 new tests FAIL (bypass still uses provider.generate, not agent_loop.run)

- [ ] **Step 3: Replace bypass section in _stage_execute**

In `sage-python/src/sage/pipeline.py`, replace the bypass block in `_stage_execute` (lines 874-966).

Find this code (the start of the bypass section):

```python
        # Single-agent mode (no topology or single node)
        if ctx.topology is None or (
            hasattr(ctx.topology, "node_count") and ctx.topology.node_count() <= 1
        ):
            if self.llm_provider:
                from sage.llm.base import Message, Role

                # Bypass mode: use model_id from Rust routing decision if available.
```

Replace the entire bypass block (from `# Single-agent mode` through `return ctx` before `# Multi-agent mode`) with:

```python
        # Single-agent mode (no topology or single node)
        if ctx.topology is None or (
            hasattr(ctx.topology, "node_count") and ctx.topology.node_count() <= 1
        ):
            if self._agent_loop:
                # Phase 1: agent_loop.run() provides tools + S2/S3 validation +
                # guardrails + memory. Replaces the raw provider.generate() loop.

                # H1: Skip routing in agent_loop (pipeline already routed in Stage 0)
                self._agent_loop._skip_routing = True
                # H4: Clear topology (pipeline owns topology, not agent_loop)
                self._agent_loop._current_topology = None

                # Set validation level from system classification
                if ctx.system >= 3:
                    self._agent_loop.config.validation_level = 3
                elif ctx.system >= 2 and self._agent_loop.sandbox_manager:
                    self._agent_loop.config.validation_level = 2
                else:
                    self._agent_loop.config.validation_level = 1

                # Resolve model from Rust routing decision (preserve model selection)
                routing_decision = getattr(self, '_last_routing_decision', None)
                _original_llm = self._agent_loop._llm
                _original_config = self._agent_loop.config.llm
                if routing_decision and routing_decision.model_id and self.provider_pool:
                    try:
                        if self.provider_pool.is_model_available(routing_decision.model_id):
                            resolved_provider, resolved_config = self.provider_pool.resolve(
                                routing_decision.model_id
                            )
                            self._agent_loop._llm = resolved_provider
                            self._agent_loop.config.llm = resolved_config
                            log.info(
                                "Stage 4 bypass: agent_loop using Rust-selected %s (S%d)",
                                routing_decision.model_id, ctx.system,
                            )
                    except Exception:
                        pass  # Keep default provider

                try:
                    ctx.result = await self._agent_loop.run(ctx.task)
                    ctx.cost = self._agent_loop.total_cost_usd
                finally:
                    # Restore agent_loop state (safe for next run)
                    self._agent_loop._skip_routing = False
                    self._agent_loop._llm = _original_llm
                    self._agent_loop.config.llm = _original_config

            elif self.llm_provider:
                # Legacy fallback: direct provider.generate() with tool-calling loop.
                # Kept for backward compat when pipeline is created without agent_loop
                # (e.g., test_pipeline.py). Will be removed in Phase 3.
                from sage.llm.base import Message, Role

                provider = self.llm_provider
                config = self.llm_config
                routing_decision = getattr(self, '_last_routing_decision', None)
                if routing_decision and routing_decision.model_id and self.provider_pool:
                    try:
                        if self.provider_pool.is_model_available(routing_decision.model_id):
                            resolved_provider, resolved_config = self.provider_pool.resolve(
                                routing_decision.model_id
                            )
                            provider = resolved_provider
                            config = resolved_config
                    except Exception:
                        pass

                messages = [Message(role=Role.USER, content=ctx.task)]
                tool_defs = None
                if self.tool_registry and self.tool_registry.list_tools():
                    tool_defs = self.tool_registry.get_tool_defs()

                max_turns = 30
                try:
                    for _turn in range(max_turns):
                        response = await provider.generate(
                            messages=messages, config=config, tools=tool_defs,
                        )
                        if not response.tool_calls:
                            ctx.result = response.content or ""
                            break
                        messages.append(
                            Message(
                                role=Role.ASSISTANT,
                                content=response.content or "",
                                tool_calls=response.tool_calls or None,
                            )
                        )
                        ctx.tool_turn_count += 1
                        for tc in response.tool_calls:
                            ctx.tool_call_count += 1
                            ctx.executed_tools.append(tc.name)
                            tool = self.tool_registry.get(tc.name) if self.tool_registry else None
                            if tool:
                                try:
                                    tool_result = await tool.execute(
                                        tc.arguments if isinstance(tc.arguments, dict) else {}
                                    )
                                    result_text = tool_result.output[:5000]
                                except Exception as te:
                                    result_text = f"Tool error: {te}"
                                cmd = tc.arguments.get("command", "") if isinstance(tc.arguments, dict) else ""
                                if cmd:
                                    ctx.executed_commands.append(cmd)
                            else:
                                result_text = f"Unknown tool: {tc.name}"
                            messages.append(
                                Message(
                                    role=Role.TOOL,
                                    content=result_text,
                                    tool_call_id=tc.id,
                                    name=tc.name,
                                )
                            )
                    else:
                        ctx.result = response.content or ""
                except (RuntimeError, TimeoutError) as exc:
                    ctx.result = f"Error: {exc}"

            return ctx
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd sage-python && python -m pytest tests/test_pipeline_bypass.py -v`
Expected: 7 passed (2 from Task 1 + 5 new)

- [ ] **Step 5: Run existing pipeline tests to verify no regression**

Run: `cd sage-python && python -m pytest tests/test_pipeline.py -v`
Expected: ALL PASS. Tests that create pipeline without `agent_loop` parameter hit the legacy
fallback path (provider.generate() loop), preserving their tool_call_count/tool_turn_count
assertions. This is by design — the fallback is kept for Phase 3 cleanup.

- [ ] **Step 6: Commit**

```bash
git add sage-python/src/sage/pipeline.py sage-python/tests/test_pipeline_bypass.py
git commit -m "feat: pipeline bypass calls agent_loop.run() — tools + validation + guardrails (H1,H3,H4)"
```

---

### Task 3: Simplify system.run() to always use pipeline

**Files:**
- Modify: `sage-python/src/sage/boot.py:106-358` (`AgentSystem.run()`)
- Modify: `sage-python/tests/test_execution_path.py`
- Test: `sage-python/tests/test_pipeline_bypass.py` (add system.run tests)

- [ ] **Step 1: Write test for simplified system.run()**

Add to `sage-python/tests/test_pipeline_bypass.py`:

```python
@pytest.mark.asyncio
async def test_system_run_mock_bypass():
    """Mock mode should bypass pipeline and call agent_loop.run() directly."""
    from sage.boot import boot_agent_system

    system = boot_agent_system(use_mock_llm=True)
    result = await system.run("test task")
    assert system._last_execution_path == "mock"
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_system_run_pipeline_path():
    """Non-mock mode should use pipeline (when pipeline is available)."""
    from sage.boot import boot_agent_system

    system = boot_agent_system(use_mock_llm=True)
    # Simulate non-mock with pipeline available
    system.agent_loop.config.llm.provider = "test"
    system.pipeline = MagicMock()
    system.pipeline.run = AsyncMock(return_value="pipeline result")

    result = await system.run("test task")
    assert system._last_execution_path == "pipeline"
    assert result == "pipeline result"
```

- [ ] **Step 2: Run test to verify test_system_run_mock_bypass fails**

Run: `cd sage-python && python -m pytest tests/test_pipeline_bypass.py::test_system_run_mock_bypass -v`
Expected: FAIL (currently mock path sets `_last_execution_path = "legacy"`, not `"mock"`)

- [ ] **Step 3: Simplify AgentSystem.run()**

In `sage-python/src/sage/boot.py`, replace the entire `run()` method (lines 106-358) with:

```python
    async def run(self, task: str) -> str:
        """Run a task through the agent system.

        Mock mode: direct AgentLoop (tested exception, preserves 2001 tests).
        Non-mock: CognitiveOrchestrationPipeline (5-stage).
        Fallback: direct AgentLoop if pipeline not initialized.
        """
        _budget = self._guardrail_budget if hasattr(self, '_guardrail_budget') else DEFAULT_BUDGET_USD

        # Mock bypass: tested exception (H9).
        # Mock goes direct to agent_loop so phase events (PERCEIVE/THINK/ACT/LEARN)
        # and guardrail wiring are exercised — important for 2001 tests.
        if self.agent_loop.config.llm.provider == "mock":
            self._last_execution_path = "mock"
            self.agent_loop._current_topology = None
            result = await self.agent_loop.run(task)
            return result

        # Pipeline is THE execution path.
        # Pipeline Stage 4 now calls agent_loop.run() for bypass (Phase 1),
        # giving every task tools + S2/S3 validation + guardrails + memory.
        if self.pipeline:
            result = await self.pipeline.run(task, budget_usd=_budget)
            self._last_execution_path = "pipeline"
            await self._persist_memory()
            return result

        # Fallback: pipeline not initialized (missing deps at boot).
        # Direct agent_loop.run() — still gets tools + validation.
        _log.warning("Pipeline not available — using direct agent_loop")
        self._last_execution_path = "direct"
        self.agent_loop._current_topology = None
        result = await self.agent_loop.run(task)
        await self._persist_memory()
        return result
```

This removes the entire legacy routing/topology/model-selection block (~250 lines).
Keep `_record_topology_outcome()` and `_persist_memory()` methods — they are still used by the pipeline path and other callers.

- [ ] **Step 4: Update test_execution_path.py**

In `sage-python/tests/test_execution_path.py`, update the mock path assertion:

```python
@pytest.mark.asyncio
async def test_mock_mode_sets_legacy_path():
    """Mock provider should use mock bypass path."""
    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=True)
    result = await system.run("test task")
    assert system._last_execution_path in ("mock", "legacy", "pipeline", "direct"), (
        f"Expected a valid path, got '{system._last_execution_path}'"
    )
    # Mock mode uses direct agent_loop bypass (H9: don't break 2001 tests)
    assert system._last_execution_path == "mock", (
        f"Mock mode should use mock bypass, got '{system._last_execution_path}'"
    )
```

- [ ] **Step 5: Run tests**

Run: `cd sage-python && python -m pytest tests/test_pipeline_bypass.py tests/test_execution_path.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add sage-python/src/sage/boot.py sage-python/tests/test_execution_path.py sage-python/tests/test_pipeline_bypass.py
git commit -m "feat: system.run() = pipeline.run() — single execution path, mock bypass as H9 exception"
```

---

### Task 4: Remove unused imports and dead code from boot.py run()

**Files:**
- Modify: `sage-python/src/sage/boot.py`

- [ ] **Step 1: Remove unused imports from boot.py**

The simplified `run()` no longer needs these imports at module level or within the method:
- `ModelRouter` import (was used for legacy `get_config(decision.llm_tier)`)
- `GoogleProvider` import (was used for legacy provider switch)
- Any remaining `from sage.llm.router import ModelRouter` in the file

Check if `ModelRouter` is used elsewhere in boot.py. If only used in the deleted run() code, remove the import.

Note: Do NOT delete `_record_topology_outcome()` or `_persist_memory()` — they are still used.

- [ ] **Step 2: Run full boot test suite**

Run: `cd sage-python && python -m pytest tests/test_boot.py tests/test_boot_topology.py tests/test_boot_warnings.py tests/test_boot_refresh.py tests/test_boot_embedder.py -v`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add sage-python/src/sage/boot.py
git commit -m "refactor: remove unused imports from boot.py after system.run() simplification"
```

---

### Task 5: Full test suite verification

**Files:** None (verification only)

- [ ] **Step 1: Run full Python test suite**

```bash
cd sage-python && python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: 1951+ passed, 0 failures.

Watch for failures in:
- `test_pipeline.py` — existing pipeline tests should pass via fallback path (no agent_loop wired)
- `test_execution_path.py` — updated in Task 3
- `test_integration.py` — mock mode tests
- `test_phases.py` — legacy env var test (still valid, `_run_legacy` still exists)

- [ ] **Step 2: Run Rust test suite**

```bash
cd sage-core && cargo test --no-default-features --features smt,tool-executor --lib 2>&1 | tail -10
```

Expected: 403+ passed, 0 failures (no Rust changes in Phase 1).

- [ ] **Step 3: Verify bypass path has agent capabilities**

```bash
cd sage-python && python -c "
from sage.boot import boot_agent_system
system = boot_agent_system(use_mock_llm=True)
pipeline = system.pipeline
if pipeline:
    print(f'Pipeline agent_loop: {pipeline._agent_loop is not None}')
    print(f'Agent tools: {len(pipeline._agent_loop._tools.list_tools())}')
    print(f'Guardrails: {pipeline._agent_loop.guardrail_pipeline is not None}')
else:
    print('Pipeline not initialized (mock mode skips provider discovery)')
print(f'Agent loop tools: {len(system.agent_loop._tools.list_tools())}')
print(f'Sandbox: {system.agent_loop.sandbox_manager is not None}')
print('Phase 1 wiring verified.')
"
```

Expected: Pipeline has agent_loop with tools and guardrails.

- [ ] **Step 4: Run execution path smoke test**

```bash
cd sage-python && python -c "
import asyncio
from sage.boot import boot_agent_system

async def test():
    system = boot_agent_system(use_mock_llm=True)
    result = await system.run('What is 2+2?')
    print(f'Path: {system._last_execution_path}')
    print(f'Result length: {len(result)}')
    assert system._last_execution_path == 'mock', f'Expected mock, got {system._last_execution_path}'
    print('Smoke test passed.')

asyncio.run(test())
"
```

Expected: `Path: mock`, `Smoke test passed.`

---

## Phase 1 Scope Summary

**What changes:**
- Pipeline Stage 4 bypass: `agent_loop.run()` replaces `provider.generate()` + 30-turn tool loop
- `system.run()`: simplified from 250 lines to ~25 lines (mock bypass + pipeline + fallback)
- Routing flags set before agent_loop call (H1: `_skip_routing=True`, H4: `_current_topology=None`)
- Validation level set from pipeline's system classification

**What stays unchanged:**
- `agent_loop.run()` — no changes to the agent loop itself
- `_run_legacy()` — stays for Phase 3 deletion
- `SAGE_AGENT_LOOP_LEGACY` env var — stays for Phase 3
- Pipeline multi-agent path (TopologyRunner) — unchanged, Phase 2 will wire agent_loop per node
- All Rust components — no changes
- 2001 mock tests — mock bypass preserved (H9)
- Existing pipeline tests — fallback to provider.generate() when no agent_loop wired

**What Phase 2 will do:**
- TopologyRunner nodes call `agent_loop.run(node_prompt)` instead of `provider.generate()`
- Per-node AgentLoop factory (fresh state, H2/H8)
- Tools filtered by node role (H6)
- Budget split per node
- Predecessor context injection (H7)

**What Phase 3 will do:**
- Delete `_run_legacy()` + `legacy_think_step` + `run_legacy_s3` + `run_legacy_avr` (~550 lines)
- Delete `SAGE_AGENT_LOOP_LEGACY` env var
- Delete `agent_loop_execution.py` legacy functions
- Audit all tests for legacy path dependencies
