# Unified Entry Point Phase 2: TopologyRunner Nodes = agent_loop

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every LLM topology node execute via `agent_loop.run()`, giving multi-agent nodes access to tools, S2/S3 validation, guardrails, and memory. Code/solver nodes stay as-is.

**Architecture:** A factory function creates independent AgentLoop instances per node with role-filtered tools and per-node validation. TopologyRunner dispatches to agent_loop for LLM nodes, keeping existing code/solver paths. Predecessor context injected in the user message (H7). Parallel nodes get independent AgentLoops (H8).

**Tech Stack:** Python 3.13, sage-python SDK, pytest

**Spec:** `docs/superpowers/specs/2026-04-09-unified-entry-point-design.md`

**Hazards addressed:** H2 (state reset), H6 (recursive validation), H7 (predecessor context), H8 (async concurrency)

**Codex migration steps covered:** 3 (AgentLoop factory), 4 (_current_topology=None), 5 (predecessor context), 6 (budget split)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `sage-python/src/sage/agent_loop_factory.py` | Create | Factory function: creates per-node AgentLoop with tool filtering, validation, skip flags |
| `sage-python/src/sage/topology/runner.py` | Modify | Accept factory; `_execute_node_via_agent_loop()` for LLM nodes; keep code/solver as-is |
| `sage-python/src/sage/pipeline.py` | Modify | Create factory in Stage 4 multi-agent, pass to TopologyRunner |
| `sage-python/tests/test_agent_loop_factory.py` | Create | Tests for factory: tool filtering, validation levels, skip flags |
| `sage-python/tests/test_runner_agent_loop.py` | Create | Tests for runner using agent_loop per node |

---

### Task 1: Create agent_loop_factory.py

**Files:**
- Create: `sage-python/src/sage/agent_loop_factory.py`
- Create: `sage-python/tests/test_agent_loop_factory.py`

- [ ] **Step 1: Write tests for the factory**

Create `sage-python/tests/test_agent_loop_factory.py`:

```python
"""Tests for per-node AgentLoop factory.

Phase 2: each topology node gets an independent AgentLoop with
role-filtered tools and appropriate validation level.
"""
import pytest
from unittest.mock import MagicMock

from sage.agent_loop_factory import create_node_agent_loop


def _make_tool_registry():
    """Create a mock registry with known tools."""
    registry = MagicMock()
    registry.list_tools.return_value = [
        MagicMock(name="execute_bash"),
        MagicMock(name="create_python_tool"),
        MagicMock(name="stm_read"),
        MagicMock(name="stm_write"),
        MagicMock(name="ltm_recall"),
    ]
    return registry


def test_actor_gets_all_tools():
    """Actor nodes should have access to all tools."""
    loop = create_node_agent_loop(
        node_role="actor",
        node_name="node-0-actor",
        llm_provider=MagicMock(),
        llm_config=MagicMock(),
        tool_registry=_make_tool_registry(),
        system_prompt="You are an actor.",
        system_level=2,
    )
    assert loop.config.tools is None  # None = all tools


def test_verifier_gets_limited_tools():
    """Verifier nodes should only get execute_bash + memory tools (H6)."""
    loop = create_node_agent_loop(
        node_role="verifier",
        node_name="node-1-verifier",
        llm_provider=MagicMock(),
        llm_config=MagicMock(),
        tool_registry=_make_tool_registry(),
        system_prompt="You are a verifier.",
        system_level=2,
    )
    assert loop.config.tools is not None
    assert "execute_bash" in loop.config.tools
    assert "create_python_tool" not in loop.config.tools


def test_verifier_validation_level_zero():
    """H6: verifier nodes must have validation_level=0 to avoid recursive AVR."""
    loop = create_node_agent_loop(
        node_role="verifier",
        node_name="node-1-verifier",
        llm_provider=MagicMock(),
        llm_config=MagicMock(),
        tool_registry=_make_tool_registry(),
        system_prompt="You are a verifier.",
        system_level=3,  # even with S3, verifier gets 0
    )
    assert loop.config.validation_level == 0


def test_actor_s3_gets_validation_3():
    """Actor nodes in S3 system should get full Z3 validation."""
    loop = create_node_agent_loop(
        node_role="actor",
        node_name="node-0-actor",
        llm_provider=MagicMock(),
        llm_config=MagicMock(),
        tool_registry=_make_tool_registry(),
        system_prompt="You are an actor.",
        system_level=3,
    )
    assert loop.config.validation_level == 3


def test_actor_s2_gets_validation_2():
    """Actor nodes in S2 system should get AVR validation."""
    loop = create_node_agent_loop(
        node_role="coder",
        node_name="node-0-coder",
        llm_provider=MagicMock(),
        llm_config=MagicMock(),
        tool_registry=_make_tool_registry(),
        system_prompt="You are a coder.",
        system_level=2,
    )
    assert loop.config.validation_level == 2


def test_skip_routing_set():
    """H1 carryover: _skip_routing must be True (pipeline already routed)."""
    loop = create_node_agent_loop(
        node_role="actor",
        node_name="node-0",
        llm_provider=MagicMock(),
        llm_config=MagicMock(),
        tool_registry=_make_tool_registry(),
        system_prompt="prompt",
        system_level=1,
    )
    assert loop._skip_routing is True


def test_topology_cleared():
    """H4 carryover: _current_topology must be None."""
    loop = create_node_agent_loop(
        node_role="actor",
        node_name="node-0",
        llm_provider=MagicMock(),
        llm_config=MagicMock(),
        tool_registry=_make_tool_registry(),
        system_prompt="prompt",
        system_level=1,
    )
    assert loop._current_topology is None


def test_independent_instances():
    """H8: two factory calls must produce independent instances (no shared state)."""
    registry = _make_tool_registry()
    provider = MagicMock()

    loop_a = create_node_agent_loop(
        node_role="actor", node_name="a",
        llm_provider=provider, llm_config=MagicMock(),
        tool_registry=registry, system_prompt="A", system_level=2,
    )
    loop_b = create_node_agent_loop(
        node_role="actor", node_name="b",
        llm_provider=provider, llm_config=MagicMock(),
        tool_registry=registry, system_prompt="B", system_level=2,
    )

    assert loop_a is not loop_b
    assert loop_a.working_memory is not loop_b.working_memory
    assert loop_a.config.name != loop_b.config.name


def test_output_formatter_minimal_tools():
    """Output formatter nodes get memory tools only (no code execution)."""
    loop = create_node_agent_loop(
        node_role="output_formatter",
        node_name="node-2-formatter",
        llm_provider=MagicMock(),
        llm_config=MagicMock(),
        tool_registry=_make_tool_registry(),
        system_prompt="Format the output.",
        system_level=1,
    )
    assert loop.config.tools is not None
    assert "execute_bash" not in loop.config.tools
    assert "create_python_tool" not in loop.config.tools


def test_max_steps_bounded():
    """Per-node loops should have bounded max_steps (lighter than standalone)."""
    loop = create_node_agent_loop(
        node_role="actor",
        node_name="node-0",
        llm_provider=MagicMock(),
        llm_config=MagicMock(),
        tool_registry=_make_tool_registry(),
        system_prompt="prompt",
        system_level=1,
    )
    assert loop.config.max_steps <= 30
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd sage-python && python -m pytest tests/test_agent_loop_factory.py -v`
Expected: FAIL (module does not exist)

- [ ] **Step 3: Implement the factory**

Create `sage-python/src/sage/agent_loop_factory.py`:

```python
"""Per-node AgentLoop factory for topology execution.

Phase 2 of unified entry point: each topology node gets an independent
AgentLoop with role-filtered tools and per-node validation.

Hazards addressed:
- H6: Verifier nodes run with validation_level=0 (no recursive AVR/Z3)
- H8: Each call creates a fresh instance (no shared mutable state)
"""
from __future__ import annotations

from typing import Any

from sage.agent import AgentConfig
from sage.agent_loop import AgentLoop
from sage.llm.base import LLMConfig, LLMProvider
from sage.tools.registry import ToolRegistry

# Tool sets per role (H6: prevent recursive validation on verifiers)
_VERIFIER_TOOLS = ["execute_bash", "stm_read", "stm_write", "ltm_recall"]
_FORMATTER_TOOLS = ["stm_read", "stm_write", "ltm_recall"]

# Roles that get restricted validation (H6)
_NO_VALIDATION_ROLES = {"verifier", "output_formatter", "formatter", "aggregator", "critic"}


def create_node_agent_loop(
    node_role: str,
    node_name: str,
    llm_provider: LLMProvider,
    llm_config: LLMConfig,
    tool_registry: ToolRegistry,
    system_prompt: str,
    system_level: int,
    on_event: Any = None,
) -> AgentLoop:
    """Create an independent AgentLoop for a topology node.

    Each call returns a FRESH instance with its own WorkingMemory,
    CircuitBreakers, and DriftMonitor (H8: no shared mutable state).

    Tool filtering (H6):
    - actor/coder/planner: all tools (config.tools = None)
    - verifier: execute_bash + memory (can run tests, no code gen)
    - output_formatter/aggregator: memory only (no code execution)

    Validation (H6):
    - actor/coder: full validation from system_level
    - verifier/formatter/aggregator: validation_level=0 (no AVR/Z3)
    """
    role_lower = node_role.lower()

    # Tool filtering
    tools: list[str] | None = None  # all tools for actors
    if any(r in role_lower for r in ("verif",)):
        tools = _VERIFIER_TOOLS
    elif any(r in role_lower for r in ("format", "output", "aggregat")):
        tools = _FORMATTER_TOOLS

    # Validation level (H6: no validation on verifiers to prevent recursion)
    if any(r in role_lower for r in _NO_VALIDATION_ROLES):
        validation = 0
    elif system_level >= 3:
        validation = 3
    elif system_level >= 2:
        validation = 2
    else:
        validation = 1

    config = AgentConfig(
        name=node_name,
        llm=llm_config,
        system_prompt=system_prompt,
        max_steps=20,  # nodes are lighter than standalone runs
        validation_level=validation,
        tools=tools,
    )

    loop = AgentLoop(
        config=config,
        llm_provider=llm_provider,
        tool_registry=tool_registry,
        on_event=on_event,
    )

    # H1/H4 carryover: pipeline already handled routing and topology
    loop._skip_routing = True
    loop._current_topology = None

    return loop
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd sage-python && python -m pytest tests/test_agent_loop_factory.py -v`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
cd /c/Code/YGN-SAGE
git add sage-python/src/sage/agent_loop_factory.py sage-python/tests/test_agent_loop_factory.py
git commit -m "feat: agent_loop factory for per-node topology instances (H6,H8)"
```

---

### Task 2: TopologyRunner dispatches LLM nodes to agent_loop

**Files:**
- Modify: `sage-python/src/sage/topology/runner.py:50-63,499-676`
- Create: `sage-python/tests/test_runner_agent_loop.py`

- [ ] **Step 1: Write tests for runner agent_loop dispatch**

Create `sage-python/tests/test_runner_agent_loop.py`:

```python
"""Tests for TopologyRunner dispatching LLM nodes to agent_loop.

Phase 2: each LLM topology node calls agent_loop.run() instead of
provider.generate(), gaining tools + validation + guardrails.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _make_graph(n_nodes=2):
    """Create a mock topology graph."""
    graph = MagicMock()
    graph.node_count.return_value = n_nodes
    nodes = []
    for i in range(n_nodes):
        node = MagicMock()
        node.role = "actor" if i == 0 else "verifier"
        node.model_id = ""
        node.prompt = ""
        node.node_type = "llm"
        node.required_capabilities = []
        nodes.append(node)
    graph.get_node = lambda idx: nodes[idx]
    graph.get_predecessors = lambda idx: list(range(idx))
    return graph


def _make_executor(ready_sequence):
    """Create a mock executor that yields nodes in sequence."""
    executor = MagicMock()
    call_count = [0]
    def _next_ready(graph):
        if call_count[0] < len(ready_sequence):
            result = ready_sequence[call_count[0]]
            call_count[0] += 1
            return result
        return []
    executor.next_ready = _next_ready
    executor.is_done = lambda: call_count[0] >= len(ready_sequence)
    return executor


@pytest.mark.asyncio
async def test_llm_node_uses_agent_loop_when_factory_set():
    """LLM nodes should call agent_loop.run() when factory is provided."""
    mock_loop = MagicMock()
    mock_loop.run = AsyncMock(return_value="agent result")
    mock_loop.total_cost_usd = 0.0

    factory = MagicMock(return_value=mock_loop)

    from sage.topology.runner import TopologyRunner

    graph = _make_graph(1)
    executor = _make_executor([[0]])

    runner = TopologyRunner(
        graph=graph,
        executor=executor,
        llm_provider=MagicMock(),
        agent_loop_factory=factory,
    )

    result = await runner.run("test task")

    factory.assert_called_once()
    mock_loop.run.assert_called_once()
    assert "agent result" in result


@pytest.mark.asyncio
async def test_code_node_skips_agent_loop():
    """Code nodes should NOT use agent_loop, even when factory is set."""
    factory = MagicMock()

    from sage.topology.runner import TopologyRunner

    graph = _make_graph(1)
    node = graph.get_node(0)
    node.node_type = "code"
    node.code_spec = "print('hello')"
    executor = _make_executor([[0]])

    runner = TopologyRunner(
        graph=graph,
        executor=executor,
        llm_provider=MagicMock(),
        agent_loop_factory=factory,
    )

    with patch.object(runner, '_execute_code_node', new_callable=AsyncMock, return_value="hello"):
        await runner.run("task")

    factory.assert_not_called()


@pytest.mark.asyncio
async def test_predecessor_context_in_task():
    """H7: predecessor output should be injected in the task passed to agent_loop."""
    captured_task = {}

    async def _capture_run(task):
        captured_task["value"] = task
        return "result"

    mock_loop = MagicMock()
    mock_loop.run = _capture_run
    mock_loop.total_cost_usd = 0.0

    factory = MagicMock(return_value=mock_loop)

    from sage.topology.runner import TopologyRunner

    graph = _make_graph(2)
    # Sequential: node 0 then node 1
    executor = _make_executor([[0], [1]])

    runner = TopologyRunner(
        graph=graph,
        executor=executor,
        llm_provider=MagicMock(),
        agent_loop_factory=factory,
    )

    # Pre-set node 0 output (simulating first node completed)
    async def _first_run(task):
        runner._node_outputs[0] = "first node output"
        return "first node output"
    # Factory returns different loops for each call
    loops = []
    for i in range(2):
        loop = MagicMock()
        loop.total_cost_usd = 0.0
        if i == 0:
            loop.run = _first_run
        else:
            loop.run = _capture_run
        loops.append(loop)
    factory.side_effect = loops

    await runner.run("original task")

    assert "first node output" in captured_task["value"]
    assert "original task" in captured_task["value"]


@pytest.mark.asyncio
async def test_no_factory_uses_provider_directly():
    """Without factory, runner should use existing provider.generate() path."""
    mock_provider = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "provider result"
    mock_provider.generate = AsyncMock(return_value=mock_response)

    from sage.topology.runner import TopologyRunner

    graph = _make_graph(1)
    executor = _make_executor([[0]])

    runner = TopologyRunner(
        graph=graph,
        executor=executor,
        llm_provider=mock_provider,
        # No agent_loop_factory
    )

    result = await runner.run("task")
    mock_provider.generate.assert_called()
    assert "provider result" in result


@pytest.mark.asyncio
async def test_factory_receives_node_role():
    """Factory should receive the node role for tool filtering."""
    factory = MagicMock()
    mock_loop = MagicMock()
    mock_loop.run = AsyncMock(return_value="result")
    mock_loop.total_cost_usd = 0.0
    factory.return_value = mock_loop

    from sage.topology.runner import TopologyRunner

    graph = _make_graph(1)
    graph.get_node(0).role = "verifier"
    executor = _make_executor([[0]])

    runner = TopologyRunner(
        graph=graph,
        executor=executor,
        llm_provider=MagicMock(),
        agent_loop_factory=factory,
    )

    await runner.run("task")

    call_kwargs = factory.call_args
    assert call_kwargs.kwargs.get("node_role") == "verifier" or \
           (call_kwargs.args and call_kwargs.args[0] == "verifier") or \
           any("verifier" in str(v) for v in call_kwargs.kwargs.values())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd sage-python && python -m pytest tests/test_runner_agent_loop.py -v`
Expected: FAIL (TopologyRunner doesn't accept agent_loop_factory yet)

- [ ] **Step 3: Add agent_loop_factory to TopologyRunner constructor**

In `sage-python/src/sage/topology/runner.py`, add `agent_loop_factory` parameter to `__init__`:

After `harness_config: Any | None = None,` add:

```python
        agent_loop_factory: Any | None = None,
```

And in the constructor body, after `self._harness = harness_config`, add:

```python
        self._agent_loop_factory = agent_loop_factory
```

- [ ] **Step 4: Add _execute_node_via_agent_loop method**

Add this method to TopologyRunner (before `_execute_code_node`):

```python
    async def _execute_node_via_agent_loop(
        self, node_idx: int, task: str, context_override: str | None = None,
    ) -> str:
        """Execute an LLM node via per-node AgentLoop (Phase 2).

        Creates an independent AgentLoop instance for this node with:
        - Role-filtered tools (H6)
        - Validation level from system classification (H6)
        - Skip routing (H1) and topology (H4) flags
        - Predecessor context in user message (H7)
        """
        node = self.graph.get_node(node_idx)
        role = getattr(node, "role", f"node-{node_idx}")
        caps = getattr(node, "required_capabilities", [])

        # Build system prompt (same logic as _execute_node)
        custom_prompt = getattr(node, "prompt", "")
        if custom_prompt:
            system_prompt = custom_prompt
        else:
            _default_tmpl = (
                self._harness.prompts.default_template if self._harness
                else "You are acting as: {role}."
            )
            system_prompt = _default_tmpl.format(
                role=role, capabilities=", ".join(caps) if caps else "",
                task_preview=task[:200], n_predecessors=0,
            )
            if caps:
                _cap_tmpl = (
                    self._harness.prompts.capability_template if self._harness
                    else " Your capabilities: {capabilities}."
                )
                system_prompt += _cap_tmpl.format(capabilities=", ".join(caps))

        if self._harness:
            if self._harness.prompts.global_prefix:
                system_prompt = self._harness.prompts.global_prefix + "\n" + system_prompt
            if self._harness.prompts.global_suffix:
                system_prompt = system_prompt + "\n" + self._harness.prompts.global_suffix

        # Resolve per-node model
        node_model_id = getattr(node, "model_id", "")
        if node_model_id and self._provider_pool:
            provider, config = self._provider_pool.resolve(node_model_id)
        else:
            provider, config = self._llm, self._config

        # Create per-node AgentLoop (H8: independent instance)
        loop = self._agent_loop_factory(
            node_role=role,
            node_name=f"node-{node_idx}-{role}",
            llm_provider=provider,
            llm_config=config,
            system_prompt=system_prompt,
        )

        # Build task with predecessor context (H7)
        context = (
            context_override
            if context_override is not None
            else self._gather_predecessor_context(node_idx)
        )
        if context:
            full_task = (
                f"## Previous agent output:\n{context}\n\n"
                f"## Task:\n{task}"
            )
        else:
            full_task = task

        # Execute
        result = await loop.run(full_task)
        self._node_outputs[node_idx] = result
        return result
```

- [ ] **Step 5: Dispatch LLM nodes to agent_loop in _execute_node**

In `_execute_node()` (line ~499), after the code and solver dispatches but BEFORE the existing LLM path, add:

```python
        # Phase 2: LLM nodes use agent_loop when factory available
        if self._agent_loop_factory:
            return await self._execute_node_via_agent_loop(node_idx, task, context_override)
```

This goes after line ~523 (after `if node_type == "solver"...` block) and before `role = getattr(node, "role", ...)`.

- [ ] **Step 6: Run tests**

Run: `cd sage-python && python -m pytest tests/test_runner_agent_loop.py tests/test_agent_loop_factory.py -v`
Expected: ALL PASS

Run: `cd sage-python && python -m pytest tests/test_topology_runner.py -v` (if exists)
Expected: ALL PASS (existing tests don't pass factory, so they use legacy path)

- [ ] **Step 7: Commit**

```bash
cd /c/Code/YGN-SAGE
git add sage-python/src/sage/topology/runner.py sage-python/tests/test_runner_agent_loop.py
git commit -m "feat: TopologyRunner dispatches LLM nodes to agent_loop (H6,H7,H8)"
```

---

### Task 3: Pipeline Stage 4 multi-agent creates and passes factory

**Files:**
- Modify: `sage-python/src/sage/pipeline.py:1017-1040` (multi-agent section)
- Test: `sage-python/tests/test_pipeline_bypass.py` (add multi-agent test)

- [ ] **Step 1: Write test for factory creation in pipeline**

Add to `sage-python/tests/test_pipeline_bypass.py`:

```python
@pytest.mark.asyncio
async def test_multi_agent_creates_factory():
    """Pipeline Stage 4 multi-agent should create agent_loop factory for TopologyRunner."""
    from sage.pipeline import CognitiveOrchestrationPipeline, PipelineContext

    mock_loop = MagicMock()
    mock_loop.run = AsyncMock(return_value="agent_loop result")
    mock_loop.total_cost_usd = 0.0

    pipeline = _make_pipeline(agent_loop=MagicMock())
    pipeline.tool_registry = MagicMock()
    pipeline.event_bus = MagicMock()

    # Create a mock topology with 2 nodes
    mock_topo = MagicMock()
    mock_topo.node_count.return_value = 2

    ctx = PipelineContext(task="complex task", system=2)
    ctx.topology = mock_topo

    # Mock the TopologyRunner and TopologyExecutor
    with patch('sage.pipeline.TopologyRunner') as MockRunner, \
         patch('sage.pipeline.TopologyExecutor') as MockExecutor:
        mock_runner_inst = MagicMock()
        mock_runner_inst.run = AsyncMock(return_value="multi-agent result")
        MockRunner.return_value = mock_runner_inst

        result_ctx = await pipeline._stage_execute(ctx)

        # Verify TopologyRunner was created with agent_loop_factory
        call_kwargs = MockRunner.call_args
        assert 'agent_loop_factory' in call_kwargs.kwargs or \
               len(call_kwargs.args) > 7  # factory passed as positional
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_pipeline_bypass.py::test_multi_agent_creates_factory -v`
Expected: FAIL (pipeline doesn't pass factory to runner yet)

- [ ] **Step 3: Create factory and pass to TopologyRunner in pipeline**

In `sage-python/src/sage/pipeline.py`, in the multi-agent section of `_stage_execute()`, find the TopologyRunner construction (around line 1031):

```python
            runner = TopologyRunner(
                graph=ctx.topology,
                executor=executor,
                llm_provider=self.llm_provider,
                llm_config=self.llm_config,
                provider_pool=self.provider_pool,
                controller=self.controller,  # Phase C
                axis_hint=ctx.axis_hint,
            )
```

Replace with:

```python
            # Phase 2: create agent_loop factory for per-node execution
            _agent_loop_factory = None
            if self._agent_loop and self.tool_registry:
                from sage.agent_loop_factory import create_node_agent_loop
                from functools import partial

                _agent_loop_factory = partial(
                    create_node_agent_loop,
                    tool_registry=self.tool_registry,
                    system_level=ctx.system,
                    on_event=(
                        self.event_bus.emit
                        if self.event_bus and hasattr(self.event_bus, "emit")
                        else None
                    ),
                )

            runner = TopologyRunner(
                graph=ctx.topology,
                executor=executor,
                llm_provider=self.llm_provider,
                llm_config=self.llm_config,
                provider_pool=self.provider_pool,
                controller=self.controller,  # Phase C
                axis_hint=ctx.axis_hint,
                agent_loop_factory=_agent_loop_factory,
            )
```

Also add the same factory to the reroute runner (~line 1072):

```python
                runner2 = TopologyRunner(
                    graph=ctx.topology, executor=executor_rerouted,
                    llm_provider=self.llm_provider, llm_config=self.llm_config,
                    provider_pool=self.provider_pool,
                    controller=None,  # no controller on retry to prevent loop
                    agent_loop_factory=_agent_loop_factory,
                )
```

And the FrugalGPT cascade retry runner (if it exists further down).

- [ ] **Step 4: Run tests**

Run: `cd sage-python && python -m pytest tests/test_pipeline_bypass.py tests/test_pipeline.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
cd /c/Code/YGN-SAGE
git add sage-python/src/sage/pipeline.py sage-python/tests/test_pipeline_bypass.py
git commit -m "feat: pipeline creates agent_loop factory for multi-agent topology nodes"
```

---

### Task 4: Full test suite verification

**Files:** None (verification only)

- [ ] **Step 1: Run full Python test suite**

```bash
cd sage-python && python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: 2070+ passed, 0 new failures.

- [ ] **Step 2: Run Rust tests**

```bash
cd sage-core && cargo test --no-default-features --features smt,tool-executor --lib 2>&1 | tail -5
```

Expected: 429+ passed, 0 failures.

- [ ] **Step 3: Verify factory wiring at boot**

```bash
cd sage-python && python -c "
from sage.agent_loop_factory import create_node_agent_loop
from unittest.mock import MagicMock
loop = create_node_agent_loop(
    node_role='actor', node_name='test',
    llm_provider=MagicMock(), llm_config=MagicMock(),
    tool_registry=MagicMock(), system_prompt='test',
    system_level=2,
)
print(f'skip_routing={loop._skip_routing}')
print(f'topology={loop._current_topology}')
print(f'validation={loop.config.validation_level}')
print(f'tools={loop.config.tools}')
print(f'max_steps={loop.config.max_steps}')
print('Factory wiring verified.')
"
```

Expected: skip_routing=True, topology=None, validation=2, tools=None, max_steps=20

---

## Phase 2 Scope Summary

**What changes:**
- New `agent_loop_factory.py`: creates independent AgentLoop per topology node
- TopologyRunner: LLM nodes dispatch to agent_loop via factory (code/solver nodes unchanged)
- Pipeline Stage 4 multi-agent: creates factory from boot dependencies, passes to runner

**What stays unchanged:**
- Code node execution (sandbox)
- Solver node execution (Rust Z3 + LLM fallback)
- TopologyRunner controller interactions (upgrade_model, spawn_subagent, etc.)
- Legacy _execute_node() path (kept when no factory provided)
- All Phase 1 changes

**What Phase 3 will do:**
- Delete `_run_legacy()` + legacy functions (~550 lines)
- Delete `SAGE_AGENT_LOOP_LEGACY` env var
- Delete legacy tool-calling loop from pipeline (the `elif self.llm_provider:` fallback)
- Audit all tests for legacy path dependencies
