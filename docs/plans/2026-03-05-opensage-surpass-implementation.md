# YGN-SAGE: Surpass OpenSAGE — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire the 5 cognitive pillars into a live agent runtime, connect real LLMs, add self-improving evolutionary topology search, and ship a production dashboard — surpassing OpenSAGE.

**Architecture:** Agent Runtime Loop orchestrates LLM providers (Codex 5.3 primary, Gemini 3.x fallback) through a perceive→think→act→learn cycle. Sub-agents are spawned dynamically via topology planner. Evolved code passes Z3 safety gate. Memory persists across sessions via Neo4j/Qdrant. A React+Tailwind dashboard provides real-time control.

**Tech Stack:** Python 3.13, Rust (PyO3), Codex `gpt-5.3-codex`, Gemini `gemini-3-flash-preview` / `gemini-3.1-pro-preview`, Claude `claude-sonnet-4-6`, z3-solver, Neo4j, Qdrant, FastAPI, React 18, Tailwind CSS v4, WebSocket

---

## LLM Model Matrix (March 2026 SOTA)

| Role | Model ID | Provider | Cost (in/out per 1M) | When |
|------|----------|----------|----------------------|------|
| **Mutation (volume)** | `gemini-3-flash-preview` | Google | $0.50 / $3.00 | Code generation, SEARCH/REPLACE diffs |
| **Reasoning (quality)** | `gemini-3.1-pro-preview` | Google | $2.00 / $12.00 | Complex evaluation, fitness scoring |
| **Orchestration** | `gpt-5.3-codex` | OpenAI | $1.75 / $14.00 | Codex exec agent tasks |
| **Budget (bulk)** | `gemini-2.5-flash-lite` | Google | $0.10 / $0.40 | High-volume simple transforms |
| **Fallback** | `gemini-2.5-flash` | Google | $0.30 / $2.50 | If 3.x unavailable |

**Structured Output:** All providers support JSON schema enforcement:
- Gemini: `response_mime_type='application/json'` + `response_json_schema=PydanticModel.model_json_schema()`
- Codex: `codex exec --output-schema schema.json`
- Claude: `client.messages.parse(output_format=PydanticModel)`

---

## Task 0: Upgrade LLM Providers (Codex 5.3 + Gemini 3.x + Structured Output)

**Files:**
- Modify: `sage-python/src/sage/llm/codex.py`
- Modify: `sage-python/src/sage/llm/google.py`
- Modify: `sage-python/src/sage/llm/router.py`
- Modify: `sage-python/src/sage/llm/base.py`
- Create: `sage-python/tests/test_llm_providers.py`

**Step 1: Write failing tests**

```python
# sage-python/tests/test_llm_providers.py
import sys, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest
from pydantic import BaseModel
from sage.llm.base import LLMConfig, Message, Role, LLMResponse

class MutationOutput(BaseModel):
    search: str
    replace: str
    description: str
    features: list[int]

def test_llm_config_has_json_schema():
    cfg = LLMConfig(provider="google", model="gemini-3-flash-preview")
    assert hasattr(cfg, 'json_schema')

def test_model_router_tiers():
    from sage.llm.router import ModelRouter
    fast = ModelRouter.get_config("fast")
    assert "flash" in fast.model.lower()

    mutator = ModelRouter.get_config("mutator")
    assert "flash" in mutator.model.lower()

    reasoner = ModelRouter.get_config("reasoner")
    assert "pro" in reasoner.model.lower()

    codex = ModelRouter.get_config("codex")
    assert codex.provider == "codex"

def test_llm_response_has_structured():
    resp = LLMResponse(content='{"search":"a","replace":"b"}', model="test")
    assert resp.content is not None
```

**Step 2: Run tests to verify they fail**

Run: `cd sage-python && python -m pytest tests/test_llm_providers.py -v`
Expected: FAIL (json_schema not on LLMConfig, "mutator"/"codex" tiers missing)

**Step 3: Update LLMConfig with json_schema support**

In `sage-python/src/sage/llm/base.py`, add to `LLMConfig`:
```python
@dataclass
class LLMConfig:
    provider: str
    model: str
    max_tokens: int = 8192
    temperature: float = 0.0
    top_p: float = 1.0
    api_key: str | None = None
    base_url: str | None = None
    json_schema: type | dict | None = None  # Pydantic model or raw dict for structured output
    extra: dict[str, Any] = field(default_factory=dict)
```

**Step 4: Update ModelRouter with all tiers**

```python
# sage-python/src/sage/llm/router.py
from typing import Literal
from sage.llm.base import LLMConfig

class ModelRouter:
    """SOTA March 2026 Model Router for YGN-SAGE."""

    MODELS = {
        "fast": "gemini-3.1-flash-lite-preview",
        "mutator": "gemini-3-flash-preview",
        "reasoner": "gemini-3.1-pro-preview",
        "codex": "gpt-5.3-codex",
        "budget": "gemini-2.5-flash-lite",
        "fallback": "gemini-2.5-flash",
    }

    @staticmethod
    def get_config(
        tier: Literal["fast", "mutator", "reasoner", "codex", "budget", "critical"],
        temperature: float = 0.7,
        json_schema: type | dict | None = None,
    ) -> LLMConfig:
        if tier == "codex":
            return LLMConfig(
                provider="codex", model=ModelRouter.MODELS["codex"],
                max_tokens=8192, temperature=temperature, json_schema=json_schema,
            )
        elif tier in ("reasoner", "critical"):
            return LLMConfig(
                provider="google", model=ModelRouter.MODELS["reasoner"],
                max_tokens=8192, temperature=temperature, json_schema=json_schema,
            )
        elif tier == "mutator":
            return LLMConfig(
                provider="google", model=ModelRouter.MODELS["mutator"],
                max_tokens=4096, temperature=temperature, json_schema=json_schema,
            )
        elif tier == "budget":
            return LLMConfig(
                provider="google", model=ModelRouter.MODELS["budget"],
                max_tokens=2048, temperature=temperature, json_schema=json_schema,
            )
        else:  # fast
            return LLMConfig(
                provider="google", model=ModelRouter.MODELS["fast"],
                max_tokens=4096, temperature=temperature, json_schema=json_schema,
            )
```

**Step 5: Update GoogleProvider with structured output**

In `sage-python/src/sage/llm/google.py`, add to `generate()`:
```python
# Inside generate(), before creating generate_config:
response_schema_config = {}
if config and config.json_schema:
    response_schema_config = {
        'response_mime_type': 'application/json',
    }
    schema = config.json_schema
    if hasattr(schema, 'model_json_schema'):
        response_schema_config['response_json_schema'] = schema.model_json_schema()
    elif isinstance(schema, dict):
        response_schema_config['response_json_schema'] = schema

generate_config = types.GenerateContentConfig(
    max_output_tokens=config.max_tokens if config else None,
    temperature=config.temperature if config else 0.1,
    system_instruction=system_instruction,
    tools=gemini_tools if gemini_tools else None,
    **response_schema_config,
)
```

**Step 6: Update CodexProvider for 5.3 + structured output**

Replace the `CodexProvider` class in `sage-python/src/sage/llm/codex.py`:
```python
class CodexProvider:
    """LLMProvider using OpenAI Codex 5.3 CLI with Gemini fallback."""

    name = "codex"

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        prompt_parts = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == Role.USER:
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == Role.ASSISTANT:
                prompt_parts.append(f"Assistant: {msg.content}")
        full_prompt = "\n".join(prompt_parts)

        codex_path = shutil.which("codex")
        if codex_path:
            model = config.model if config else "gpt-5.3-codex"
            cmd = [codex_path, "exec", full_prompt, "--json"]

            # Structured output via --output-schema
            schema_file = None
            if config and config.json_schema:
                import tempfile
                schema = config.json_schema
                if hasattr(schema, 'model_json_schema'):
                    schema = schema.model_json_schema()
                schema_file = tempfile.NamedTemporaryFile(
                    mode='w', suffix='.json', delete=False
                )
                json.dump(schema, schema_file)
                schema_file.close()
                cmd.extend(["--output-schema", schema_file.name])

            cmd.extend(["-c", f"model={model}"])

            try:
                self.logger.info(f"Codex exec: model={model}")
                result = subprocess.run(
                    cmd, capture_output=True, text=True, check=False,
                    encoding="utf-8", shell=(sys.platform == "win32"),
                    timeout=120,
                )
                if schema_file:
                    os.unlink(schema_file.name)

                if result.returncode == 0 and result.stdout:
                    final_text = self._extract_final_text(result.stdout)
                    if final_text:
                        return LLMResponse(content=final_text, model=model)

                self.logger.warning(f"Codex failed (rc={result.returncode})")
            except subprocess.TimeoutExpired:
                self.logger.warning("Codex exec timed out")
            except Exception as e:
                self.logger.warning(f"Codex error: {e}")

        # Fallback to Gemini
        self.logger.info("Falling back to Gemini 3 Flash")
        fallback = GoogleProvider()
        fallback_config = LLMConfig(
            provider="google", model="gemini-3-flash-preview",
            max_tokens=config.max_tokens if config else 4096,
            temperature=config.temperature if config else 0.7,
            json_schema=config.json_schema if config else None,
        )
        return await fallback.generate(messages, tools, fallback_config)

    def _extract_final_text(self, stdout: str) -> str:
        for line in reversed(stdout.splitlines()):
            try:
                data = json.loads(line)
                if data.get("type") == "item.completed":
                    return data.get("item", {}).get("text", "")
            except (json.JSONDecodeError, KeyError):
                continue
        return stdout.strip()
```

**Step 7: Run tests**

Run: `cd sage-python && python -m pytest tests/test_llm_providers.py -v`
Expected: PASS

**Step 8: Commit**

```bash
git add sage-python/src/sage/llm/ sage-python/tests/test_llm_providers.py
git commit -m "feat(llm): upgrade providers to Codex 5.3 + Gemini 3.x with structured output"
```

---

## Task 1: Agent Runtime Loop (perceive → think → act → learn)

**Files:**
- Modify: `sage-python/src/sage/agent.py`
- Create: `sage-python/src/sage/agent_loop.py`
- Create: `sage-python/tests/test_agent_loop.py`

**Context:** Currently `Agent.run()` is a simple while-loop. We need a structured perceive→think→act→learn cycle with event emission for the dashboard.

**Step 1: Write failing tests**

```python
# sage-python/tests/test_agent_loop.py
import sys, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest
import asyncio
from sage.agent_loop import AgentLoop, LoopEvent, LoopPhase

@pytest.fixture
def mock_llm():
    from sage.llm.mock import MockProvider
    return MockProvider(responses=["<think>Analyzing task</think>\nDone."])

def test_loop_phases_exist():
    assert LoopPhase.PERCEIVE.value == "perceive"
    assert LoopPhase.THINK.value == "think"
    assert LoopPhase.ACT.value == "act"
    assert LoopPhase.LEARN.value == "learn"

def test_loop_event_structure():
    evt = LoopEvent(phase=LoopPhase.THINK, data={"content": "reasoning"})
    assert evt.phase == LoopPhase.THINK
    assert "content" in evt.data

@pytest.mark.asyncio
async def test_agent_loop_emits_events(mock_llm):
    from sage.agent import AgentConfig
    from sage.llm.base import LLMConfig

    config = AgentConfig(
        name="test", llm=LLMConfig(provider="mock", model="mock"),
        max_steps=3, enforce_system3=False,
    )
    events = []
    loop = AgentLoop(config=config, llm_provider=mock_llm, on_event=events.append)
    result = await loop.run("test task")

    phases = [e.phase for e in events]
    assert LoopPhase.PERCEIVE in phases
    assert LoopPhase.THINK in phases

@pytest.mark.asyncio
async def test_agent_loop_learn_updates_memory(mock_llm):
    from sage.agent import AgentConfig
    from sage.llm.base import LLMConfig

    config = AgentConfig(
        name="test", llm=LLMConfig(provider="mock", model="mock"),
        max_steps=3, enforce_system3=False,
    )
    loop = AgentLoop(config=config, llm_provider=mock_llm)
    await loop.run("test task")

    # Learn phase should have recorded something
    assert loop.working_memory.event_count() > 0
```

**Step 2: Run tests to verify failure**

Run: `cd sage-python && python -m pytest tests/test_agent_loop.py -v`
Expected: FAIL (agent_loop module doesn't exist)

**Step 3: Implement AgentLoop**

```python
# sage-python/src/sage/agent_loop.py
"""Structured agent runtime: perceive → think → act → learn."""
from __future__ import annotations

import time
import json
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Callable
from pathlib import Path

from sage.agent import AgentConfig, Agent
from sage.llm.base import LLMProvider, LLMResponse, Message, Role, ToolDef
from sage.tools.registry import ToolRegistry
from sage.memory.working import WorkingMemory
from sage.memory.compressor import MemoryCompressor
from sage.sandbox.manager import SandboxManager
from sage.topology.kg_rlvr import ProcessRewardModel

log = logging.getLogger(__name__)

STREAM_FILE = Path("docs/plans/agent_stream.jsonl")


class LoopPhase(str, Enum):
    PERCEIVE = "perceive"
    THINK = "think"
    ACT = "act"
    LEARN = "learn"


@dataclass
class LoopEvent:
    phase: LoopPhase
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    step: int = 0


class AgentLoop:
    """Structured agent loop with event emission for dashboard."""

    def __init__(
        self,
        config: AgentConfig,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry | None = None,
        memory_compressor: MemoryCompressor | None = None,
        on_event: Callable[[LoopEvent], None] | None = None,
    ):
        self.config = config
        self._llm = llm_provider
        self._tools = tool_registry or ToolRegistry()
        self._on_event = on_event or self._default_event_handler
        self.working_memory = WorkingMemory(agent_id=config.name)
        self.memory_compressor = memory_compressor
        self.prm = ProcessRewardModel()
        self.agent_pool: dict[str, Any] = {}

        # Stats
        self.step_count = 0
        self.total_inference_time = 0.0
        self.start_time = 0.0

    def _emit(self, phase: LoopPhase, **data: Any) -> None:
        evt = LoopEvent(phase=phase, data=data, step=self.step_count)
        self._on_event(evt)

    def _default_event_handler(self, event: LoopEvent) -> None:
        log.info(f"[{event.phase.value}] step={event.step} {event.data}")
        # Append to stream file for dashboard
        try:
            STREAM_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(STREAM_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "type": event.phase.value.upper(),
                    "step": event.step,
                    "timestamp": event.timestamp,
                    "meta": event.data,
                }) + "\n")
        except Exception:
            pass

    async def run(self, task: str) -> str:
        """Execute the full perceive→think→act→learn cycle."""
        self.start_time = time.perf_counter()

        # === PERCEIVE: Gather context ===
        self._emit(LoopPhase.PERCEIVE, task=task, agent=self.config.name)

        system_prompt = self.config.system_prompt
        if self.config.enforce_system3:
            system_prompt += (
                "\n\nCRITICAL: Use <think>...</think> tags to reason step-by-step. "
                "Your reasoning is evaluated by a Process Reward Model."
            )

        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=task),
        ]
        self.working_memory.add_event("USER", task)
        tool_defs = self._tools.get_tool_defs(
            self.config.tools if self.config.tools else None
        )

        result_text = ""

        while self.step_count < self.config.max_steps:
            self.step_count += 1

            # Memory compression if needed
            if self.memory_compressor:
                compressed = await self.memory_compressor.step(self.working_memory)
                if compressed:
                    messages = self._rebuild_messages(system_prompt)

            # === THINK: Call LLM ===
            self._emit(LoopPhase.THINK, model=self.config.llm.model, step=self.step_count)

            t0 = time.perf_counter()
            response = await self._llm.generate(
                messages=messages,
                tools=tool_defs if tool_defs else None,
                config=self.config.llm,
            )
            self.total_inference_time += (time.perf_counter() - t0)

            content = response.content or ""

            # System 3 validation
            if self.config.enforce_system3 and content:
                r_path, details = self.prm.calculate_r_path(content)
                self._emit(LoopPhase.THINK, r_path=r_path, details=details)
                if r_path < 0.0 and "error" in details:
                    messages.append(Message(
                        role=Role.USER,
                        content="SYSTEM: Use <think> tags for structured reasoning.",
                    ))
                    continue

            self.working_memory.add_event("ASSISTANT", content)

            # No tool calls → final answer
            if not response.tool_calls:
                result_text = content
                messages.append(Message(role=Role.ASSISTANT, content=content))
                break

            # === ACT: Execute tools ===
            messages.append(Message(role=Role.ASSISTANT, content=content))

            for tc in response.tool_calls:
                self._emit(LoopPhase.ACT, tool=tc.name, args=tc.arguments)

                tool = self._tools.get(tc.name)
                if tool is None:
                    output = f"Error: Unknown tool '{tc.name}'"
                else:
                    kwargs = tc.arguments.copy()
                    result = await tool.execute(kwargs)
                    output = result.output

                self.working_memory.add_event("TOOL", f"{tc.name} -> {output}")
                messages.append(Message(
                    role=Role.TOOL, content=output,
                    tool_call_id=tc.id, name=tc.name,
                ))

            # === LEARN: Update memory and stats ===
            aio = self._compute_aio()
            self._emit(LoopPhase.LEARN, aio_ratio=aio, events=self.working_memory.event_count())

        self._emit(LoopPhase.LEARN, result="complete", steps=self.step_count)
        return result_text or f"Agent finished at step {self.step_count}"

    def _compute_aio(self) -> float:
        wall = time.perf_counter() - self.start_time
        if wall <= 0:
            return 0.0
        return max(0.0, (wall - self.total_inference_time) / wall)

    def _rebuild_messages(self, system_prompt: str) -> list[Message]:
        msgs = [Message(role=Role.SYSTEM, content=system_prompt)]
        for event in self.working_memory._events:
            role_map = {
                "SYSTEM": Role.SYSTEM, "USER": Role.USER,
                "ASSISTANT": Role.ASSISTANT, "TOOL": Role.USER,
                "summary": Role.SYSTEM,
            }
            role = role_map.get(event["type"], Role.USER)
            msgs.append(Message(role=role, content=event["content"]))
        return msgs
```

**Step 4: Run tests**

Run: `cd sage-python && python -m pytest tests/test_agent_loop.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add sage-python/src/sage/agent_loop.py sage-python/tests/test_agent_loop.py
git commit -m "feat(agent): add structured perceive→think→act→learn runtime loop"
```

---

## Task 2: Dynamic Sub-Agent API

**Files:**
- Modify: `sage-python/src/sage/tools/agent_mgmt.py`
- Create: `sage-python/src/sage/agent_pool.py`
- Create: `sage-python/tests/test_agent_pool.py`

**Step 1: Write failing tests**

```python
# sage-python/tests/test_agent_pool.py
import sys, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest
from sage.agent_pool import AgentPool, SubAgentSpec

def test_create_sub_agent_spec():
    spec = SubAgentSpec(
        name="researcher", role="research",
        system_prompt="You research topics.", tools=["web_search"],
    )
    assert spec.name == "researcher"
    assert spec.role == "research"

def test_pool_register_and_list():
    pool = AgentPool()
    spec = SubAgentSpec(name="coder", role="code", system_prompt="Write code.")
    pool.register(spec)
    agents = pool.list_agents()
    assert len(agents) == 1
    assert agents[0]["name"] == "coder"

def test_pool_deregister():
    pool = AgentPool()
    spec = SubAgentSpec(name="temp", role="temp", system_prompt="Temporary.")
    pool.register(spec)
    pool.deregister("temp")
    assert len(pool.list_agents()) == 0

def test_pool_ensemble_empty():
    pool = AgentPool()
    results = pool.collect_results()
    assert results == {}
```

**Step 2: Run tests — expect FAIL**

Run: `cd sage-python && python -m pytest tests/test_agent_pool.py -v`

**Step 3: Implement AgentPool**

```python
# sage-python/src/sage/agent_pool.py
"""Dynamic sub-agent pool for multi-agent orchestration."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class SubAgentSpec:
    """Specification for creating a sub-agent."""
    name: str
    role: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)
    llm_tier: str = "fast"  # fast, mutator, reasoner, codex
    max_steps: int = 50
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentPool:
    """Thread-safe pool for managing dynamically created sub-agents.

    OpenSAGE-style: parent agent can create, run, and ensemble sub-agents.
    """

    def __init__(self):
        self._specs: dict[str, SubAgentSpec] = {}
        self._results: dict[str, str] = {}
        self._running: set[str] = set()

    def register(self, spec: SubAgentSpec) -> None:
        """Register a new sub-agent specification."""
        self._specs[spec.name] = spec
        log.info(f"Registered sub-agent: {spec.name} (role={spec.role})")

    def deregister(self, name: str) -> None:
        """Remove a sub-agent from the pool."""
        self._specs.pop(name, None)
        self._results.pop(name, None)
        self._running.discard(name)

    def get_spec(self, name: str) -> SubAgentSpec | None:
        return self._specs.get(name)

    def list_agents(self) -> list[dict[str, Any]]:
        return [
            {
                "name": s.name,
                "role": s.role,
                "llm_tier": s.llm_tier,
                "tools": s.tools,
                "running": s.name in self._running,
                "has_result": s.name in self._results,
            }
            for s in self._specs.values()
        ]

    def mark_running(self, name: str) -> None:
        self._running.add(name)

    def store_result(self, name: str, result: str) -> None:
        self._results[name] = result
        self._running.discard(name)

    def collect_results(self) -> dict[str, str]:
        """Collect all completed results (ensemble pattern)."""
        return dict(self._results)

    def clear_results(self) -> None:
        self._results.clear()
```

**Step 4: Run tests — expect PASS**

Run: `cd sage-python && python -m pytest tests/test_agent_pool.py -v`

**Step 5: Update agent_mgmt.py tools to use AgentPool**

In `sage-python/src/sage/tools/agent_mgmt.py`, update `create_agent` and `run_agent` tools to instantiate `AgentLoop` with specs from the pool, and add `ensemble_agents` tool that calls `pool.collect_results()`.

**Step 6: Commit**

```bash
git add sage-python/src/sage/agent_pool.py sage-python/tests/test_agent_pool.py sage-python/src/sage/tools/agent_mgmt.py
git commit -m "feat(agents): add dynamic sub-agent pool with create/run/ensemble API"
```

---

## Task 3: LLM Mutator Integration (Real Code Evolution)

**Files:**
- Modify: `sage-python/src/sage/evolution/llm_mutator.py`
- Create: `sage-python/tests/test_llm_mutator.py`

**Context:** The `llm_mutator.py` needs to call a real LLM (Gemini 3 Flash) to generate SEARCH/REPLACE mutations with behavioral features, using structured JSON output.

**Step 1: Write failing tests**

```python
# sage-python/tests/test_llm_mutator.py
import sys, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest
from sage.evolution.llm_mutator import LLMMutator, MutationRequest, MutationResponse

def test_mutation_request_structure():
    req = MutationRequest(
        code="def sort(arr): return sorted(arr)",
        objective="Optimize sorting for large arrays",
        context="Previous best: O(n log n)"
    )
    assert req.code is not None
    assert req.objective is not None

def test_mutation_response_structure():
    resp = MutationResponse(
        mutations=[{"search": "sorted(arr)", "replace": "arr.sort(); return arr", "description": "in-place sort"}],
        features=[3, 7],
        reasoning="In-place sorting reduces memory allocation",
    )
    assert len(resp.mutations) == 1
    assert len(resp.features) == 2

def test_mutator_builds_prompt():
    mutator = LLMMutator(llm_tier="budget")
    prompt = mutator._build_mutation_prompt(
        code="x = 1", objective="optimize", context=""
    )
    assert "SEARCH" in prompt
    assert "REPLACE" in prompt
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement LLMMutator with Pydantic structured output**

```python
# sage-python/src/sage/evolution/llm_mutator.py
"""LLM-based code mutator using Gemini/Codex with structured JSON output."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from sage.llm.base import LLMConfig, Message, Role
from sage.llm.router import ModelRouter

log = logging.getLogger(__name__)

MUTATION_SYSTEM_PROMPT = """You are an expert code evolution engine. Given source code and an objective,
generate SEARCH/REPLACE mutations that improve the code toward the objective.

Rules:
1. Each mutation must have an exact `search` string (verbatim from the code) and a `replace` string.
2. Mutations must be syntactically valid and maintain the function signature.
3. Provide `features` as a list of 2 integers (0-9) describing behavioral dimensions.
4. Provide brief `reasoning` explaining the improvement.
"""


class MutationItem(BaseModel):
    search: str
    replace: str
    description: str


class MutationResponse(BaseModel):
    mutations: list[MutationItem]
    features: list[int]
    reasoning: str


@dataclass
class MutationRequest:
    code: str
    objective: str
    context: str = ""


class LLMMutator:
    """Generates code mutations via LLM with structured JSON output."""

    def __init__(self, llm_tier: str = "mutator"):
        self.llm_tier = llm_tier

    def _build_mutation_prompt(self, code: str, objective: str, context: str) -> str:
        prompt = f"## Objective\n{objective}\n\n"
        if context:
            prompt += f"## Context\n{context}\n\n"
        prompt += f"## Source Code\n```\n{code}\n```\n\n"
        prompt += "Generate 1-3 mutations as SEARCH/REPLACE pairs. Respond in the required JSON format."
        return prompt

    async def mutate(self, request: MutationRequest) -> MutationResponse:
        """Generate mutations using LLM with structured output."""
        config = ModelRouter.get_config(
            self.llm_tier,
            temperature=0.8,
            json_schema=MutationResponse,
        )

        prompt = self._build_mutation_prompt(
            request.code, request.objective, request.context
        )

        messages = [
            Message(role=Role.SYSTEM, content=MUTATION_SYSTEM_PROMPT),
            Message(role=Role.USER, content=prompt),
        ]

        # Get provider based on tier
        if config.provider == "codex":
            from sage.llm.codex import CodexProvider
            provider = CodexProvider()
        else:
            from sage.llm.google import GoogleProvider
            provider = GoogleProvider()

        response = await provider.generate(messages, config=config)

        try:
            return MutationResponse.model_validate_json(response.content)
        except Exception as e:
            log.warning(f"Failed to parse mutation response: {e}")
            # Attempt lenient JSON extraction
            text = response.content
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                return MutationResponse.model_validate_json(text[start:end])
            raise
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add sage-python/src/sage/evolution/llm_mutator.py sage-python/tests/test_llm_mutator.py
git commit -m "feat(evolution): wire LLM mutator to Gemini 3 Flash with structured output"
```

---

## Task 4: Memory Agent + Neo4j Persistence + Temporal KG

**Files:**
- Create: `sage-python/src/sage/memory/memory_agent.py`
- Modify: `sage-python/src/sage/memory/neo4j_driver.py`
- Modify: `sage-core/src/memory/smmu.rs` (add timestamp to edges)
- Create: `sage-python/tests/test_memory_agent.py`

**Step 1: Write failing tests**

```python
# sage-python/tests/test_memory_agent.py
import sys, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest
from sage.memory.memory_agent import MemoryAgent, ExtractionResult

def test_extraction_result():
    r = ExtractionResult(
        entities=["Z3", "eBPF"],
        relationships=[("Z3", "verifies", "eBPF")],
        summary="Z3 verifies eBPF bytecode safety"
    )
    assert len(r.entities) == 2
    assert len(r.relationships) == 1

@pytest.mark.asyncio
async def test_memory_agent_extract_from_text():
    agent = MemoryAgent(use_llm=False)
    result = await agent.extract("The Z3 solver verified that the eBPF bytecode is safe.")
    assert isinstance(result, ExtractionResult)
    assert len(result.entities) >= 0  # Heuristic mode may find some

def test_memory_agent_should_compress():
    agent = MemoryAgent(use_llm=False, compress_threshold=5)
    assert not agent.should_compress(event_count=3)
    assert agent.should_compress(event_count=6)
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement MemoryAgent**

```python
# sage-python/src/sage/memory/memory_agent.py
"""Autonomous Memory Agent: extracts entities/relations, manages compression."""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    entities: list[str] = field(default_factory=list)
    relationships: list[tuple[str, str, str]] = field(default_factory=list)
    summary: str = ""


class MemoryAgent:
    """Runs asynchronously to compress working memory into graph knowledge.

    Extracts entities and relationships from agent events, stores in
    Neo4j/Qdrant for cross-session persistence.
    """

    def __init__(
        self,
        use_llm: bool = True,
        llm_tier: str = "budget",
        compress_threshold: int = 50,
    ):
        self.use_llm = use_llm
        self.llm_tier = llm_tier
        self.compress_threshold = compress_threshold

    def should_compress(self, event_count: int) -> bool:
        return event_count > self.compress_threshold

    async def extract(self, text: str) -> ExtractionResult:
        """Extract entities and relationships from text."""
        if self.use_llm:
            return await self._llm_extract(text)
        return self._heuristic_extract(text)

    def _heuristic_extract(self, text: str) -> ExtractionResult:
        """Fast heuristic extraction (no LLM cost)."""
        # Extract capitalized terms as entities
        entities = list(set(re.findall(r'\b[A-Z][A-Za-z0-9_-]{2,}\b', text)))

        # Simple verb-based relationship extraction
        relationships = []
        for ent in entities:
            pattern = rf'{ent}\s+(verif\w+|uses?|creates?|calls?|returns?)\s+(\w+)'
            matches = re.findall(pattern, text, re.IGNORECASE)
            for verb, obj in matches:
                relationships.append((ent, verb, obj))

        return ExtractionResult(
            entities=entities[:20],
            relationships=relationships[:10],
            summary=text[:200] if len(text) > 200 else text,
        )

    async def _llm_extract(self, text: str) -> ExtractionResult:
        """LLM-powered extraction with structured output."""
        from pydantic import BaseModel
        from sage.llm.router import ModelRouter
        from sage.llm.google import GoogleProvider
        from sage.llm.base import Message, Role

        class KGExtraction(BaseModel):
            entities: list[str]
            relationships: list[list[str]]  # [subject, predicate, object]
            summary: str

        config = ModelRouter.get_config(
            self.llm_tier, temperature=0.1, json_schema=KGExtraction,
        )
        provider = GoogleProvider()
        response = await provider.generate(
            messages=[
                Message(role=Role.SYSTEM, content=(
                    "Extract entities and relationships from the text. "
                    "Return JSON with entities (list of names), "
                    "relationships (list of [subject, predicate, object]), "
                    "and a one-sentence summary."
                )),
                Message(role=Role.USER, content=text),
            ],
            config=config,
        )

        try:
            parsed = KGExtraction.model_validate_json(response.content)
            return ExtractionResult(
                entities=parsed.entities,
                relationships=[tuple(r) for r in parsed.relationships if len(r) == 3],
                summary=parsed.summary,
            )
        except Exception as e:
            log.warning(f"LLM extraction failed: {e}, falling back to heuristic")
            return self._heuristic_extract(text)
```

**Step 4: Add temporal timestamps to S-MMU edges (Rust)**

In `sage-core/src/memory/smmu.rs`, add `created_at: f64` to `EdgePayload`:
```rust
struct EdgePayload {
    kind: EdgeKind,
    weight: f32,
    created_at: f64, // Unix timestamp for temporal queries
}
```

Update `add_edge_internal()` to set `created_at: std::time::SystemTime::now()...`.

**Step 5: Run tests — expect PASS**

**Step 6: Commit**

```bash
git add sage-python/src/sage/memory/memory_agent.py sage-python/tests/test_memory_agent.py sage-core/src/memory/smmu.rs
git commit -m "feat(memory): add MemoryAgent with entity extraction and temporal KG edges"
```

---

## Task 5: Evolutionary Topology Search (MAP-Elites on Agent DAGs)

**Files:**
- Create: `sage-python/src/sage/topology/evo_topology.py`
- Modify: `sage-python/src/sage/topology/planner.py`
- Create: `sage-python/tests/test_evo_topology.py`

**Context:** Instead of zero-shot generating topologies like OpenSAGE, we evolve them via MAP-Elites. Topology configurations are the "genome"; task performance is fitness.

**Step 1: Write failing tests**

```python
# sage-python/tests/test_evo_topology.py
import sys, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest
from sage.topology.evo_topology import (
    TopologyGenome, TopologyPopulation, TopologyEvolver
)

def test_topology_genome():
    g = TopologyGenome(
        nodes=["planner", "coder", "reviewer"],
        edges=[("planner", "coder"), ("coder", "reviewer")],
        pattern="vertical",
    )
    assert len(g.nodes) == 3
    assert g.pattern == "vertical"

def test_population_add_and_best():
    pop = TopologyPopulation(bins_per_dim=5)
    g1 = TopologyGenome(nodes=["a"], edges=[], pattern="single", features=(1, 1))
    pop.add(g1, score=0.5)
    g2 = TopologyGenome(nodes=["a", "b"], edges=[("a","b")], pattern="vertical", features=(1, 1))
    pop.add(g2, score=0.8)
    best = pop.best()
    assert best is not None
    assert best[1] == 0.8  # Higher score wins

def test_evolver_mutate_genome():
    evolver = TopologyEvolver()
    g = TopologyGenome(
        nodes=["planner", "coder"],
        edges=[("planner", "coder")],
        pattern="vertical",
        features=(3, 5),
    )
    mutated = evolver.mutate_genome(g)
    assert isinstance(mutated, TopologyGenome)
    # Mutation should change something
    assert mutated.nodes != g.nodes or mutated.edges != g.edges or mutated.pattern != g.pattern
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement evolutionary topology search**

```python
# sage-python/src/sage/topology/evo_topology.py
"""Evolutionary Topology Search: MAP-Elites on Agent DAG configurations."""
from __future__ import annotations

import random
import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

PATTERNS = ["vertical", "horizontal", "mesh", "coordinator", "pipeline"]
ROLES = ["planner", "coder", "reviewer", "researcher", "debugger", "tester", "analyst"]


@dataclass
class TopologyGenome:
    """Genome encoding an agent DAG topology."""
    nodes: list[str]
    edges: list[tuple[str, str]]
    pattern: str
    features: tuple[int, int] = (0, 0)  # Behavioral features for MAP-Elites
    metadata: dict[str, Any] = field(default_factory=dict)


class TopologyPopulation:
    """MAP-Elites grid for topology diversity."""

    def __init__(self, bins_per_dim: int = 10):
        self.bins = bins_per_dim
        self._grid: dict[tuple[int, int], tuple[TopologyGenome, float]] = {}

    def add(self, genome: TopologyGenome, score: float) -> bool:
        key = (
            min(genome.features[0], self.bins - 1),
            min(genome.features[1], self.bins - 1),
        )
        if key not in self._grid or self._grid[key][1] < score:
            self._grid[key] = (genome, score)
            return True
        return False

    def best(self) -> tuple[TopologyGenome, float] | None:
        if not self._grid:
            return None
        return max(self._grid.values(), key=lambda x: x[1])

    def sample(self, n: int = 1) -> list[TopologyGenome]:
        items = list(self._grid.values())
        if not items:
            return []
        chosen = random.choices(items, k=min(n, len(items)))
        return [g for g, _ in chosen]

    def size(self) -> int:
        return len(self._grid)


class TopologyEvolver:
    """Evolves agent DAG topologies using mutation operators."""

    def mutate_genome(self, genome: TopologyGenome) -> TopologyGenome:
        """Apply a random mutation to the topology genome."""
        nodes = list(genome.nodes)
        edges = list(genome.edges)
        pattern = genome.pattern

        op = random.choice(["add_node", "remove_node", "rewire", "change_pattern"])

        if op == "add_node" and len(nodes) < 8:
            new_role = random.choice([r for r in ROLES if r not in nodes])
            nodes.append(new_role)
            # Connect to a random existing node
            if nodes:
                parent = random.choice(nodes[:-1]) if len(nodes) > 1 else nodes[0]
                edges.append((parent, new_role))

        elif op == "remove_node" and len(nodes) > 1:
            victim = random.choice(nodes[1:])  # Never remove first node
            nodes.remove(victim)
            edges = [(a, b) for a, b in edges if a != victim and b != victim]

        elif op == "rewire" and edges:
            idx = random.randrange(len(edges))
            src, _ = edges[idx]
            tgt = random.choice([n for n in nodes if n != src])
            edges[idx] = (src, tgt)

        elif op == "change_pattern":
            pattern = random.choice([p for p in PATTERNS if p != pattern])

        # Compute behavioral features: (node_count_bin, edge_density_bin)
        nc = min(9, len(nodes))
        max_edges = max(1, len(nodes) * (len(nodes) - 1))
        ed = min(9, int(10 * len(edges) / max_edges))

        return TopologyGenome(
            nodes=nodes, edges=edges, pattern=pattern,
            features=(nc, ed),
        )

    def crossover(self, a: TopologyGenome, b: TopologyGenome) -> TopologyGenome:
        """Combine two topologies."""
        # Take nodes from both, edges from the one with better diversity
        all_nodes = list(set(a.nodes + b.nodes))[:8]
        all_edges = list(set(a.edges + b.edges))
        # Filter edges to valid nodes
        valid_edges = [(s, t) for s, t in all_edges if s in all_nodes and t in all_nodes]
        pattern = random.choice([a.pattern, b.pattern])

        nc = min(9, len(all_nodes))
        max_e = max(1, len(all_nodes) * (len(all_nodes) - 1))
        ed = min(9, int(10 * len(valid_edges) / max_e))

        return TopologyGenome(
            nodes=all_nodes, edges=valid_edges, pattern=pattern,
            features=(nc, ed),
        )
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add sage-python/src/sage/topology/evo_topology.py sage-python/tests/test_evo_topology.py
git commit -m "feat(topology): add evolutionary topology search with MAP-Elites on DAGs"
```

---

## Task 6: Metacognitive Controller (SOFAI Routing + Self-Braking)

**Files:**
- Create: `sage-python/src/sage/strategy/metacognition.py`
- Create: `sage-python/tests/test_metacognition.py`

**Step 1: Write failing tests**

```python
# sage-python/tests/test_metacognition.py
import sys, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest
from sage.strategy.metacognition import (
    MetacognitiveController, CognitiveProfile, RoutingDecision
)

def test_cognitive_profile():
    p = CognitiveProfile(complexity=0.3, uncertainty=0.2, tool_required=False)
    assert p.complexity < 0.5

def test_route_simple_to_system1():
    ctrl = MetacognitiveController()
    decision = ctrl.route(CognitiveProfile(complexity=0.1, uncertainty=0.1, tool_required=False))
    assert decision.system == 1
    assert decision.llm_tier == "fast"

def test_route_complex_to_system3():
    ctrl = MetacognitiveController()
    decision = ctrl.route(CognitiveProfile(complexity=0.9, uncertainty=0.8, tool_required=True))
    assert decision.system == 3
    assert decision.llm_tier in ("reasoner", "codex")

def test_self_braking_detects_convergence():
    ctrl = MetacognitiveController()
    # Simulate repeated similar outputs
    ctrl.record_output_entropy(0.1)
    ctrl.record_output_entropy(0.08)
    ctrl.record_output_entropy(0.05)
    assert ctrl.should_brake()

def test_self_braking_allows_divergence():
    ctrl = MetacognitiveController()
    ctrl.record_output_entropy(0.9)
    ctrl.record_output_entropy(0.85)
    assert not ctrl.should_brake()
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement MetacognitiveController**

```python
# sage-python/src/sage/strategy/metacognition.py
"""Metacognitive Controller: SOFAI-style System 1/3 routing + self-braking."""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class CognitiveProfile:
    """Assessment of a task's cognitive requirements."""
    complexity: float     # 0.0 = trivial, 1.0 = extremely complex
    uncertainty: float    # 0.0 = certain, 1.0 = highly uncertain
    tool_required: bool   # Whether tool use is expected


@dataclass
class RoutingDecision:
    """Which system and LLM tier to use."""
    system: int           # 1 = fast/intuitive, 3 = formal/deliberate
    llm_tier: str         # fast, mutator, reasoner, codex
    max_tokens: int
    use_z3: bool          # Whether to validate with Z3 PRM


class MetacognitiveController:
    """Routes tasks between System 1 (fast) and System 3 (formal reasoning).

    SOFAI pattern: evaluates task complexity and model confidence to decide
    whether to use fast heuristic LLM or full Z3-backed reasoning pipeline.

    Self-braking (CGRS): monitors output entropy to detect convergence
    and suppress unnecessary reasoning loops.
    """

    def __init__(
        self,
        complexity_threshold: float = 0.5,
        uncertainty_threshold: float = 0.4,
        brake_window: int = 3,
        brake_entropy_threshold: float = 0.15,
    ):
        self.complexity_threshold = complexity_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.brake_window = brake_window
        self.brake_entropy_threshold = brake_entropy_threshold
        self._entropy_history: deque[float] = deque(maxlen=10)

    def route(self, profile: CognitiveProfile) -> RoutingDecision:
        """Decide which cognitive system to engage."""
        needs_system3 = (
            profile.complexity > self.complexity_threshold
            or profile.uncertainty > self.uncertainty_threshold
            or profile.tool_required
        )

        if needs_system3:
            # High complexity: use formal reasoning
            tier = "reasoner"
            if profile.complexity > 0.8:
                tier = "codex"  # Hardest tasks get agentic Codex
            return RoutingDecision(
                system=3, llm_tier=tier,
                max_tokens=8192, use_z3=True,
            )
        else:
            # Simple task: fast heuristic
            return RoutingDecision(
                system=1, llm_tier="fast",
                max_tokens=2048, use_z3=False,
            )

    def record_output_entropy(self, entropy: float) -> None:
        """Record the entropy of the latest LLM output for self-braking."""
        self._entropy_history.append(entropy)

    def should_brake(self) -> bool:
        """Determine if the agent should stop reasoning (convergence detected).

        CGRS: If the last N outputs all have low entropy, the model has
        implicitly converged and further reasoning is wasteful.
        """
        if len(self._entropy_history) < self.brake_window:
            return False
        recent = list(self._entropy_history)[-self.brake_window:]
        return all(e < self.brake_entropy_threshold for e in recent)

    def assess_complexity(self, task: str) -> CognitiveProfile:
        """Quick heuristic assessment of task complexity."""
        lower = task.lower()

        complexity = 0.3  # Base
        if any(w in lower for w in ["debug", "fix", "error", "crash"]):
            complexity += 0.3
        if any(w in lower for w in ["optimize", "evolve", "design", "architect"]):
            complexity += 0.2
        if len(task) > 500:
            complexity += 0.1

        uncertainty = 0.2
        if "?" in task:
            uncertainty += 0.2
        if any(w in lower for w in ["maybe", "possibly", "explore", "investigate"]):
            uncertainty += 0.2

        tool_required = any(w in lower for w in [
            "file", "search", "run", "execute", "compile", "test", "deploy"
        ])

        return CognitiveProfile(
            complexity=min(1.0, complexity),
            uncertainty=min(1.0, uncertainty),
            tool_required=tool_required,
        )
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git add sage-python/src/sage/strategy/metacognition.py sage-python/tests/test_metacognition.py
git commit -m "feat(strategy): add SOFAI metacognitive controller with self-braking"
```

---

## Task 7: Control Dashboard (React + Tailwind + FastAPI WebSocket)

**Files:**
- Modify: `ui/app.py`
- Create: `ui/static/index.html` (complete rewrite)
- Create: `ui/static/app.js`
- Create: `ui/static/styles.css`

**Context:** Replace the basic log viewer with a production dashboard featuring:
- Real-time agent loop telemetry (perceive/think/act/learn phases)
- Sub-agent pool visualization
- Evolution progress (MAP-Elites grid)
- Memory graph view
- LLM provider status and costs
- Metacognitive routing decisions

**Step 1: Upgrade FastAPI backend with REST endpoints**

```python
# ui/app.py
import asyncio
import json
import time
from pathlib import Path
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="YGN-SAGE Control Dashboard")
app.mount("/static", StaticFiles(directory="ui/static"), name="static")

LOG_FILE = Path("docs/plans/agent_stream.jsonl")

# In-memory state for dashboard
dashboard_state = {
    "agent_status": "idle",
    "current_phase": None,
    "step_count": 0,
    "llm_calls": 0,
    "total_cost_usd": 0.0,
    "sub_agents": [],
    "evolution_stats": {},
    "memory_events": 0,
    "aio_ratio": 0.0,
    "metacognitive_system": 1,
}


@app.get("/")
async def root():
    with open("ui/static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/api/state")
async def get_state():
    return JSONResponse(dashboard_state)


@app.post("/api/task")
async def submit_task(request: Request):
    body = await request.json()
    task = body.get("task", "")
    dashboard_state["agent_status"] = "running"
    dashboard_state["current_phase"] = "perceive"
    # In production, this would trigger AgentLoop.run(task)
    return JSONResponse({"status": "accepted", "task": task})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    if LOG_FILE.exists():
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    await websocket.send_text(line)

    last_pos = LOG_FILE.stat().st_size if LOG_FILE.exists() else 0

    try:
        while True:
            if LOG_FILE.exists():
                current_size = LOG_FILE.stat().st_size
                if current_size < last_pos:
                    last_pos = 0
                if current_size > last_pos:
                    with open(LOG_FILE, "r", encoding="utf-8") as f:
                        f.seek(last_pos)
                        for line in f:
                            if line.strip():
                                await websocket.send_text(line)
                                # Update dashboard state from events
                                try:
                                    evt = json.loads(line)
                                    dashboard_state["current_phase"] = evt.get("type", "").lower()
                                    dashboard_state["step_count"] = evt.get("step", 0)
                                    meta = evt.get("meta", {})
                                    if "aio_ratio" in meta:
                                        dashboard_state["aio_ratio"] = meta["aio_ratio"]
                                except json.JSONDecodeError:
                                    pass
                        last_pos = f.tell()
            await asyncio.sleep(0.1)
    except Exception:
        pass


if __name__ == "__main__":
    print("Starting YGN-SAGE Dashboard on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
```

**Step 2: Create the dashboard frontend**

The HTML file should be a single-page app with these panels:

1. **Header**: YGN-SAGE logo, agent status indicator, current phase badge
2. **Left sidebar**: Agent pool list, LLM provider selector, quick actions
3. **Main area (grid layout)**:
   - **Agent Loop Phase indicator** (4-step circular progress: perceive→think→act→learn)
   - **Live Event Stream** (scrollable log with phase-colored entries)
   - **Telemetry cards** (AIO ratio, step count, LLM cost, memory events)
   - **Evolution Grid** (MAP-Elites heatmap visualization)
4. **Right sidebar**: Metacognitive routing display, Z3 verification status

**Design specifications:**
- Dark theme: bg `#0f172a` (slate-900), cards `#1e293b` (slate-800)
- Accent: `#38bdf8` (sky-400) for primary, `#10b981` (emerald-500) for success
- Phase colors: perceive=`#818cf8` (indigo), think=`#38bdf8` (sky), act=`#f59e0b` (amber), learn=`#10b981` (emerald)
- Font: Inter for UI, JetBrains Mono for code/logs
- Animations: smooth phase transitions, pulse on new events
- Responsive: works on 1920x1080 and 1440x900

Use vanilla JS (no build step) with Tailwind CDN for rapid iteration. WebSocket for real-time updates, fetch for REST API calls.

The exact HTML/CSS/JS will be generated by the `frontend-design` skill during implementation.

**Step 3: Run the dashboard**

Run: `cd /c/Code/YGN-SAGE && python ui/app.py`
Expected: Dashboard at http://localhost:8000 with all panels rendering

**Step 4: Commit**

```bash
git add ui/
git commit -m "feat(ui): production dashboard with real-time agent telemetry"
```

---

## Task 8: Integration Wiring + End-to-End Test

**Files:**
- Create: `sage-python/src/sage/boot.py`
- Create: `sage-python/tests/test_integration.py`
- Modify: `sage-python/src/sage/__init__.py`

**Context:** Wire all components together: AgentLoop + AgentPool + LLMMutator + MemoryAgent + MetacognitiveController + TopologyEvolver. Create a single `boot()` function that initializes the full stack.

**Step 1: Write integration test**

```python
# sage-python/tests/test_integration.py
import sys, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest
from sage.boot import boot_agent_system, AgentSystem

def test_boot_creates_system():
    system = boot_agent_system(use_mock_llm=True)
    assert isinstance(system, AgentSystem)
    assert system.agent_loop is not None
    assert system.agent_pool is not None
    assert system.metacognition is not None
    assert system.topology_evolver is not None

@pytest.mark.asyncio
async def test_full_cycle_with_mock():
    system = boot_agent_system(use_mock_llm=True)
    result = await system.run("What is 2+2?")
    assert result is not None
    assert system.agent_loop.step_count > 0
```

**Step 2: Implement boot.py**

```python
# sage-python/src/sage/boot.py
"""Boot sequence: initialize the full YGN-SAGE agent stack."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sage.agent import AgentConfig
from sage.agent_loop import AgentLoop
from sage.agent_pool import AgentPool
from sage.llm.base import LLMConfig
from sage.llm.mock import MockProvider
from sage.llm.router import ModelRouter
from sage.strategy.metacognition import MetacognitiveController
from sage.topology.evo_topology import TopologyEvolver, TopologyPopulation
from sage.memory.memory_agent import MemoryAgent
from sage.tools.registry import ToolRegistry


@dataclass
class AgentSystem:
    """The complete YGN-SAGE agent system."""
    agent_loop: AgentLoop
    agent_pool: AgentPool
    metacognition: MetacognitiveController
    topology_evolver: TopologyEvolver
    topology_population: TopologyPopulation
    memory_agent: MemoryAgent
    tool_registry: ToolRegistry

    async def run(self, task: str) -> str:
        # 1. Assess task complexity
        profile = self.metacognition.assess_complexity(task)
        decision = self.metacognition.route(profile)

        # 2. Update agent LLM tier based on routing
        self.agent_loop.config.llm = ModelRouter.get_config(decision.llm_tier)

        # 3. Run the agent loop
        return await self.agent_loop.run(task)


def boot_agent_system(
    use_mock_llm: bool = False,
    llm_tier: str = "fast",
    agent_name: str = "sage-main",
) -> AgentSystem:
    """Initialize the complete agent stack."""
    # LLM
    if use_mock_llm:
        provider = MockProvider(responses=["<think>Processing</think>\nDone."])
        llm_config = LLMConfig(provider="mock", model="mock")
    else:
        llm_config = ModelRouter.get_config(llm_tier)
        if llm_config.provider == "codex":
            from sage.llm.codex import CodexProvider
            provider = CodexProvider()
        else:
            from sage.llm.google import GoogleProvider
            provider = GoogleProvider()

    # Components
    tool_registry = ToolRegistry()
    agent_pool = AgentPool()
    metacognition = MetacognitiveController()
    topology_evolver = TopologyEvolver()
    topology_population = TopologyPopulation()
    memory_agent = MemoryAgent(use_llm=not use_mock_llm)

    # Agent config
    config = AgentConfig(
        name=agent_name,
        llm=llm_config,
        system_prompt=(
            "You are YGN-SAGE, an advanced AI agent with 5 cognitive pillars: "
            "Topology, Tools, Memory, Evolution, Strategy. "
            "Use <think> tags for structured reasoning."
        ),
        max_steps=100,
        enforce_system3=True,
    )

    # Agent loop
    loop = AgentLoop(
        config=config,
        llm_provider=provider,
        tool_registry=tool_registry,
    )
    loop.agent_pool = agent_pool

    return AgentSystem(
        agent_loop=loop,
        agent_pool=agent_pool,
        metacognition=metacognition,
        topology_evolver=topology_evolver,
        topology_population=topology_population,
        memory_agent=memory_agent,
        tool_registry=tool_registry,
    )
```

**Step 3: Run integration tests**

Run: `cd sage-python && python -m pytest tests/test_integration.py -v`
Expected: PASS

**Step 4: Run full test suite**

Run: `cd sage-python && python -m pytest tests/ -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add sage-python/src/sage/boot.py sage-python/tests/test_integration.py sage-python/src/sage/__init__.py
git commit -m "feat: wire all pillars into unified boot sequence with integration tests"
```

---

## Execution Order & Dependencies

```
Task 0 (LLM Providers) ──┐
                          ├── Task 1 (Agent Loop) ──┬── Task 2 (Sub-Agent API)
                          │                         ├── Task 3 (LLM Mutator)
                          │                         ├── Task 4 (Memory Agent)
                          │                         └── Task 7 (Dashboard)
                          │
                          ├── Task 5 (Evo Topology) ── requires Task 2 + Task 3
                          ├── Task 6 (Metacognition) ── requires Task 1
                          └── Task 8 (Integration) ── requires ALL above
```

Tasks 2, 3, 4, 7 can be parallelized after Task 1 completes.

---

## Verification Checklist

- [ ] `cd sage-python && python -m pytest tests/ -v` — All tests pass
- [ ] `cd sage-core && cargo test` — All 29+ Rust tests pass
- [ ] `cd sage-core && cargo clippy` — No new warnings
- [ ] `python ui/app.py` — Dashboard loads at localhost:8000
- [ ] `python -c "from sage.boot import boot_agent_system; s = boot_agent_system(use_mock_llm=True); print('OK')"` — Boot sequence works
- [ ] `codex --version` — Codex CLI available (or Gemini fallback works)
