# YGN-SAGE Hardening & Maturation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 4 verified agent_loop bugs, add CI/CD, externalize model config, add structured observability, and clean stale artifacts.

**Architecture:** Surgical fixes to `agent_loop.py` (split retry counters, async metacognition, self-brake memory, AVR last-block). New `AgentEvent` schema for structured observability. `models.toml` config loader for ModelRouter. GitHub Actions CI for Python + Rust. Cleanup of stale generated files.

**Tech Stack:** Python 3.12, Rust (cargo), pytest, pytest-asyncio, GitHub Actions, TOML (tomllib stdlib)

---

### Task 1: Self-brake memory gap fix

**Files:**
- Modify: `sage-python/src/sage/agent_loop.py:350-357`
- Test: `sage-python/tests/test_agent_loop.py`

**Step 1: Write the failing test**

Add to `sage-python/tests/test_agent_loop.py`:

```python
@pytest.mark.asyncio
async def test_self_brake_stores_in_working_memory():
    """CGRS self-brake must store response in working_memory before breaking."""
    from sage.agent import AgentConfig
    from sage.llm.base import LLMConfig
    from sage.llm.mock import MockProvider
    from sage.strategy.metacognition import MetacognitiveController

    # Create a controller that will brake immediately
    ctrl = MetacognitiveController(brake_window=1, brake_entropy_threshold=1.0)
    # Pre-fill with low entropy to trigger brake on first output
    ctrl.record_output_entropy(0.01)

    provider = MockProvider(responses=["Braked response content here."])
    config = AgentConfig(
        name="test-brake", llm=LLMConfig(provider="mock", model="mock"),
        max_steps=5, validation_level=1,
    )
    loop = AgentLoop(config=config, llm_provider=provider)
    loop.metacognition = ctrl

    result = await loop.run("test task")

    # The braked response MUST be in working memory
    events = loop.working_memory._events
    assistant_events = [e for e in events if e.event_type == "ASSISTANT"]
    assert len(assistant_events) >= 1, "Braked response must be stored in working_memory"
    assert "Braked response" in assistant_events[-1].content
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_agent_loop.py::test_self_brake_stores_in_working_memory -v`
Expected: FAIL — the braked response is NOT in working_memory because `break` at line 355 skips `add_event` at line 357.

**Step 3: Write minimal implementation**

In `sage-python/src/sage/agent_loop.py`, change lines 350-357 from:

```python
            # CGRS: stop if converged
            if brake:
                log.info("CGRS self-brake triggered — stopping reasoning loop.")
                result_text = content
                messages.append(Message(role=Role.ASSISTANT, content=content))
                break

            self.working_memory.add_event("ASSISTANT", content)
```

to:

```python
            # CGRS: stop if converged
            if brake:
                log.info("CGRS self-brake triggered — stopping reasoning loop.")
                result_text = content
                self.working_memory.add_event("ASSISTANT", content)
                messages.append(Message(role=Role.ASSISTANT, content=content))
                break

            self.working_memory.add_event("ASSISTANT", content)
```

**Step 4: Run tests to verify they pass**

Run: `cd sage-python && python -m pytest tests/test_agent_loop.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add sage-python/src/sage/agent_loop.py sage-python/tests/test_agent_loop.py
git commit -m "fix(agent_loop): store braked response in working_memory before break"
```

---

### Task 2: Split retry counters (S2/S3)

**Files:**
- Modify: `sage-python/src/sage/agent_loop.py:112-118, 254-348`
- Test: `sage-python/tests/test_agent_loop.py`

**Step 1: Write the failing test**

Add to `sage-python/tests/test_agent_loop.py`:

```python
def test_separate_retry_counters_exist():
    """S2 and S3 must have independent retry counters."""
    from sage.agent import AgentConfig
    from sage.llm.base import LLMConfig
    from sage.llm.mock import MockProvider

    config = AgentConfig(
        name="test", llm=LLMConfig(provider="mock", model="mock"),
        max_steps=3, validation_level=1,
    )
    loop = AgentLoop(config=config, llm_provider=MockProvider())

    assert hasattr(loop, '_s3_retries'), "Must have _s3_retries counter"
    assert hasattr(loop, '_s2_avr_retries'), "Must have _s2_avr_retries counter"
    assert hasattr(loop, '_max_s3_retries'), "Must have _max_s3_retries"
    assert hasattr(loop, '_max_s2_avr_retries'), "Must have _max_s2_avr_retries"
    assert loop._s3_retries == 0
    assert loop._s2_avr_retries == 0
    # Old shared counter must be gone
    assert not hasattr(loop, '_prm_retries'), "_prm_retries must be removed"
    assert not hasattr(loop, '_max_prm_retries'), "_max_prm_retries must be removed"
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_agent_loop.py::test_separate_retry_counters_exist -v`
Expected: FAIL — `_prm_retries` exists, `_s3_retries` does not.

**Step 3: Write minimal implementation**

In `sage-python/src/sage/agent_loop.py`, replace lines 117-118:

```python
        self._prm_retries = 0
        self._max_prm_retries = 2
```

with:

```python
        self._s3_retries = 0
        self._max_s3_retries = 2
        self._s2_avr_retries = 0
        self._max_s2_avr_retries = 3
```

In `run()` at line 143, replace `self._prm_retries = 0` with:

```python
        self._s3_retries = 0
        self._s2_avr_retries = 0
```

In the S3 validation block (lines 254-277), replace all `self._prm_retries` with `self._s3_retries` and `self._max_prm_retries` with `self._max_s3_retries`.

In the S2 validation block (lines 279-348), replace all `self._prm_retries` with `self._s2_avr_retries` and `self._max_prm_retries` with `self._max_s2_avr_retries`. In the escalation check at line 334, change `self._prm_retries > self._max_prm_retries` to `self._s2_avr_retries > self._max_s2_avr_retries`, and reset `self._s3_retries = 0` (not `self._prm_retries = 0`) on escalation.

**Step 4: Run tests to verify they pass**

Run: `cd sage-python && python -m pytest tests/ -v`
Expected: ALL 182+ PASS (no regressions)

**Step 5: Commit**

```bash
git add sage-python/src/sage/agent_loop.py sage-python/tests/test_agent_loop.py
git commit -m "fix(agent_loop): split shared _prm_retries into independent S2/S3 counters"
```

---

### Task 3: Async metacognition in loop

**Files:**
- Modify: `sage-python/src/sage/agent_loop.py:149-160`
- Test: `sage-python/tests/test_agent_loop.py`

**Step 1: Write the failing test**

Add to `sage-python/tests/test_agent_loop.py`:

```python
@pytest.mark.asyncio
async def test_loop_uses_async_metacognition():
    """Agent loop must call assess_complexity_async, not sync assess_complexity."""
    from sage.agent import AgentConfig
    from sage.llm.base import LLMConfig
    from sage.llm.mock import MockProvider
    from sage.strategy.metacognition import MetacognitiveController
    from unittest.mock import AsyncMock, patch

    ctrl = MetacognitiveController()
    provider = MockProvider(responses=["Done."])
    config = AgentConfig(
        name="test", llm=LLMConfig(provider="mock", model="mock"),
        max_steps=3, validation_level=1,
    )
    loop = AgentLoop(config=config, llm_provider=provider)
    loop.metacognition = ctrl

    events = []
    loop._on_event = events.append

    # Patch the async method to track calls
    ctrl.assess_complexity_async = AsyncMock(return_value=ctrl._assess_heuristic("test"))

    await loop.run("test task")

    ctrl.assess_complexity_async.assert_called_once()
    # Check routing_source is emitted
    perceive_events = [e for e in events if e.phase == LoopPhase.PERCEIVE]
    assert len(perceive_events) >= 1
    assert "routing_source" in perceive_events[0].data
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_agent_loop.py::test_loop_uses_async_metacognition -v`
Expected: FAIL — loop calls sync `assess_complexity()`, not `assess_complexity_async()`.

**Step 3: Write minimal implementation**

In `sage-python/src/sage/agent_loop.py`, replace lines 149-160:

```python
        # Metacognitive routing (if wired)
        if self.metacognition:
            profile = self.metacognition.assess_complexity(task)
            decision = self.metacognition.route(profile)
            perceive_meta["system"] = decision.system
            perceive_meta["routed_tier"] = decision.llm_tier
            perceive_meta["use_z3"] = decision.use_z3
            perceive_meta["validation_level"] = decision.validation_level
            perceive_meta["complexity"] = round(profile.complexity, 2)
            perceive_meta["uncertainty"] = round(profile.uncertainty, 2)
            if profile.reasoning:
                perceive_meta["routing_reason"] = profile.reasoning
```

with:

```python
        # Metacognitive routing (if wired)
        if self.metacognition:
            profile = await self.metacognition.assess_complexity_async(task)
            decision = self.metacognition.route(profile)
            perceive_meta["system"] = decision.system
            perceive_meta["routed_tier"] = decision.llm_tier
            perceive_meta["use_z3"] = decision.use_z3
            perceive_meta["validation_level"] = decision.validation_level
            perceive_meta["complexity"] = round(profile.complexity, 2)
            perceive_meta["uncertainty"] = round(profile.uncertainty, 2)
            perceive_meta["routing_source"] = "llm" if profile.reasoning != "heuristic" else "heuristic"
            if profile.reasoning:
                perceive_meta["routing_reason"] = profile.reasoning
```

**Step 4: Run tests to verify they pass**

Run: `cd sage-python && python -m pytest tests/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add sage-python/src/sage/agent_loop.py sage-python/tests/test_agent_loop.py
git commit -m "fix(agent_loop): use async metacognition assessment with routing_source tracking"
```

---

### Task 4: S2 AVR validate last code block

**Files:**
- Modify: `sage-python/src/sage/agent_loop.py:281-288`
- Test: `sage-python/tests/test_agent_loop.py`

**Step 1: Write the failing test**

Add to `sage-python/tests/test_agent_loop.py`:

```python
def test_extract_code_blocks_returns_multiple():
    """_extract_code_blocks must return ALL blocks from multi-block content."""
    from sage.agent_loop import _extract_code_blocks

    content = '''Here is my first attempt:
```python
x = 1
```
Actually, let me fix that:
```python
x = 2
```
'''
    blocks = _extract_code_blocks(content)
    assert len(blocks) == 2
    assert "x = 1" in blocks[0]
    assert "x = 2" in blocks[1]


@pytest.mark.asyncio
async def test_s2_avr_validates_last_block():
    """S2 AVR must validate the LAST code block, not the first."""
    from sage.agent import AgentConfig
    from sage.llm.base import LLMConfig
    from sage.llm.mock import MockProvider
    from sage.sandbox.manager import SandboxManager
    from unittest.mock import AsyncMock, MagicMock

    # Response with two blocks: first fails, last succeeds
    content = '```python\nraise Exception("bad")\n```\nFixed:\n```python\nprint("ok")\n```'
    provider = MockProvider(responses=[content])
    config = AgentConfig(
        name="test", llm=LLMConfig(provider="mock", model="mock"),
        max_steps=3, validation_level=2,
    )
    loop = AgentLoop(config=config, llm_provider=provider)

    # Mock sandbox that records what code was executed
    executed_code = []
    mock_sandbox = AsyncMock()
    mock_result = MagicMock()
    mock_result.exit_code = 0
    mock_result.stdout = "ok"
    mock_result.stderr = ""
    mock_sandbox.execute = AsyncMock(side_effect=lambda cmd: (executed_code.append(cmd), mock_result)[1])
    mock_sandbox.id = "test-sandbox"

    mock_manager = AsyncMock(spec=SandboxManager)
    mock_manager.create = AsyncMock(return_value=mock_sandbox)
    mock_manager.destroy = AsyncMock()
    loop.sandbox_manager = mock_manager

    events = []
    loop._on_event = events.append
    await loop.run("test task")

    # The LAST block ('print("ok")') should have been executed, not the first
    assert len(executed_code) == 1
    assert 'print("ok")' in executed_code[0]
    assert 'raise Exception' not in executed_code[0]
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_agent_loop.py::test_s2_avr_validates_last_block -v`
Expected: FAIL — current code uses `code_blocks[0]` which contains the first (bad) block.

**Step 3: Write minimal implementation**

In `sage-python/src/sage/agent_loop.py`, change line 288 from:

```python
                        result = await sandbox.execute(
                            f"python3 -c {_shell_quote(code_blocks[0])}"
                        )
```

to:

```python
                        avr_block_idx = len(code_blocks) - 1
                        result = await sandbox.execute(
                            f"python3 -c {_shell_quote(code_blocks[avr_block_idx])}"
                        )
```

**Step 4: Run tests to verify they pass**

Run: `cd sage-python && python -m pytest tests/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add sage-python/src/sage/agent_loop.py sage-python/tests/test_agent_loop.py
git commit -m "fix(agent_loop): S2 AVR validates last code block instead of first"
```

---

### Task 5: AgentEvent schema dataclass

**Files:**
- Modify: `sage-python/src/sage/agent_loop.py` (add dataclass near top, after `LoopEvent`)
- Test: `sage-python/tests/test_agent_loop.py`

**Step 1: Write the failing test**

Add to `sage-python/tests/test_agent_loop.py`:

```python
def test_agent_event_schema():
    """AgentEvent must have structured fields with schema_version."""
    from sage.agent_loop import AgentEvent

    evt = AgentEvent(
        type="THINK",
        step=1,
        timestamp=1234567890.0,
        latency_ms=42.5,
        cost_usd=0.001,
        model="gemini-3.1-flash-lite-preview",
        system=1,
        routing_source="heuristic",
    )
    assert evt.schema_version == 1
    assert evt.type == "THINK"
    assert evt.latency_ms == 42.5
    assert evt.meta == {}


def test_agent_event_to_dict():
    """AgentEvent must serialize to dict for JSONL output."""
    from sage.agent_loop import AgentEvent
    import dataclasses

    evt = AgentEvent(type="PERCEIVE", step=0, timestamp=0.0, system=2)
    d = dataclasses.asdict(evt)
    assert d["schema_version"] == 1
    assert d["type"] == "PERCEIVE"
    assert d["system"] == 2
    assert d["latency_ms"] is None  # Optional fields default to None
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_agent_loop.py::test_agent_event_schema -v`
Expected: FAIL — `ImportError: cannot import name 'AgentEvent'`

**Step 3: Write minimal implementation**

In `sage-python/src/sage/agent_loop.py`, add after the `LoopEvent` dataclass (after line 83):

```python
@dataclass
class AgentEvent:
    """Versioned structured event for observability (v1)."""
    type: str                           # PERCEIVE, THINK, ACT, LEARN
    step: int
    timestamp: float
    schema_version: int = 1
    latency_ms: float | None = None
    cost_usd: float | None = None
    tokens_est: int | None = None
    model: str | None = None
    system: int | None = None           # 1, 2, or 3
    routing_source: str | None = None   # "llm" or "heuristic"
    validation: str | None = None       # s2_avr_pass, s2_avr_fail, s3_prm_pass, etc.
    meta: dict[str, Any] = field(default_factory=dict)
```

**Step 4: Run tests to verify they pass**

Run: `cd sage-python && python -m pytest tests/test_agent_loop.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add sage-python/src/sage/agent_loop.py sage-python/tests/test_agent_loop.py
git commit -m "feat(agent_loop): add AgentEvent versioned schema dataclass"
```

---

### Task 6: Migrate _emit() to structured events

**Files:**
- Modify: `sage-python/src/sage/agent_loop.py:120-137` (event handler), and all `_emit()` calls throughout `run()`
- Test: `sage-python/tests/test_agent_loop.py`

**Step 1: Write the failing test**

Add to `sage-python/tests/test_agent_loop.py`:

```python
@pytest.mark.asyncio
async def test_events_contain_schema_version():
    """All emitted events must include schema_version=1."""
    from sage.agent import AgentConfig
    from sage.llm.base import LLMConfig
    from sage.llm.mock import MockProvider
    from sage.agent_loop import AgentEvent

    provider = MockProvider(responses=["Done."])
    config = AgentConfig(
        name="test", llm=LLMConfig(provider="mock", model="mock"),
        max_steps=3, validation_level=1,
    )
    agent_events: list[AgentEvent] = []
    loop = AgentLoop(config=config, llm_provider=provider,
                     on_event=agent_events.append)
    await loop.run("test task")

    assert len(agent_events) >= 2  # At least PERCEIVE + THINK
    for evt in agent_events:
        assert isinstance(evt, AgentEvent), f"Expected AgentEvent, got {type(evt)}"
        assert evt.schema_version == 1
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_agent_loop.py::test_events_contain_schema_version -v`
Expected: FAIL — current `_emit` produces `LoopEvent`, not `AgentEvent`.

**Step 3: Write minimal implementation**

In `sage-python/src/sage/agent_loop.py`:

1. Change `__init__` signature: `on_event: Callable[[AgentEvent], None] | None = None`

2. Replace `_emit` method (lines 120-122):

```python
    def _emit(self, phase: LoopPhase, **data: Any) -> None:
        evt = AgentEvent(
            type=phase.value.upper(),
            step=self.step_count,
            timestamp=time.time(),
            latency_ms=data.pop("latency_ms", None),
            cost_usd=data.pop("cost_usd", None),
            tokens_est=data.pop("tokens_est", None),
            model=data.pop("model", None),
            system=data.pop("system", None),
            routing_source=data.pop("routing_source", None),
            validation=data.pop("validation", None),
            meta=data,
        )
        self._on_event(evt)
```

3. Replace `_default_event_handler` (lines 124-137):

```python
    def _default_event_handler(self, event: AgentEvent) -> None:
        log.info(f"[{event.type}] step={event.step} model={event.model}")
        try:
            STREAM_FILE.parent.mkdir(parents=True, exist_ok=True)
            import dataclasses
            with open(STREAM_FILE, "a", encoding="utf-8") as f:
                d = dataclasses.asdict(event)
                f.write(json.dumps(d, default=str) + "\n")
        except Exception:
            pass
```

4. Update all existing tests that reference `LoopEvent` or `event.phase` to use `AgentEvent` and `event.type`. Specifically, update `test_loop_event_structure` and `test_agent_loop_emits_events` to use `AgentEvent`.

**Step 4: Run tests to verify they pass**

Run: `cd sage-python && python -m pytest tests/ -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add sage-python/src/sage/agent_loop.py sage-python/tests/test_agent_loop.py
git commit -m "feat(agent_loop): migrate event emission to structured AgentEvent schema"
```

---

### Task 7: Create models.toml + config loader

**Files:**
- Create: `sage-python/config/models.toml`
- Create: `sage-python/src/sage/llm/config_loader.py`
- Test: `sage-python/tests/test_config_loader.py`

**Step 1: Write the failing test**

Create `sage-python/tests/test_config_loader.py`:

```python
import sys, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest
from pathlib import Path


def test_load_models_from_toml(tmp_path):
    """Load model config from a TOML file."""
    from sage.llm.config_loader import load_model_config

    toml_content = '''
[tiers]
fast = "gemini-test-fast"
codex = "gpt-test-codex"

[defaults]
temperature = 0.5
'''
    config_file = tmp_path / "models.toml"
    config_file.write_text(toml_content)

    config = load_model_config(config_file)
    assert config["tiers"]["fast"] == "gemini-test-fast"
    assert config["tiers"]["codex"] == "gpt-test-codex"
    assert config["defaults"]["temperature"] == 0.5


def test_load_models_returns_empty_on_missing():
    """Missing file returns empty dict, not error."""
    from sage.llm.config_loader import load_model_config

    config = load_model_config(Path("/nonexistent/models.toml"))
    assert config == {}


def test_env_override():
    """Environment variables override TOML values."""
    import os
    from sage.llm.config_loader import resolve_model_id

    os.environ["SAGE_MODEL_FAST"] = "override-model"
    try:
        result = resolve_model_id("fast", toml_tiers={"fast": "toml-model"})
        assert result == "override-model"
    finally:
        del os.environ["SAGE_MODEL_FAST"]


def test_toml_fallback():
    """TOML value used when no env var is set."""
    import os
    from sage.llm.config_loader import resolve_model_id

    os.environ.pop("SAGE_MODEL_FAST", None)
    result = resolve_model_id("fast", toml_tiers={"fast": "toml-model"})
    assert result == "toml-model"


def test_hardcoded_fallback():
    """Hardcoded default used when no env var or TOML value."""
    import os
    from sage.llm.config_loader import resolve_model_id

    os.environ.pop("SAGE_MODEL_FAST", None)
    result = resolve_model_id("fast", toml_tiers={}, hardcoded="default-model")
    assert result == "default-model"
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_config_loader.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_model_config'`

**Step 3: Write minimal implementation**

Create `sage-python/src/sage/llm/config_loader.py`:

```python
"""Model config loader: TOML file + env var overrides."""
from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any


def load_model_config(path: Path) -> dict[str, Any]:
    """Load model configuration from a TOML file. Returns {} if missing."""
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return {}


def resolve_model_id(
    tier: str,
    toml_tiers: dict[str, str] | None = None,
    hardcoded: str | None = None,
) -> str | None:
    """Resolve model ID: env var > TOML > hardcoded default."""
    env_key = f"SAGE_MODEL_{tier.upper()}"
    env_val = os.environ.get(env_key)
    if env_val:
        return env_val
    if toml_tiers and tier in toml_tiers:
        return toml_tiers[tier]
    return hardcoded
```

Create `sage-python/config/models.toml`:

```toml
# YGN-SAGE Model Configuration
# Override per-tier model IDs here or via env vars (SAGE_MODEL_FAST, etc.)

[tiers]
fast = "gemini-3.1-flash-lite-preview"
mutator = "gemini-3-flash-preview"
reasoner = "gemini-3.1-pro-preview"
codex = "gpt-5.3-codex"
codex_max = "gpt-5.2"
budget = "gemini-2.5-flash-lite"
fallback = "gemini-2.5-flash"

[defaults]
temperature = 0.7
```

**Step 4: Run tests to verify they pass**

Run: `cd sage-python && python -m pytest tests/test_config_loader.py -v`
Expected: ALL 5 PASS

**Step 5: Commit**

```bash
git add sage-python/src/sage/llm/config_loader.py sage-python/config/models.toml sage-python/tests/test_config_loader.py
git commit -m "feat(router): add models.toml config loader with env var overrides"
```

---

### Task 8: Wire ModelRouter to TOML + env overrides

**Files:**
- Modify: `sage-python/src/sage/llm/router.py`
- Test: `sage-python/tests/test_router.py` (new)

**Step 1: Write the failing test**

Create `sage-python/tests/test_router.py`:

```python
import sys, types
if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = types.ModuleType("sage_core")

import pytest
import os


def test_router_loads_default_models():
    """ModelRouter must have MODELS dict populated."""
    from sage.llm.router import ModelRouter
    assert "fast" in ModelRouter.MODELS
    assert "codex" in ModelRouter.MODELS


def test_router_get_config_returns_llmconfig():
    """get_config must return LLMConfig with correct model."""
    from sage.llm.router import ModelRouter
    config = ModelRouter.get_config("fast")
    assert config.model == ModelRouter.MODELS["fast"]
    assert config.provider == "google"


def test_router_env_override():
    """Env var SAGE_MODEL_FAST overrides the MODELS dict."""
    from sage.llm.router import ModelRouter
    original = ModelRouter.MODELS.get("fast")
    os.environ["SAGE_MODEL_FAST"] = "test-override-model"
    try:
        # Force reload
        ModelRouter._load_config()
        assert ModelRouter.MODELS["fast"] == "test-override-model"
        config = ModelRouter.get_config("fast")
        assert config.model == "test-override-model"
    finally:
        os.environ.pop("SAGE_MODEL_FAST", None)
        ModelRouter.MODELS["fast"] = original


def test_router_codex_provider():
    """Codex tiers must return provider='codex'."""
    from sage.llm.router import ModelRouter
    config = ModelRouter.get_config("codex")
    assert config.provider == "codex"


def test_router_critical_is_reasoner():
    """'critical' tier is an alias for 'reasoner'."""
    from sage.llm.router import ModelRouter
    config = ModelRouter.get_config("critical")
    assert config.model == ModelRouter.MODELS["reasoner"]
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_router.py -v`
Expected: FAIL on `test_router_env_override` — `ModelRouter` has no `_load_config` method.

**Step 3: Write minimal implementation**

Replace `sage-python/src/sage/llm/router.py` entirely:

```python
"""SOTA March 2026 Model Router for YGN-SAGE.

Model IDs loaded from: env vars > config/models.toml > hardcoded defaults.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from sage.llm.base import LLMConfig
from sage.llm.config_loader import load_model_config, resolve_model_id

Tier = Literal[
    "fast", "mutator", "reasoner", "codex", "codex_max",
    "budget", "critical", "fallback",
]

# Hardcoded defaults (used when no TOML or env var)
_HARDCODED = {
    "codex": "gpt-5.3-codex",
    "codex_max": "gpt-5.2",
    "fast": "gemini-3.1-flash-lite-preview",
    "mutator": "gemini-3-flash-preview",
    "reasoner": "gemini-3.1-pro-preview",
    "budget": "gemini-2.5-flash-lite",
    "fallback": "gemini-2.5-flash",
}

# Max tokens per tier
_MAX_TOKENS = {
    "codex": 8192,
    "codex_max": 16384,
    "reasoner": 8192,
    "critical": 8192,
    "mutator": 4096,
    "budget": 2048,
    "fallback": 4096,
    "fast": 4096,
}

# Tiers that use Codex CLI provider
_CODEX_TIERS = {"codex", "codex_max"}


class ModelRouter:
    """Routes requests to the optimal model for each task type."""

    MODELS: dict[str, str] = {}

    @classmethod
    def _load_config(cls) -> None:
        """Load model IDs from TOML + env vars, falling back to hardcoded."""
        # Search for models.toml
        toml_tiers: dict[str, str] = {}
        for search_dir in [
            Path.cwd() / "config",
            Path(__file__).parent.parent.parent.parent / "config",  # sage-python/config/
            Path.home() / ".sage",
        ]:
            toml_path = search_dir / "models.toml"
            cfg = load_model_config(toml_path)
            if cfg:
                toml_tiers = cfg.get("tiers", {})
                break

        for tier, hardcoded in _HARDCODED.items():
            cls.MODELS[tier] = resolve_model_id(tier, toml_tiers, hardcoded) or hardcoded

    @staticmethod
    def get_config(
        tier: Tier = "fast",
        temperature: float = 0.7,
        json_schema: type | dict | None = None,
    ) -> LLMConfig:
        if not ModelRouter.MODELS:
            ModelRouter._load_config()

        # Resolve "critical" alias
        lookup_tier = "reasoner" if tier == "critical" else tier
        model = ModelRouter.MODELS.get(lookup_tier, _HARDCODED.get(lookup_tier, _HARDCODED["fast"]))
        provider = "codex" if tier in _CODEX_TIERS else "google"
        max_tokens = _MAX_TOKENS.get(tier, 4096)

        extra = {}
        if tier == "codex_max":
            extra["reasoning_effort"] = "xhigh"

        return LLMConfig(
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            json_schema=json_schema,
            extra=extra if extra else None,
        )


# Load on import
ModelRouter._load_config()
```

**Step 4: Run tests to verify they pass**

Run: `cd sage-python && python -m pytest tests/ -v`
Expected: ALL PASS (including existing tests that use `ModelRouter.get_config`)

**Step 5: Commit**

```bash
git add sage-python/src/sage/llm/router.py sage-python/tests/test_router.py
git commit -m "feat(router): wire ModelRouter to models.toml with env var overrides"
```

---

### Task 9: GitHub Actions CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

**Step 1: Write the workflow file**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [master]
  pull_request:
    branches: [master]

jobs:
  rust:
    name: Rust (cargo test + clippy)
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: sage-core
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy, rustfmt
      - run: cargo fmt --check
      - run: cargo clippy -- -D warnings
      - run: cargo test

  python-sage:
    name: Python SDK (sage-python)
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: sage-python
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e ".[all,dev]"
      - run: ruff check src/
      - run: python -m pytest tests/ -v --tb=short

  python-discover:
    name: Python Discovery (sage-discover)
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: sage-discover
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -e "../sage-python[all,dev]"
      - run: pip install -e "."
      - run: python -m pytest tests/ -v --tb=short
```

**Step 2: Verify syntax**

Run: `cd C:/Code/YGN-SAGE && cat .github/workflows/ci.yml | python -c "import sys,yaml; yaml.safe_load(sys.stdin)" 2>/dev/null || echo "Install pyyaml to validate, or trust the structure"`

**Step 3: Commit**

```bash
mkdir -p .github/workflows
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow for Rust + Python tests"
```

---

### Task 10: Gitignore + delete stale artifacts

**Files:**
- Modify: `.gitignore`
- Delete: `memory-bank/` (7 files)
- Delete: `research_journal/deep_self_awareness_report.md`

**Step 1: Update .gitignore**

Add to `.gitignore`:

```
# Auto-generated evolution logs (keep .md files)
research_journal/H*.json

# Agent stream (regenerated on each run)
docs/plans/agent_stream.jsonl
```

**Step 2: Delete stale files**

```bash
rm -rf memory-bank/
rm -f research_journal/deep_self_awareness_report.md
git rm -r memory-bank/ 2>/dev/null || true
git rm research_journal/deep_self_awareness_report.md 2>/dev/null || true
```

**Step 3: Remove H*.json from tracking**

```bash
git rm --cached research_journal/H*.json 2>/dev/null || true
git rm --cached docs/plans/agent_stream.jsonl 2>/dev/null || true
```

**Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore evolution logs, remove stale Cline artifacts + AI roleplay doc"
```

---

### Task 11: License alignment Cargo.toml

**Files:**
- Modify: `sage-core/Cargo.toml:1-4`

**Step 1: Add license field**

In `sage-core/Cargo.toml`, after `edition = "2021"` (line 4), add:

```toml
license = "Proprietary"
```

**Step 2: Verify build**

Run: `cd sage-core && cargo check 2>&1 | head -5`
Expected: No errors (cargo accepts `license = "Proprietary"` — it's a free-form string for non-SPDX).

Note: Cargo will warn that "Proprietary" is not a valid SPDX identifier. Use `license-file = "../LICENSE"` if a LICENSE file exists, or accept the warning. Since README says "Proprietary. All rights reserved." and there's no LICENSE file, the `license` field is informational only.

**Step 3: Commit**

```bash
git add sage-core/Cargo.toml
git commit -m "chore: align Cargo.toml license with README (Proprietary)"
```

---

### Task 12: Conductor ASI language cleanup

**Files:**
- Modify: `conductor/tracks/asi_convergence/plan.md`

**Step 1: Rewrite the file**

Replace the content of `conductor/tracks/asi_convergence/plan.md` with factual engineering language:

```markdown
# Advanced Architecture Plan (2026+)

Engineering track for refactoring YGN-SAGE core pillars based on research synthesis.

## Phase 1: Cognitive Memory Refactoring (S-MMU)
- [x] **Task 1: TierMem Architecture**
  - Two-tier memory hierarchy in `sage-core`: active buffer (fast, mutable) + Arrow chunks (immutable, columnar).
  - Rust-based DAG (`petgraph`) for multi-view semantic memory routing.
- [x] **Task 2: Context-Aware Memory (MEM1)**
  - Per-step internal state generation via `MemoryCompressor.generate_internal_state()`.
  - Active forgetting: semantic node updates replace linear context appending.

## Phase 2: Neuro-Symbolic Execution
- [x] **Task 3: Sub-millisecond Sandboxing**
  - eBPF executor (`solana_rbpf`) for evolution engine bytecode evaluation.
  - SnapBPF CoW memory snapshots for stateful rollback.
  - WASM sandbox (`wasmtime`) for general code execution.
- [x] **Task 4: Z3 Validation (Process Reward Model)**
  - Z3-based assertion verification in `<think>` blocks.
  - 4 assertion types: `bounds`, `loop`, `arithmetic`, `invariant`.

## Phase 3: Dynamic Topology
- [x] **Task 5: Topology Planner**
  - S-DTS (Stochastic Differentiable Tree Search) for agent DAG generation.
  - MAP-Elites evolutionary topology search.

## Phase 4: Evolution Engine
- [x] **Task 6: DGM + SAMPO**
  - MAP-Elites with 5 strategic actions (optimize, fix, explore, constrain, simplify).
  - Sequence-level policy optimization (SAMPO clipping).
- [x] **Task 7: KG-RLVR Process Reward Model**
  - R_path scoring of compositional reasoning via knowledge graph verification.
```

**Step 2: Commit**

```bash
git add conductor/tracks/asi_convergence/plan.md
git commit -m "docs: rewrite conductor ASI plan with factual engineering language"
```

---

## Regression Check

After all 12 tasks, run the full test suite:

```bash
cd sage-python && python -m pytest tests/ -v
cd ../sage-discover && python -m pytest tests/ -v
cd ../sage-core && cargo test
```

Expected: 182+ sage-python, 45 sage-discover, 38 Rust = 265+ total (plus new tests from tasks 1-8).
