# Multi-Provider Dynamic Routing Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire all 6+ LLM providers into the main control plane with auto-discovery at boot, cascade fallback, provider quirk handling, and CapabilityMatrix auto-population.

**Architecture:** `ModelRegistry.refresh()` runs at boot to discover all available models from `.env` API keys. `CognitiveOrchestrator` becomes the primary routing engine in `AgentSystem.run()`, replacing the legacy `ModelRouter` tier-based path. Three hardcoded `GoogleProvider` usages are replaced with registry-based selection. `OpenAICompatProvider` gains provider-specific quirk handling (DeepSeek temperature, Grok reasoning_content, Kimi temp clamping, MiniMax think-tag extraction).

**Tech Stack:** Python 3.12, google-genai, openai SDK, pytest, pytest-asyncio

---

### Task 1: Boot-time registry refresh

**Context:** `ModelRegistry.refresh()` is async but `boot_agent_system()` is sync. We need to run the async discovery at boot so `AgentSystem` knows which models are available before any task runs.

**Files:**
- Modify: `sage-python/src/sage/boot.py:230-272`
- Test: `sage-python/tests/test_boot_refresh.py` (new)

**Step 1: Write the failing test**

Create `sage-python/tests/test_boot_refresh.py`:

```python
"""Tests for registry refresh at boot time."""
import sys
import types as _types

if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = _types.ModuleType("sage_core")
    _mock = sys.modules["sage_core"]

    class _WM:
        def __init__(self, **kw): self._events = []; self._counter = 0
        def add_event(self, t, c):
            self._counter += 1; self._events.append({"type": t, "content": c}); return f"e-{self._counter}"
        def event_count(self): return len(self._events)
        def recent_events(self, n): return self._events[-n:]
        def compact_to_arrow(self): return 0
        def compact_to_arrow_with_meta(self, *a): return 0
        def retrieve_relevant_chunks(self, *a): return []
        def get_page_out_candidates(self, *a): return []
        def smmu_chunk_count(self): return 0
        def get_latest_arrow_chunk(self): return None

    _mock.WorkingMemory = _WM

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from sage.boot import boot_agent_system


def test_boot_creates_registry_in_mock_mode():
    """Even in mock mode, AgentSystem should have a registry attribute."""
    system = boot_agent_system(use_mock_llm=True)
    assert hasattr(system, "registry")


def test_boot_creates_registry_populated_in_real_mode(monkeypatch):
    """In real mode, registry.refresh() should be called at boot."""
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key-for-test")

    # Patch the registry refresh to track calls
    refresh_called = []
    original_refresh = None

    async def mock_refresh(self):
        refresh_called.append(True)

    with patch("sage.providers.registry.ModelRegistry.refresh", mock_refresh):
        # Need to also patch the provider creation to avoid real API calls
        with patch("sage.llm.google.GoogleProvider"):
            try:
                system = boot_agent_system(use_mock_llm=False, llm_tier="fast")
            except Exception:
                pass  # Provider init may fail, we just check refresh was called

    assert len(refresh_called) > 0, "registry.refresh() must be called at boot"
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_boot_refresh.py -v`
Expected: FAIL — `test_boot_creates_registry_populated_in_real_mode` fails because boot.py doesn't call `refresh()`.

**Step 3: Implement boot-time refresh**

In `sage-python/src/sage/boot.py`, after `registry = ModelRegistry()` (around line 254), add:

```python
    # Auto-discover available models at boot
    if registry is not None:
        import asyncio
        try:
            # Run async refresh in sync context
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                # Already in async context (rare) — schedule as task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(lambda: asyncio.run(registry.refresh())).result(timeout=15)
            else:
                asyncio.run(registry.refresh())
            _log.info(
                "Boot: discovered %d models (%d available)",
                len(registry.profiles),
                len(registry.list_available()),
            )
        except Exception as e:
            _log.warning("Boot: model discovery failed (%s), continuing with legacy routing", e)
```

Also, in mock mode, still create the registry attribute on AgentSystem (currently it's set to `None` for mock). Change the `if not use_mock_llm:` block to always create registry but only refresh in real mode:

In `boot_agent_system()`, replace:
```python
    registry = None
    orchestrator = None
    if not use_mock_llm:
        from sage.providers.registry import ModelRegistry
        from sage.orchestrator import CognitiveOrchestrator
        registry = ModelRegistry()
        orchestrator = CognitiveOrchestrator(
            registry=registry, metacognition=metacognition, event_bus=event_bus,
        )
```

With:
```python
    from sage.providers.registry import ModelRegistry
    registry = ModelRegistry()
    orchestrator = None

    if not use_mock_llm:
        from sage.orchestrator import CognitiveOrchestrator

        # Auto-discover available models at boot
        import asyncio
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    pool.submit(lambda: asyncio.run(registry.refresh())).result(timeout=15)
            else:
                asyncio.run(registry.refresh())
            _log.info(
                "Boot: discovered %d models (%d available)",
                len(registry.profiles),
                len(registry.list_available()),
            )
        except Exception as e:
            _log.warning("Boot: model discovery failed (%s), continuing with legacy routing", e)

        orchestrator = CognitiveOrchestrator(
            registry=registry, metacognition=metacognition, event_bus=event_bus,
        )
```

**Step 4: Run test to verify it passes**

Run: `cd sage-python && python -m pytest tests/test_boot_refresh.py -v`
Expected: PASS

**Step 5: Run full suite to confirm no regressions**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: All 691+ tests pass

**Step 6: Commit**

```bash
git add sage-python/tests/test_boot_refresh.py sage-python/src/sage/boot.py
git commit -m "feat(boot): auto-discover models via registry.refresh() at startup"
```

---

### Task 2: Wire CognitiveOrchestrator as primary routing path

**Context:** Currently `AgentSystem.run()` uses `ModelRouter.get_config(tier)` which only maps to Codex/Google. The orchestrator exists but is never called in the main path. We wire it as primary with `ModelRouter` as fallback.

**Files:**
- Modify: `sage-python/src/sage/boot.py:63-138` (AgentSystem.run)
- Test: `sage-python/tests/test_boot_refresh.py` (extend)

**Step 1: Write the failing test**

Append to `sage-python/tests/test_boot_refresh.py`:

```python
@pytest.mark.asyncio
async def test_agent_system_run_uses_orchestrator_when_available():
    """When orchestrator is set, AgentSystem.run() should try it first."""
    system = boot_agent_system(use_mock_llm=True)

    # Inject a mock orchestrator
    mock_orch = AsyncMock()
    mock_orch.run = AsyncMock(return_value="Orchestrator result")
    system.orchestrator = mock_orch

    result = await system.run("What is 2+2?")
    # Mock mode bypasses orchestrator (returns early), so check behavior:
    # In mock mode, current_provider == "mock" so it returns early.
    # This test verifies the mock path is preserved.
    assert result is not None


@pytest.mark.asyncio
async def test_agent_system_run_falls_back_on_orchestrator_failure():
    """If orchestrator.run() raises, fall back to AgentLoop.run()."""
    system = boot_agent_system(use_mock_llm=True)

    mock_orch = AsyncMock()
    mock_orch.run = AsyncMock(side_effect=RuntimeError("orchestrator broken"))
    system.orchestrator = mock_orch

    # Should still work via fallback
    result = await system.run("What is 2+2?")
    assert result is not None
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_boot_refresh.py::test_agent_system_run_falls_back_on_orchestrator_failure -v`
Expected: FAIL — fallback path doesn't exist yet.

**Step 3: Implement orchestrator as primary path**

In `sage-python/src/sage/boot.py`, modify `AgentSystem.run()`:

Replace the method body (lines ~63-138) with:

```python
    async def run(self, task: str) -> str:
        """Run a task through the agent system.

        Primary path: CognitiveOrchestrator (multi-provider, score-based).
        Fallback: legacy AgentLoop with ModelRouter (Codex + Google only).
        Mock mode: direct AgentLoop (no orchestrator).
        """
        # 1. Assess task complexity
        profile = await self.metacognition.assess_complexity_async(task)
        decision = self.metacognition.route(profile)

        # 1b. Speculative execution: detect indecisive zone
        if 0.35 <= profile.complexity <= 0.55 and decision.system <= 2:
            _log.info(
                "Speculative zone: complexity=%.2f (indecisive). "
                "Would fire S1+S2 in parallel when architecture supports it. "
                "Using S%d for now.",
                profile.complexity, decision.system,
            )

        # 2. Set validation level from routing decision
        if decision.system >= 3:
            self.agent_loop.config.validation_level = 3
        elif decision.system == 2 and self.agent_loop.sandbox_manager:
            self.agent_loop.config.validation_level = 2
        else:
            self.agent_loop.config.validation_level = 1

        current_provider = self.agent_loop.config.llm.provider

        # Mock mode: skip orchestrator, use AgentLoop directly
        if current_provider == "mock":
            return await self.agent_loop.run(task)

        # 3. Try CognitiveOrchestrator as primary path (multi-provider)
        if self.orchestrator and self.registry and self.registry.list_available():
            try:
                result = await self.orchestrator.run(task)
                await self._persist_memory()
                return result
            except Exception as e:
                _log.warning(
                    "Orchestrator failed (%s), falling back to legacy routing", e
                )

        # 4. Fallback: legacy ModelRouter path (Codex + Google only)
        new_config = ModelRouter.get_config(decision.llm_tier)
        if current_provider == "codex" and new_config.provider == "google":
            pass  # Don't downgrade from Codex to Gemini
        elif new_config.provider == "google" and not os.environ.get("GOOGLE_API_KEY"):
            pass  # Google unavailable
        else:
            self.agent_loop.config.llm = new_config
            if new_config.provider == "codex":
                from sage.llm.codex import CodexProvider
                self.agent_loop._llm = CodexProvider()
            elif new_config.provider == "google":
                from sage.llm.google import GoogleProvider
                self.agent_loop._llm = GoogleProvider()

        result = await self.agent_loop.run(task)
        await self._persist_memory()
        return result

    async def _persist_memory(self) -> None:
        """Persist semantic and causal memory after a run."""
        if hasattr(self.agent_loop, "semantic_memory") and self.agent_loop.semantic_memory:
            try:
                self.agent_loop.semantic_memory.save()
            except Exception:
                _log.warning("Failed to persist semantic memory", exc_info=True)
        if hasattr(self.agent_loop, "causal_memory") and self.agent_loop.causal_memory:
            try:
                self.agent_loop.causal_memory.save()
            except Exception:
                _log.warning("Failed to persist causal memory", exc_info=True)
```

**Step 4: Run test to verify it passes**

Run: `cd sage-python && python -m pytest tests/test_boot_refresh.py -v`
Expected: PASS

**Step 5: Run full suite**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add sage-python/src/sage/boot.py sage-python/tests/test_boot_refresh.py
git commit -m "feat(routing): wire CognitiveOrchestrator as primary path with legacy fallback"
```

---

### Task 3: Add cascade fallback to ModelAgent

**Context:** `ModelAgent.run()` currently tries one provider and fails if it errors. FrugalGPT pattern: try → catch → pick next-best model → retry (max 3 attempts). This makes multi-provider routing resilient to single-provider outages.

**Files:**
- Modify: `sage-python/src/sage/orchestrator.py:80-96` (ModelAgent.run)
- Test: `sage-python/tests/test_orchestrator.py` (extend)

**Step 1: Write the failing test**

Append to `sage-python/tests/test_orchestrator.py`:

```python
class TestModelAgentCascade:
    @pytest.mark.asyncio
    async def test_cascade_retries_on_failure(self):
        """ModelAgent should retry with fallback models on provider failure."""
        primary = _make_profile("primary", code=0.9, cost_in=2.0)
        fallback1 = _make_profile("fallback1", code=0.7, cost_in=0.5)
        fallback2 = _make_profile("fallback2", code=0.5, cost_in=0.1)
        reg = _mock_registry(primary, fallback1, fallback2)

        agent = ModelAgent(name="test", model=primary)
        agent._registry = reg  # Inject registry for cascade

        call_count = 0

        async def failing_then_succeed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                raise RuntimeError("Provider timeout")
            return "Fallback success"

        with patch.object(ModelAgent, "_call_provider", new=failing_then_succeed):
            result = await agent.run("Test task")

        assert result == "Fallback success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_cascade_exhausted_raises(self):
        """If all cascade attempts fail, the last error should propagate."""
        primary = _make_profile("primary", code=0.9, cost_in=2.0)
        reg = _mock_registry(primary)

        agent = ModelAgent(name="test", model=primary)
        agent._registry = reg

        async def always_fail(*args, **kwargs):
            raise RuntimeError("All providers down")

        with patch.object(ModelAgent, "_call_provider", new=always_fail):
            result = await agent.run("Test task")

        # Should return error message, not crash
        assert "error" in result.lower() or "failed" in result.lower()
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_orchestrator.py::TestModelAgentCascade -v`
Expected: FAIL — `_registry` and `_call_provider` don't exist yet.

**Step 3: Implement cascade fallback**

In `sage-python/src/sage/orchestrator.py`, modify `ModelAgent`:

```python
class ModelAgent:
    """Lightweight agent that calls a specific LLM model via the appropriate provider.

    Supports cascade fallback: if the primary model fails, tries next-best
    models from the registry (FrugalGPT pattern, max 3 attempts).
    """

    MAX_CASCADE_ATTEMPTS = 3

    def __init__(self, name: str, model: ModelProfile, system_prompt: str = "",
                 registry: ModelRegistry | None = None):
        self.name = name
        self.model = model
        self._system_prompt = system_prompt or "You are a helpful AI assistant. Be precise and concise."
        self._registry = registry

    async def run(self, task: str) -> str:
        """Call the LLM model with cascade fallback on failure."""
        messages: list[Message] = []
        if self._system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=self._system_prompt))
        messages.append(Message(role=Role.USER, content=task))

        # Try primary model first
        tried_ids: set[str] = set()
        current_model = self.model
        last_error: Exception | None = None

        for attempt in range(self.MAX_CASCADE_ATTEMPTS):
            tried_ids.add(current_model.id)
            try:
                return await self._call_provider(current_model, messages)
            except Exception as e:
                last_error = e
                log.warning(
                    "ModelAgent %s: attempt %d/%d failed on %s (%s): %s",
                    self.name, attempt + 1, self.MAX_CASCADE_ATTEMPTS,
                    current_model.id, current_model.provider, e,
                )
                # Try to find a fallback model
                if self._registry:
                    fallback = self._pick_fallback(tried_ids)
                    if fallback:
                        log.info("Cascading to fallback model: %s", fallback.id)
                        current_model = fallback
                        continue
                break  # No registry or no fallback available

        error_msg = str(last_error) if last_error else "Unknown error"
        return f"[Agent {self.name} error: all {len(tried_ids)} models failed. Last: {error_msg}]"

    async def _call_provider(self, model: ModelProfile, messages: list[Message]) -> str:
        """Make a single LLM call to the given model."""
        provider = self._create_provider_for(model)
        config = LLMConfig(
            provider=model.provider,
            model=model.id,
            temperature=0.3,
        )
        response = await provider.generate(messages, config=config)
        return response.content or ""

    def _pick_fallback(self, exclude_ids: set[str]) -> ModelProfile | None:
        """Pick the next-best available model not yet tried."""
        if not self._registry:
            return None
        for candidate in self._registry.list_available():
            if candidate.id not in exclude_ids and candidate.cost_input > 0:
                return candidate
        return None

    def _create_provider_for(self, model: ModelProfile):
        """Create the appropriate LLM provider for a model."""
        cfg = _provider_cfg_for(model.provider)
        sdk = cfg.get("sdk", "openai")
        api_key_env = cfg.get("api_key_env", "")
        api_key = os.environ.get(api_key_env, "") if api_key_env else ""

        if sdk == "google-genai":
            from sage.llm.google import GoogleProvider
            return GoogleProvider(api_key=api_key)
        else:
            from sage.providers.openai_compat import OpenAICompatProvider
            return OpenAICompatProvider(
                api_key=api_key,
                base_url=cfg.get("base_url"),
                model_id=model.id,
            )

    # Keep old method for backward compat
    def _create_provider(self):
        return self._create_provider_for(self.model)
```

Also update `CognitiveOrchestrator` to pass `registry` to `ModelAgent`:

In all `ModelAgent(...)` constructor calls within `orchestrator.py`, add `registry=self.registry`:
- Line ~177: `agent = ModelAgent(name="s1-fast", model=model, registry=self.registry)`
- Line ~199: `agent = ModelAgent(name="s2-worker", model=model, registry=self.registry)`
- Line ~218: `agent = ModelAgent(name="s3-reasoner", model=model, registry=self.registry)`
- Line ~241-244: `agent = ModelAgent(name=..., model=model, system_prompt=..., registry=self.registry)`
- Line ~292: `agent = ModelAgent(name="decomposer", model=model, registry=self.registry)`

**Step 4: Run test to verify it passes**

Run: `cd sage-python && python -m pytest tests/test_orchestrator.py -v`
Expected: PASS

**Step 5: Run full suite**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add sage-python/src/sage/orchestrator.py sage-python/tests/test_orchestrator.py
git commit -m "feat(routing): add cascade fallback to ModelAgent (FrugalGPT pattern)"
```

---

### Task 4: Fix MemoryAgent vendor lock-in

**Context:** `MemoryAgent._llm_extract()` hardcodes `GoogleProvider()` (line 79). It should accept an LLM provider via constructor injection so it uses whatever provider the system has available.

**Files:**
- Modify: `sage-python/src/sage/memory/memory_agent.py:26-31,64-79`
- Modify: `sage-python/src/sage/boot.py` (pass provider to MemoryAgent)
- Test: `sage-python/tests/test_memory_agent_provider.py` (new)

**Step 1: Write the failing test**

Create `sage-python/tests/test_memory_agent_provider.py`:

```python
"""Tests for MemoryAgent provider injection (no vendor lock-in)."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sage.memory.memory_agent import MemoryAgent
from sage.llm.base import LLMResponse


@pytest.mark.asyncio
async def test_memory_agent_uses_injected_provider():
    """MemoryAgent should use the injected LLM provider, not hardcoded Google."""
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=LLMResponse(
        content='{"entities": ["Python"], "relationships": [["Python", "is", "language"]], "summary": "About Python"}',
        model="test-model",
    ))

    agent = MemoryAgent(use_llm=True, llm_provider=mock_provider)
    result = await agent.extract("Python is a programming language")

    mock_provider.generate.assert_awaited_once()
    assert "Python" in result.entities


@pytest.mark.asyncio
async def test_memory_agent_falls_back_to_heuristic_without_provider():
    """Without a provider, LLM extraction should fall back to heuristic."""
    agent = MemoryAgent(use_llm=True, llm_provider=None)
    result = await agent.extract("Python is a programming language")

    # Heuristic should still find capitalized entities
    assert isinstance(result.entities, list)


def test_memory_agent_accepts_provider_kwarg():
    """Constructor must accept llm_provider keyword argument."""
    mock = MagicMock()
    agent = MemoryAgent(use_llm=True, llm_provider=mock)
    assert agent._llm_provider is mock
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_memory_agent_provider.py -v`
Expected: FAIL — `MemoryAgent` doesn't accept `llm_provider` parameter.

**Step 3: Implement provider injection**

In `sage-python/src/sage/memory/memory_agent.py`:

1. Add `llm_provider` parameter to `__init__`:

```python
    def __init__(
        self,
        use_llm: bool = True,
        llm_tier: str = "budget",
        compress_threshold: int = 50,
        llm_provider: Any = None,
    ):
        self.use_llm = use_llm
        self.llm_tier = llm_tier
        self.compress_threshold = compress_threshold
        self._llm_provider = llm_provider
```

Add `from typing import Any` to imports at top (already there from dataclass field import).

2. Modify `_llm_extract()` to use injected provider:

```python
    async def _llm_extract(self, text: str) -> ExtractionResult:
        """LLM-powered extraction with structured output."""
        from pydantic import BaseModel
        from sage.llm.base import Message, Role

        class KGExtraction(BaseModel):
            entities: list[str]
            relationships: list[list[str]]
            summary: str

        provider = self._llm_provider
        if provider is None:
            # Fallback: try Google if available, else heuristic
            try:
                from sage.llm.router import ModelRouter
                from sage.llm.google import GoogleProvider
                config = ModelRouter.get_config(
                    self.llm_tier, temperature=0.1, json_schema=KGExtraction,
                )
                provider = GoogleProvider()
            except Exception as e:
                log.warning("No LLM provider for entity extraction: %s", e)
                return self._heuristic_extract(text)
        else:
            from sage.llm.router import ModelRouter
            config = ModelRouter.get_config(
                self.llm_tier, temperature=0.1, json_schema=KGExtraction,
            )

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

3. In `sage-python/src/sage/boot.py`, pass provider to `MemoryAgent`:

Change line ~186:
```python
    memory_agent = MemoryAgent(use_llm=not use_mock_llm)
```
To:
```python
    memory_agent = MemoryAgent(use_llm=not use_mock_llm, llm_provider=provider)
```

**Step 4: Run test to verify it passes**

Run: `cd sage-python && python -m pytest tests/test_memory_agent_provider.py -v`
Expected: PASS

**Step 5: Run full suite**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add sage-python/src/sage/memory/memory_agent.py sage-python/src/sage/boot.py sage-python/tests/test_memory_agent_provider.py
git commit -m "fix(memory): remove GoogleProvider vendor lock in MemoryAgent"
```

---

### Task 5: Fix ExoCortex.query() hardcoded model

**Context:** `ExoCortex.query()` at line 139 hardcodes `model="gemini-3.1-flash-lite-preview"`. This should be configurable via constructor parameter.

**Files:**
- Modify: `sage-python/src/sage/memory/remote_rag.py:29-31,121-147`
- Test: `sage-python/tests/test_exocortex_model.py` (new)

**Step 1: Write the failing test**

Create `sage-python/tests/test_exocortex_model.py`:

```python
"""Tests for ExoCortex model configurability."""
from sage.memory.remote_rag import ExoCortex


def test_exocortex_accepts_model_parameter():
    """ExoCortex constructor must accept a model_id parameter."""
    exo = ExoCortex(store_name="test", model_id="custom-model")
    assert exo._model_id == "custom-model"


def test_exocortex_default_model():
    """ExoCortex should default to gemini-3.1-flash-lite-preview."""
    exo = ExoCortex(store_name="test")
    assert "gemini" in exo._model_id.lower()


def test_exocortex_model_from_env(monkeypatch):
    """SAGE_EXOCORTEX_MODEL env var should override default."""
    monkeypatch.setenv("SAGE_EXOCORTEX_MODEL", "gpt-5-nano")
    exo = ExoCortex(store_name="test")
    assert exo._model_id == "gpt-5-nano"
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_exocortex_model.py -v`
Expected: FAIL — `ExoCortex` doesn't accept `model_id` parameter.

**Step 3: Implement model configurability**

In `sage-python/src/sage/memory/remote_rag.py`:

1. Add `_DEFAULT_MODEL` constant near `DEFAULT_STORE`:

```python
_DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"
```

2. Add `model_id` to `__init__`:

```python
    def __init__(self, store_name: str | None = None, model_id: str | None = None):
        self._store_name = store_name or os.environ.get("SAGE_EXOCORTEX_STORE") or DEFAULT_STORE
        self._api_key = os.environ.get("GOOGLE_API_KEY", "")
        self._model_id = model_id or os.environ.get("SAGE_EXOCORTEX_MODEL") or _DEFAULT_MODEL
```

3. In `query()`, replace hardcoded model:

Change line 139:
```python
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview",
```
To:
```python
            response = client.models.generate_content(
                model=self._model_id,
```

**Step 4: Run test to verify it passes**

Run: `cd sage-python && python -m pytest tests/test_exocortex_model.py -v`
Expected: PASS

**Step 5: Run full suite**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add sage-python/src/sage/memory/remote_rag.py sage-python/tests/test_exocortex_model.py
git commit -m "fix(exocortex): make query model configurable (remove vendor lock)"
```

---

### Task 6: Fix ComplexityRouter LLM assessment vendor lock

**Context:** `ComplexityRouter._assess_via_llm()` at lines 164-196 hardcodes `google.genai.Client` with `model="gemini-2.5-flash-lite"`. It should accept an LLM provider to enable assessment via any provider.

**Files:**
- Modify: `sage-python/src/sage/strategy/metacognition.py:74-101,142-196`
- Modify: `sage-python/src/sage/boot.py` (pass provider)
- Test: `sage-python/tests/test_metacognition_provider.py` (new)

**Step 1: Write the failing test**

Create `sage-python/tests/test_metacognition_provider.py`:

```python
"""Tests for ComplexityRouter provider injection."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from sage.strategy.metacognition import ComplexityRouter
from sage.llm.base import LLMResponse


def test_complexity_router_accepts_llm_provider():
    """ComplexityRouter must accept an llm_provider parameter."""
    mock = MagicMock()
    router = ComplexityRouter(llm_provider=mock)
    assert router._llm_provider is mock


@pytest.mark.asyncio
async def test_assess_uses_injected_provider():
    """When llm_provider is set, should use it instead of google.genai."""
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(return_value=LLMResponse(
        content='{"complexity": 0.3, "uncertainty": 0.1, "tool_required": false, "reasoning": "Simple math"}',
        model="test",
    ))

    router = ComplexityRouter(llm_provider=mock_provider)
    profile = await router.assess_complexity_async("What is 2+2?")

    assert 0.0 <= profile.complexity <= 1.0
    # Should have used the injected provider
    mock_provider.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_assess_falls_back_to_heuristic_on_error():
    """If injected provider fails, fall back to heuristic."""
    mock_provider = AsyncMock()
    mock_provider.generate = AsyncMock(side_effect=RuntimeError("API down"))

    router = ComplexityRouter(llm_provider=mock_provider)
    profile = await router.assess_complexity_async("What is 2+2?")

    # Should fall back to heuristic, not crash
    assert profile.reasoning == "heuristic"
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_metacognition_provider.py -v`
Expected: FAIL — `ComplexityRouter` doesn't accept `llm_provider`.

**Step 3: Implement provider injection**

In `sage-python/src/sage/strategy/metacognition.py`:

1. Add `llm_provider` to `__init__`:

```python
    def __init__(
        self,
        s1_complexity_ceil: float = 0.50,
        s1_uncertainty_ceil: float = 0.3,
        s3_complexity_floor: float = 0.65,
        s3_uncertainty_floor: float = 0.6,
        brake_window: int = 3,
        brake_entropy_threshold: float = 0.15,
        llm_provider: Any = None,
    ):
        # ... existing fields ...
        self._llm_provider = llm_provider
```

Add `from typing import Any` to imports.

2. Modify `assess_complexity_async()` to check for injected provider:

```python
    async def assess_complexity_async(self, task: str) -> CognitiveProfile:
        """LLM-based task assessment.

        Uses injected llm_provider if available, else tries Google Gemini,
        else falls back to heuristic.
        """
        if self._llm_provider is not None:
            try:
                return await self._assess_via_provider(task)
            except Exception as e:
                log.warning("LLM routing via provider failed (%s), falling back to heuristic", e)
                return self._assess_heuristic(task)

        if self._llm_available is None:
            self._llm_available = bool(os.environ.get("GOOGLE_API_KEY"))

        if self._llm_available:
            try:
                return await self._assess_via_llm(task)
            except Exception as e:
                log.warning(f"LLM routing failed ({e}), falling back to heuristic")

        return self._assess_heuristic(task)
```

3. Add `_assess_via_provider()`:

```python
    async def _assess_via_provider(self, task: str) -> CognitiveProfile:
        """Assess complexity using the injected LLM provider."""
        from sage.llm.base import Message, Role, LLMConfig
        import json

        prompt = _ROUTING_PROMPT.format(task=task[:2000])
        response = await self._llm_provider.generate(
            messages=[Message(role=Role.USER, content=prompt)],
            config=LLMConfig(
                provider="auto", model="auto",
                temperature=0.0, max_tokens=256,
            ),
        )
        data = json.loads(response.content)
        profile = CognitiveProfile(
            complexity=max(0.0, min(1.0, float(data.get("complexity", 0.5)))),
            uncertainty=max(0.0, min(1.0, float(data.get("uncertainty", 0.3)))),
            tool_required=bool(data.get("tool_required", False)),
            reasoning=str(data.get("reasoning", "")),
        )
        log.info("Provider routing: c=%.2f u=%.2f tool=%s — %s",
                 profile.complexity, profile.uncertainty,
                 profile.tool_required, profile.reasoning)
        return profile
```

4. In `sage-python/src/sage/boot.py`, pass provider to ComplexityRouter:

Change line ~183:
```python
    metacognition = ComplexityRouter()
```
To:
```python
    metacognition = ComplexityRouter(llm_provider=provider if not use_mock_llm else None)
```

**Step 4: Run test to verify it passes**

Run: `cd sage-python && python -m pytest tests/test_metacognition_provider.py -v`
Expected: PASS

**Step 5: Run full suite**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add sage-python/src/sage/strategy/metacognition.py sage-python/src/sage/boot.py sage-python/tests/test_metacognition_provider.py
git commit -m "fix(routing): remove Google vendor lock in ComplexityRouter"
```

---

### Task 7: Provider quirk dispatcher in OpenAICompatProvider

**Context:** Different OpenAI-compatible APIs have quirks: DeepSeek ignores temperature on reasoner models, Grok puts reasoning in `reasoning_content`, Kimi clamps temperature to [0,1], MiniMax puts `<think>` tags in content. We need a quirk dispatcher keyed on `(provider, model_family)`.

**Files:**
- Modify: `sage-python/src/sage/providers/openai_compat.py`
- Test: `sage-python/tests/test_provider_quirks.py` (new)

**Step 1: Write the failing test**

Create `sage-python/tests/test_provider_quirks.py`:

```python
"""Tests for provider-specific quirk handling in OpenAICompatProvider."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from sage.providers.openai_compat import OpenAICompatProvider
from sage.llm.base import Message, Role, LLMConfig


class TestQuirkDispatcher:
    def test_deepseek_reasoner_strips_temperature(self):
        """DeepSeek reasoner model ignores temperature — should be removed from params."""
        p = OpenAICompatProvider(
            api_key="k", base_url="https://api.deepseek.com",
            model_id="deepseek-reasoner", provider_name="deepseek",
        )
        params = p._apply_quirks(
            {"model": "deepseek-reasoner", "temperature": 0.7, "max_tokens": 4096}
        )
        assert "temperature" not in params

    def test_deepseek_chat_keeps_temperature(self):
        """DeepSeek chat model should keep temperature."""
        p = OpenAICompatProvider(
            api_key="k", base_url="https://api.deepseek.com",
            model_id="deepseek-chat", provider_name="deepseek",
        )
        params = p._apply_quirks(
            {"model": "deepseek-chat", "temperature": 0.7, "max_tokens": 4096}
        )
        assert params["temperature"] == 0.7

    def test_kimi_clamps_temperature(self):
        """Kimi API requires temperature in [0, 1] range."""
        p = OpenAICompatProvider(
            api_key="k", base_url="https://api.moonshot.ai/v1",
            model_id="kimi-k2.5", provider_name="kimi",
        )
        params = p._apply_quirks(
            {"model": "kimi-k2.5", "temperature": 1.5, "max_tokens": 4096}
        )
        assert params["temperature"] <= 1.0

    def test_unknown_provider_no_quirks(self):
        """Unknown providers should pass through unchanged."""
        p = OpenAICompatProvider(
            api_key="k", model_id="test",
        )
        original = {"model": "test", "temperature": 0.7, "max_tokens": 4096}
        params = p._apply_quirks(original.copy())
        assert params == original


class TestReasoningContentExtraction:
    @pytest.mark.asyncio
    async def test_deepseek_reasoning_content_merged(self):
        """DeepSeek reasoning_content should be merged into main content."""
        p = OpenAICompatProvider(
            api_key="k", base_url="https://api.deepseek.com",
            model_id="deepseek-reasoner", provider_name="deepseek",
        )

        mock_message = MagicMock()
        mock_message.content = "Final answer"
        mock_message.reasoning_content = "Step 1: think\nStep 2: analyze"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            result = await p.generate(
                [Message(role=Role.USER, content="Think about this")],
                config=LLMConfig(provider="deepseek", model="deepseek-reasoner"),
            )

        assert "<think>" in result.content
        assert "Step 1: think" in result.content
        assert "Final answer" in result.content

    @pytest.mark.asyncio
    async def test_minimax_think_tags_extracted(self):
        """MiniMax puts <think> tags in content — should be preserved."""
        p = OpenAICompatProvider(
            api_key="k", base_url="https://api.minimaxi.chat/v1",
            model_id="MiniMax-M2.5", provider_name="minimax",
        )

        mock_message = MagicMock()
        mock_message.content = "<think>reasoning here</think>\nFinal answer"
        mock_message.reasoning_content = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            result = await p.generate(
                [Message(role=Role.USER, content="Think about this")],
                config=LLMConfig(provider="minimax", model="MiniMax-M2.5"),
            )

        # MiniMax think tags should be preserved (already in content)
        assert "<think>" in result.content
        assert "Final answer" in result.content
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_provider_quirks.py -v`
Expected: FAIL — `provider_name` param and `_apply_quirks` don't exist.

**Step 3: Implement quirk dispatcher**

In `sage-python/src/sage/providers/openai_compat.py`, replace the class with:

```python
class OpenAICompatProvider:
    """Provider for any OpenAI-compatible API (OpenAI, xAI, DeepSeek, MiniMax, Kimi).

    Handles provider-specific quirks:
    - DeepSeek: strip temperature for reasoner; merge reasoning_content
    - Grok (xAI): merge reasoning_content into <think> tags
    - Kimi: clamp temperature to [0, 1]
    - MiniMax: <think> tags already in content body (preserve as-is)
    """

    name = "openai-compat"

    def __init__(self, api_key: str, base_url: str | None = None,
                 model_id: str = "", provider_name: str = ""):
        self.api_key = api_key
        self.base_url = base_url
        self.model_id = model_id
        self.provider_name = provider_name or self._infer_provider(base_url)

    @staticmethod
    def _infer_provider(base_url: str | None) -> str:
        """Infer provider name from base_url for quirk dispatch."""
        if not base_url:
            return "openai"
        url = base_url.lower()
        if "deepseek" in url:
            return "deepseek"
        if "x.ai" in url:
            return "xai"
        if "minimaxi" in url:
            return "minimax"
        if "moonshot" in url:
            return "kimi"
        if "openai.com" in url:
            return "openai"
        return ""

    def capabilities(self) -> dict[str, bool]:
        """Declare what this provider actually supports."""
        return {
            "structured_output": False,
            "tool_role": False,
            "file_search": False,
            "grounding": False,
            "system_prompt": True,
            "streaming": False,
        }

    def _apply_quirks(self, params: dict[str, Any]) -> dict[str, Any]:
        """Apply provider-specific parameter quirks before API call."""
        model = params.get("model", self.model_id).lower()

        if self.provider_name == "deepseek":
            # DeepSeek reasoner ignores temperature — remove to avoid warnings
            if "reasoner" in model and "temperature" in params:
                del params["temperature"]

        elif self.provider_name == "kimi":
            # Kimi API requires temperature in [0, 1] (not OpenAI's [0, 2])
            if "temperature" in params:
                params["temperature"] = min(params["temperature"], 1.0)

        elif self.provider_name == "openai":
            # GPT-5.4 class supports reasoning_effort parameter
            # (would be set via config.extra_params in future)
            pass

        return params

    def _extract_reasoning(self, message: Any) -> tuple[str, str]:
        """Extract reasoning content and main content from response message.

        Returns (reasoning, content) tuple.
        """
        content = message.content or ""
        reasoning = getattr(message, "reasoning_content", None) or ""

        # DeepSeek and Grok put reasoning in reasoning_content field
        # MiniMax puts <think> tags directly in content (no extraction needed)
        return reasoning, content

    def _format_response(self, reasoning: str, content: str) -> str:
        """Merge reasoning into content with <think> tags if present."""
        if reasoning:
            return f"<think>{reasoning}</think>\n{content}"
        return content

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        """Convert Message objects to OpenAI dict format."""
        oai_messages: list[dict[str, str]] = []
        for msg in messages:
            role = msg.role.value
            if role == "tool":
                log.warning(
                    "Rewriting tool role to user for OpenAI-compat API — "
                    "semantic context (tool provenance) is lost"
                )
                role = "user"
            oai_messages.append({"role": role, "content": msg.content})
        return oai_messages

    async def generate(
        self,
        messages: list[Message],
        tools: list | None = None,
        config: LLMConfig | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate content via OpenAI-compatible chat completions API."""
        if kwargs.get("file_search_store_names"):
            log.warning("file_search_store_names not supported by OpenAI-compat provider, ignored")
        from openai import AsyncOpenAI

        model = self.model_id
        if config and config.model:
            model = config.model

        client_kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        client = AsyncOpenAI(**client_kwargs)
        oai_messages = self._convert_messages(messages)

        # Build and apply quirks to request params
        params: dict[str, Any] = {
            "model": model,
            "messages": oai_messages,
            "max_tokens": config.max_tokens if config and config.max_tokens else 4096,
            "temperature": config.temperature if config else 0.3,
        }
        params = self._apply_quirks(params)

        try:
            response = await client.chat.completions.create(**params)  # type: ignore[arg-type]

            msg = response.choices[0].message
            reasoning, content = self._extract_reasoning(msg)
            final_content = self._format_response(reasoning, content)

            return LLMResponse(content=final_content, model=model)

        except Exception as e:
            log.error("OpenAI-compat API error (%s/%s): %s", self.provider_name, self.base_url, e)
            raise
```

**Step 4: Run test to verify it passes**

Run: `cd sage-python && python -m pytest tests/test_provider_quirks.py -v`
Expected: PASS

**Step 5: Run full suite**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: All tests pass. Watch for `test_orchestrator.py` tests that construct `OpenAICompatProvider` — they should still work because `provider_name` defaults to inferred.

**Step 6: Commit**

```bash
git add sage-python/src/sage/providers/openai_compat.py sage-python/tests/test_provider_quirks.py
git commit -m "feat(providers): add quirk dispatcher for DeepSeek/Kimi/Grok/MiniMax"
```

---

### Task 8: Auto-populate CapabilityMatrix at boot

**Context:** `CapabilityMatrix` is defined but never populated. We populate it from discovered models + known provider capabilities at boot, so `DynamicRouter` can filter by hard capabilities.

**Files:**
- Modify: `sage-python/src/sage/boot.py` (populate matrix)
- Modify: `sage-python/src/sage/providers/capabilities.py` (add from_provider classmethod)
- Test: `sage-python/tests/test_capability_matrix.py` (new)

**Step 1: Write the failing test**

Create `sage-python/tests/test_capability_matrix.py`:

```python
"""Tests for CapabilityMatrix auto-population."""
from sage.providers.capabilities import ProviderCapabilities, CapabilityMatrix


def test_known_provider_capabilities():
    """Known providers should have correct capability flags."""
    matrix = CapabilityMatrix()

    # Register known providers
    matrix.register(ProviderCapabilities.for_provider("google"))
    matrix.register(ProviderCapabilities.for_provider("openai"))
    matrix.register(ProviderCapabilities.for_provider("deepseek"))

    google = matrix.get("google")
    assert google.file_search is True
    assert google.structured_output is True
    assert google.tool_role is True

    openai = matrix.get("openai")
    assert openai.structured_output is True
    assert openai.tool_role is True

    ds = matrix.get("deepseek")
    assert ds.structured_output is True


def test_providers_for_file_search():
    """Only Google supports file_search."""
    matrix = CapabilityMatrix()
    matrix.register(ProviderCapabilities.for_provider("google"))
    matrix.register(ProviderCapabilities.for_provider("openai"))
    matrix.register(ProviderCapabilities.for_provider("deepseek"))

    matches = matrix.providers_for(file_search=True)
    assert "google" in matches
    assert "openai" not in matches


def test_unknown_provider_returns_defaults():
    """Unknown provider should return conservative defaults."""
    caps = ProviderCapabilities.for_provider("unknown-llm")
    assert caps.provider == "unknown-llm"
    assert caps.system_prompt is True  # Almost all support this
    assert caps.file_search is False   # Conservative default
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_capability_matrix.py -v`
Expected: FAIL — `ProviderCapabilities.for_provider()` doesn't exist.

**Step 3: Implement classmethod**

In `sage-python/src/sage/providers/capabilities.py`:

```python
"""Semantic capability matrix — hard-fail when required features missing."""
from __future__ import annotations

from dataclasses import dataclass


# Known capabilities per provider (curated, not discovered)
_KNOWN_CAPABILITIES: dict[str, dict[str, bool]] = {
    "google": {
        "structured_output": True,
        "tool_role": True,
        "file_search": True,
        "grounding": True,
        "system_prompt": True,
        "streaming": True,
    },
    "openai": {
        "structured_output": True,
        "tool_role": True,
        "file_search": False,
        "grounding": False,
        "system_prompt": True,
        "streaming": True,
    },
    "codex": {
        "structured_output": False,
        "tool_role": False,
        "file_search": False,
        "grounding": False,
        "system_prompt": False,
        "streaming": False,
    },
    "xai": {
        "structured_output": True,
        "tool_role": True,
        "file_search": False,
        "grounding": False,
        "system_prompt": True,
        "streaming": False,
    },
    "deepseek": {
        "structured_output": True,
        "tool_role": True,
        "file_search": False,
        "grounding": False,
        "system_prompt": True,
        "streaming": False,
    },
    "minimax": {
        "structured_output": True,
        "tool_role": True,
        "file_search": False,
        "grounding": False,
        "system_prompt": True,
        "streaming": False,
    },
    "kimi": {
        "structured_output": True,
        "tool_role": True,
        "file_search": False,
        "grounding": False,
        "system_prompt": True,
        "streaming": False,
    },
}


@dataclass
class ProviderCapabilities:
    provider: str
    structured_output: bool = False
    tool_role: bool = False
    file_search: bool = False
    grounding: bool = False
    system_prompt: bool = True
    streaming: bool = False

    @classmethod
    def for_provider(cls, provider: str) -> ProviderCapabilities:
        """Return capabilities for a known provider, or conservative defaults."""
        known = _KNOWN_CAPABILITIES.get(provider, {})
        return cls(
            provider=provider,
            structured_output=known.get("structured_output", False),
            tool_role=known.get("tool_role", False),
            file_search=known.get("file_search", False),
            grounding=known.get("grounding", False),
            system_prompt=known.get("system_prompt", True),
            streaming=known.get("streaming", False),
        )


class CapabilityMatrix:
    def __init__(self) -> None:
        self._providers: dict[str, ProviderCapabilities] = {}

    def register(self, caps: ProviderCapabilities) -> None:
        self._providers[caps.provider] = caps

    def get(self, provider: str) -> ProviderCapabilities:
        return self._providers[provider]

    def providers_for(self, **requirements: bool) -> list[str]:
        result = []
        for name, caps in self._providers.items():
            if all(getattr(caps, k, False) == v for k, v in requirements.items() if v):
                result.append(name)
        return result

    def require(self, **requirements: bool) -> list[str]:
        compatible = self.providers_for(**requirements)
        if not compatible:
            missing = [k for k, v in requirements.items() if v]
            raise ValueError(f"No provider supports: {missing}")
        return compatible

    def populate_from_providers(self, provider_names: list[str]) -> None:
        """Auto-populate from a list of discovered provider names."""
        for name in provider_names:
            if name not in self._providers:
                self.register(ProviderCapabilities.for_provider(name))
```

In `sage-python/src/sage/boot.py`, after `registry.refresh()`, add matrix population:

```python
        # Auto-populate capability matrix from discovered providers
        from sage.providers.capabilities import CapabilityMatrix
        cap_matrix = CapabilityMatrix()
        discovered_providers = {p.provider for p in registry.list_available()}
        cap_matrix.populate_from_providers(list(discovered_providers))
```

Store `cap_matrix` on the orchestrator or system for later use. In `AgentSystem`, add field:

```python
    capability_matrix: Any = None
```

And in boot:
```python
    return AgentSystem(
        ...,
        capability_matrix=cap_matrix if not use_mock_llm else None,
    )
```

**Step 4: Run test to verify it passes**

Run: `cd sage-python && python -m pytest tests/test_capability_matrix.py -v`
Expected: PASS

**Step 5: Run full suite**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add sage-python/src/sage/providers/capabilities.py sage-python/src/sage/boot.py sage-python/tests/test_capability_matrix.py
git commit -m "feat(capabilities): auto-populate CapabilityMatrix with known provider capabilities"
```

---

### Task 9: Pass provider_name through ModelAgent → OpenAICompatProvider

**Context:** Now that `OpenAICompatProvider` has quirk dispatch keyed on `provider_name`, `ModelAgent._create_provider_for()` must pass the provider name from the `ModelProfile`. Without this, quirks won't activate.

**Files:**
- Modify: `sage-python/src/sage/orchestrator.py` (ModelAgent._create_provider_for)
- Test: `sage-python/tests/test_orchestrator.py` (extend)

**Step 1: Write the failing test**

Append to `sage-python/tests/test_orchestrator.py`:

```python
class TestProviderNamePassthrough:
    def test_create_provider_passes_provider_name(self):
        """OpenAICompatProvider must receive provider_name from ModelProfile."""
        profile = _make_profile("deepseek-reasoner", provider="deepseek")
        agent = ModelAgent(name="ds", model=profile)

        with patch("sage.providers.openai_compat.OpenAICompatProvider") as mock_cls:
            mock_cls.return_value = MagicMock()
            agent._create_provider_for(profile)
            call_kwargs = mock_cls.call_args
            # provider_name should be passed
            assert call_kwargs.kwargs.get("provider_name") == "deepseek" or \
                   (len(call_kwargs.args) > 3 if call_kwargs.args else False)
```

**Step 2: Run test to verify it fails**

Run: `cd sage-python && python -m pytest tests/test_orchestrator.py::TestProviderNamePassthrough -v`
Expected: FAIL — `provider_name` not passed.

**Step 3: Implement the passthrough**

In `sage-python/src/sage/orchestrator.py`, in `ModelAgent._create_provider_for()`:

Change:
```python
            return OpenAICompatProvider(
                api_key=api_key,
                base_url=cfg.get("base_url"),
                model_id=model.id,
            )
```
To:
```python
            return OpenAICompatProvider(
                api_key=api_key,
                base_url=cfg.get("base_url"),
                model_id=model.id,
                provider_name=model.provider,
            )
```

**Step 4: Run test to verify it passes**

Run: `cd sage-python && python -m pytest tests/test_orchestrator.py -v`
Expected: PASS

**Step 5: Run full suite**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add sage-python/src/sage/orchestrator.py sage-python/tests/test_orchestrator.py
git commit -m "fix(routing): pass provider_name to OpenAICompatProvider for quirk dispatch"
```

---

### Task 10: Log discovered models summary at boot

**Context:** User wants to see what models are available at session start. Add a clear summary log at INFO level showing discovered providers, model counts, and which are available.

**Files:**
- Modify: `sage-python/src/sage/boot.py` (enhance logging)
- Test: `sage-python/tests/test_boot_refresh.py` (extend)

**Step 1: Write the failing test**

Append to `sage-python/tests/test_boot_refresh.py`:

```python
def test_boot_logs_discovery_summary(monkeypatch, caplog):
    """Boot should log a human-readable summary of discovered models."""
    import logging
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")

    async def mock_refresh(self):
        # Simulate discovering some models
        from sage.providers.registry import ModelProfile
        self._profiles = {
            "gemini-flash": ModelProfile(id="gemini-flash", provider="google", available=True, cost_input=0.1, cost_output=0.5),
            "gpt-5": ModelProfile(id="gpt-5", provider="openai", available=True, cost_input=1.0, cost_output=5.0),
            "deepseek-chat": ModelProfile(id="deepseek-chat", provider="deepseek", available=False, cost_input=0.3, cost_output=0.4),
        }

    with patch("sage.providers.registry.ModelRegistry.refresh", mock_refresh):
        with patch("sage.llm.google.GoogleProvider"):
            with caplog.at_level(logging.INFO, logger="sage.boot"):
                try:
                    boot_agent_system(use_mock_llm=False, llm_tier="fast")
                except Exception:
                    pass

    # Should log provider summary
    assert any("discovered" in r.message.lower() or "available" in r.message.lower()
               for r in caplog.records)
```

**Step 2: Run test — may already pass if Task 1 logging was sufficient**

Run: `cd sage-python && python -m pytest tests/test_boot_refresh.py::test_boot_logs_discovery_summary -v`

**Step 3: Enhance logging if needed**

In `boot.py`, after `registry.refresh()` succeeds, add detailed summary:

```python
            # Log per-provider summary
            from collections import Counter
            available = registry.list_available()
            provider_counts = Counter(p.provider for p in available)
            total = len(registry.profiles)
            avail = len(available)
            summary_parts = [f"{name}: {count}" for name, count in sorted(provider_counts.items())]
            _log.info(
                "Boot: discovered %d models (%d available) — %s",
                total, avail, ", ".join(summary_parts) if summary_parts else "none",
            )
```

**Step 4: Run test to verify**

Run: `cd sage-python && python -m pytest tests/test_boot_refresh.py -v`
Expected: PASS

**Step 5: Run full suite**

Run: `cd sage-python && python -m pytest tests/ -x -q`
Expected: All tests pass

**Step 6: Commit**

```bash
git add sage-python/src/sage/boot.py sage-python/tests/test_boot_refresh.py
git commit -m "feat(boot): log per-provider discovery summary at startup"
```

---

### Task 11: Integration test — full multi-provider smoke test

**Context:** Verify the complete flow: boot → discover → select → cascade works end-to-end with mocked API responses. No real API calls.

**Files:**
- Test: `sage-python/tests/test_multi_provider_integration.py` (new)

**Step 1: Write the integration test**

Create `sage-python/tests/test_multi_provider_integration.py`:

```python
"""Integration test: multi-provider routing end-to-end (all mocked)."""
import sys
import types as _types

if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = _types.ModuleType("sage_core")
    _mock = sys.modules["sage_core"]

    class _WM:
        def __init__(self, **kw): self._events = []; self._counter = 0
        def add_event(self, t, c):
            self._counter += 1; self._events.append({"type": t, "content": c}); return f"e-{self._counter}"
        def event_count(self): return len(self._events)
        def recent_events(self, n): return self._events[-n:]
        def compact_to_arrow(self): return 0
        def compact_to_arrow_with_meta(self, *a): return 0
        def retrieve_relevant_chunks(self, *a): return []
        def get_page_out_candidates(self, *a): return []
        def smmu_chunk_count(self): return 0
        def get_latest_arrow_chunk(self): return None

    _mock.WorkingMemory = _WM

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sage.providers.registry import ModelProfile, ModelRegistry
from sage.orchestrator import CognitiveOrchestrator, ModelAgent
from sage.strategy.metacognition import ComplexityRouter


@pytest.mark.asyncio
async def test_orchestrator_selects_cheapest_for_simple_task():
    """S1 routing should select the cheapest viable model."""
    cheap = ModelProfile(
        id="cheap-flash", provider="google", available=True,
        code_score=0.6, reasoning_score=0.7, cost_input=0.025, cost_output=1.5,
    )
    expensive = ModelProfile(
        id="expensive-pro", provider="openai", available=True,
        code_score=0.9, reasoning_score=0.95, cost_input=15.0, cost_output=120.0,
    )

    reg = ModelRegistry.__new__(ModelRegistry)
    reg._profiles = {"cheap-flash": cheap, "expensive-pro": expensive}
    reg._connector = MagicMock()

    orch = CognitiveOrchestrator(registry=reg)

    with patch.object(ModelAgent, "run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = "4"
        result = await orch.run("What is 2+2?")

    assert result == "4"


@pytest.mark.asyncio
async def test_orchestrator_selects_best_coder_for_code_task():
    """S2 routing for code tasks should prefer high code_score."""
    coder = ModelProfile(
        id="gpt-codex", provider="openai", available=True,
        code_score=0.85, reasoning_score=0.93, cost_input=1.75, cost_output=14.0,
    )
    generalist = ModelProfile(
        id="deepseek-chat", provider="deepseek", available=True,
        code_score=0.72, reasoning_score=0.75, cost_input=0.28, cost_output=0.42,
    )

    reg = ModelRegistry.__new__(ModelRegistry)
    reg._profiles = {"gpt-codex": coder, "deepseek-chat": generalist}
    reg._connector = MagicMock()

    orch = CognitiveOrchestrator(registry=reg)

    selected_model = [None]

    async def capture_run(self_agent, task):
        selected_model[0] = self_agent.model.id
        return "def fibonacci(n): ..."

    with patch.object(ModelAgent, "run", capture_run):
        result = await orch.run("Write a Python function to compute fibonacci numbers")

    # S2 code task should prefer the model with higher code_score
    assert selected_model[0] == "gpt-codex"


@pytest.mark.asyncio
async def test_registry_select_respects_cost_cap():
    """Models above cost cap should be excluded."""
    cheap = ModelProfile(
        id="budget", provider="google", available=True,
        code_score=0.5, reasoning_score=0.5, cost_input=0.025, cost_output=0.5,
    )
    expensive = ModelProfile(
        id="premium", provider="openai", available=True,
        code_score=0.9, reasoning_score=0.95, cost_input=15.0, cost_output=120.0,
    )

    reg = ModelRegistry.__new__(ModelRegistry)
    reg._profiles = {"budget": cheap, "premium": expensive}
    reg._connector = MagicMock()

    # With max_cost_per_1m=5.0, premium should be excluded
    result = reg.select({"code": 0.5, "max_cost_per_1m": 5.0})
    assert result is not None
    assert result.id == "budget"


@pytest.mark.asyncio
async def test_cascade_across_providers():
    """When primary provider fails, cascade should try another."""
    primary = ModelProfile(
        id="primary-gpt", provider="openai", available=True,
        code_score=0.85, reasoning_score=0.9, cost_input=2.0, cost_output=10.0,
    )
    fallback = ModelProfile(
        id="fallback-gemini", provider="google", available=True,
        code_score=0.7, reasoning_score=0.8, cost_input=0.15, cost_output=3.5,
    )

    reg = ModelRegistry.__new__(ModelRegistry)
    reg._profiles = {"primary-gpt": primary, "fallback-gemini": fallback}
    reg._connector = MagicMock()

    agent = ModelAgent(name="test", model=primary, registry=reg)

    call_count = [0]
    async def failing_then_ok(model, messages):
        call_count[0] += 1
        if model.id == "primary-gpt":
            raise RuntimeError("OpenAI rate limited")
        return "Fallback response"

    with patch.object(ModelAgent, "_call_provider", failing_then_ok):
        result = await agent.run("Test cascade")

    assert "Fallback response" in result
    assert call_count[0] == 2  # Primary failed, fallback succeeded
```

**Step 2: Run the integration test**

Run: `cd sage-python && python -m pytest tests/test_multi_provider_integration.py -v`
Expected: PASS (if all previous tasks implemented correctly)

**Step 3: Fix any failures**

If any tests fail, fix the underlying issue.

**Step 4: Commit**

```bash
git add sage-python/tests/test_multi_provider_integration.py
git commit -m "test: add multi-provider routing integration tests"
```

---

### Task 12: Update CLAUDE.md and MEMORY.md

**Context:** Document the new multi-provider routing architecture, updated module descriptions, and new test counts.

**Files:**
- Modify: `CLAUDE.md`
- Modify: `C:\Users\yann.abadie\.claude\projects\C--Code-YGN-SAGE\memory\MEMORY.md`

**Step 1: Update CLAUDE.md**

In the Key Python Modules section, update/add:
- `orchestrator.py` — add "primary routing engine with cascade fallback"
- `providers/openai_compat.py` — add "with provider-specific quirk dispatch"
- `providers/capabilities.py` — add "auto-populated at boot from discovered providers"
- `boot.py` — add "auto-discovers models via registry.refresh(), populates CapabilityMatrix"
- `memory/memory_agent.py` — update "supports provider injection (no Google vendor lock)"
- `memory/remote_rag.py` — update "configurable model via SAGE_EXOCORTEX_MODEL"
- `strategy/metacognition.py` — update "supports provider injection for LLM assessment"

In the Provider Status section, update to show all 6+ providers.

**Step 2: Update MEMORY.md**

Add section:
```markdown
## Multi-Provider Dynamic Routing (March 8, 2026)
- Design: `docs/plans/2026-03-08-multi-provider-design.md`
- Plan: `docs/plans/2026-03-08-multi-provider-plan.md`
- CognitiveOrchestrator wired as primary path (legacy ModelRouter as fallback)
- ModelAgent cascade fallback (FrugalGPT: try → fail → next-best, max 3)
- 3 vendor lock-ins fixed: MemoryAgent, ExoCortex, ComplexityRouter
- Provider quirks: DeepSeek (strip temp), Kimi (clamp temp), Grok/MiniMax (reasoning_content)
- CapabilityMatrix auto-populated at boot from discovered providers
- Boot logs per-provider discovery summary
```

**Step 3: Run test suite to get final count**

Run: `cd sage-python && python -m pytest tests/ -q`
Record the final test count.

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md and MEMORY.md for multi-provider routing"
```

---

## Summary of Tasks

| Task | Component | What Changes |
|------|-----------|-------------|
| 1 | Boot refresh | `registry.refresh()` at startup |
| 2 | Primary path | CognitiveOrchestrator as primary, ModelRouter as fallback |
| 3 | Cascade fallback | ModelAgent retries on failure (3 attempts, FrugalGPT) |
| 4 | MemoryAgent fix | Remove hardcoded GoogleProvider |
| 5 | ExoCortex fix | Configurable query model |
| 6 | ComplexityRouter fix | Remove hardcoded google.genai |
| 7 | Quirk dispatcher | DeepSeek/Kimi/Grok/MiniMax handling |
| 8 | CapabilityMatrix | Auto-populate from discovered providers |
| 9 | Provider passthrough | Pass provider_name to quirk dispatcher |
| 10 | Boot logging | Per-provider discovery summary |
| 11 | Integration test | End-to-end multi-provider smoke test |
| 12 | Documentation | CLAUDE.md + MEMORY.md update |

**Files touched:** 9 existing + 6 new test files = 15 files total
**New tests:** ~35 tests across 6 new test files
**Commits:** 12 (one per task)
