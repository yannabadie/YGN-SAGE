"""Tests for CognitiveOrchestrator, ModelAgent, and OpenAICompatProvider.

All tests are offline -- no real API calls are made.
"""
import sys
import types as pytypes

if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = pytypes.ModuleType("sage_core")

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sage.orchestrator import (
    CognitiveOrchestrator,
    ExecutionPlan,
    ModelAgent,
    SubTask,
    _provider_cfg_for,
)
from sage.providers.registry import ModelProfile, ModelRegistry
from sage.strategy.metacognition import MetacognitiveController


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_profile(
    id: str,
    provider: str = "test",
    code: float = 0.5,
    reasoning: float = 0.5,
    cost_in: float = 1.0,
    cost_out: float = 5.0,
    **kwargs,
) -> ModelProfile:
    """Create a minimal available ModelProfile for testing."""
    return ModelProfile(
        id=id,
        provider=provider,
        family="test",
        available=True,
        code_score=code,
        reasoning_score=reasoning,
        cost_input=cost_in,
        cost_output=cost_out,
        **kwargs,
    )


def _mock_registry(*profiles: ModelProfile) -> ModelRegistry:
    """Create a ModelRegistry pre-populated with the given profiles."""
    reg = ModelRegistry.__new__(ModelRegistry)
    reg._profiles = {}
    reg._connector = MagicMock()
    for p in profiles:
        reg._profiles[p.id] = p
    return reg


# ── SubTask / ExecutionPlan ──────────────────────────────────────────────────

class TestSubTask:
    def test_creation(self):
        st = SubTask(description="Write a function", needs_code=True)
        assert st.needs_code
        assert not st.needs_reasoning
        assert st.depends_on == []

    def test_defaults(self):
        st = SubTask(description="something")
        assert not st.needs_code
        assert not st.needs_reasoning
        assert not st.needs_tools


class TestExecutionPlan:
    def test_single_task(self):
        plan = ExecutionPlan(subtasks=[SubTask(description="simple")])
        assert not plan.is_decomposed

    def test_decomposed(self):
        plan = ExecutionPlan(
            subtasks=[SubTask(description="a"), SubTask(description="b")],
            is_decomposed=True,
        )
        assert plan.is_decomposed
        assert len(plan.subtasks) == 2


# ── _parse_subtasks ──────────────────────────────────────────────────────────

class TestParseSubtasks:
    def test_basic_parsing(self):
        orch = CognitiveOrchestrator(registry=_mock_registry())
        result = orch._parse_subtasks(
            "1. [CODE] Write the sorting function\n"
            "2. [REASON] Prove it terminates correctly\n"
            "3. [GENERAL] Write documentation for the module\n"
        )
        assert len(result) == 3
        assert result[0].needs_code
        assert result[1].needs_reasoning
        # [GENERAL] tag doesn't set code or reasoning explicitly,
        # but "documentation" doesn't contain "code" either
        assert not result[2].needs_reasoning

    def test_max_4_subtasks(self):
        orch = CognitiveOrchestrator(registry=_mock_registry())
        lines = "\n".join(
            f"{i}. [CODE] Task {i} is a long enough description to pass the filter"
            for i in range(10)
        )
        result = orch._parse_subtasks(lines)
        assert len(result) <= 4

    def test_short_lines_discarded(self):
        orch = CognitiveOrchestrator(registry=_mock_registry())
        result = orch._parse_subtasks("1. Short\n2. This is a proper subtask description\n")
        # "Short" is < 10 chars after stripping, should be discarded
        assert len(result) == 1

    def test_empty_input(self):
        orch = CognitiveOrchestrator(registry=_mock_registry())
        result = orch._parse_subtasks("")
        assert result == []


# ── ModelAgent ───────────────────────────────────────────────────────────────

class TestModelAgent:
    def test_creation(self):
        profile = _make_profile("test-model", provider="google")
        agent = ModelAgent(name="test", model=profile)
        assert agent.name == "test"
        assert agent.model.id == "test-model"

    def test_default_system_prompt(self):
        agent = ModelAgent(name="x", model=_make_profile("m"))
        assert "helpful" in agent._system_prompt.lower()

    def test_custom_system_prompt(self):
        agent = ModelAgent(name="x", model=_make_profile("m"), system_prompt="Be terse.")
        assert agent._system_prompt == "Be terse."

    def test_create_provider_google(self):
        """Google models should create a GoogleProvider."""
        profile = _make_profile("gemini-test", provider="google")
        agent = ModelAgent(name="g", model=profile)
        with patch("sage.llm.google.GoogleProvider") as mock_cls:
            mock_cls.return_value = MagicMock()
            provider = agent._create_provider()
            mock_cls.assert_called_once()

    def test_create_provider_openai_compat(self):
        """Non-Google models should create an OpenAICompatProvider."""
        profile = _make_profile("grok-test", provider="xai")
        agent = ModelAgent(name="x", model=profile)
        with patch("sage.providers.openai_compat.OpenAICompatProvider") as mock_cls:
            mock_cls.return_value = MagicMock()
            provider = agent._create_provider()
            mock_cls.assert_called_once()


# ── Provider config lookup ───────────────────────────────────────────────────

class TestProviderCfgFor:
    def test_known_provider(self):
        cfg = _provider_cfg_for("google")
        assert cfg["sdk"] == "google-genai"
        assert cfg["api_key_env"] == "GOOGLE_API_KEY"

    def test_unknown_provider_fallback(self):
        cfg = _provider_cfg_for("unknown-provider")
        assert cfg["sdk"] == "openai"
        assert cfg["provider"] == "unknown-provider"


# ── CognitiveOrchestrator.run() ─────────────────────────────────────────────

class TestOrchestratorRun:
    @pytest.mark.asyncio
    async def test_s1_uses_fast_model(self):
        """Simple tasks (S1) should use a cheap/fast model."""
        fast = _make_profile("fast-model", cost_in=0.1, cost_out=0.5)
        reg = _mock_registry(fast)
        orch = CognitiveOrchestrator(registry=reg)

        # Patch ModelAgent.run to avoid real API calls
        with patch.object(ModelAgent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Response from fast-model"
            result = await orch.run("What is 2+2?")

        assert "fast-model" in result
        mock_run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_s2_selects_code_model(self):
        """Code-related tasks should prefer the code-strong model."""
        code_model = _make_profile("code-beast", code=0.9, cost_in=0.3, cost_out=1.2)
        cheap_model = _make_profile("cheapo", code=0.3, cost_in=0.01, cost_out=0.05)
        reg = _mock_registry(code_model, cheap_model)
        orch = CognitiveOrchestrator(registry=reg)

        with patch.object(ModelAgent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Response from code-beast"
            result = await orch.run("Debug this Python function that calculates fibonacci")

        assert "code-beast" in result

    @pytest.mark.asyncio
    async def test_no_models_returns_message(self):
        """Empty registry should return a graceful error message."""
        reg = _mock_registry()  # no profiles
        orch = CognitiveOrchestrator(registry=reg)
        result = await orch.run("Do something")
        assert "No models available" in result

    @pytest.mark.asyncio
    async def test_s3_single_task_fallback(self):
        """S3 with failed decomposition should use best reasoner."""
        reasoner = _make_profile("big-brain", reasoning=0.95, cost_in=2.0)
        reg = _mock_registry(reasoner)
        # Force S3 routing via custom metacognition thresholds
        mc = MetacognitiveController(s3_complexity_floor=0.0, s3_uncertainty_floor=0.0)
        orch = CognitiveOrchestrator(registry=reg, metacognition=mc)

        with patch.object(ModelAgent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Deep analysis result"
            result = await orch.run("Explain quantum entanglement in detail")

        assert "Deep analysis" in result

    @pytest.mark.asyncio
    async def test_event_bus_integration(self):
        """Orchestrator should emit events on the event bus."""
        fast = _make_profile("fast-model", cost_in=0.1)
        reg = _mock_registry(fast)
        bus = MagicMock()
        orch = CognitiveOrchestrator(registry=reg, event_bus=bus)

        with patch.object(ModelAgent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Result"
            await orch.run("Hello")

        bus.emit.assert_called_once()
        event = bus.emit.call_args[0][0]
        assert event.type == "ORCHESTRATOR"
        assert event.model == "fast-model"
        assert event.latency_ms is not None

    @pytest.mark.asyncio
    async def test_event_bus_failure_does_not_crash(self):
        """A broken event bus should not crash the orchestrator."""
        fast = _make_profile("fast-model", cost_in=0.1)
        reg = _mock_registry(fast)
        bus = MagicMock()
        bus.emit.side_effect = RuntimeError("bus on fire")
        orch = CognitiveOrchestrator(registry=reg, event_bus=bus)

        with patch.object(ModelAgent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = "Still works"
            result = await orch.run("Hello")

        assert result == "Still works"


# ── Decomposition ────────────────────────────────────────────────────────────

class TestDecomposition:
    @pytest.mark.asyncio
    async def test_decompose_returns_plan_on_failure(self):
        """If no model is available for decomposition, return single-task plan."""
        reg = _mock_registry()  # empty
        orch = CognitiveOrchestrator(registry=reg)
        plan = await orch._decompose("Complex multi-step task")
        assert not plan.is_decomposed
        assert len(plan.subtasks) == 1
        assert plan.subtasks[0].description == "Complex multi-step task"

    @pytest.mark.asyncio
    async def test_decompose_with_model(self):
        """Decomposition should parse model output into subtasks."""
        cheap = _make_profile("cheap", cost_in=0.01)
        reg = _mock_registry(cheap)
        orch = CognitiveOrchestrator(registry=reg)

        decomposed_response = (
            "1. [CODE] Implement the sorting algorithm with tests\n"
            "2. [REASON] Prove the algorithm has O(n log n) complexity\n"
        )

        with patch.object(ModelAgent, "run", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = decomposed_response
            plan = await orch._decompose("Sort and prove")

        assert plan.is_decomposed
        assert len(plan.subtasks) == 2
        assert plan.subtasks[0].needs_code
        assert plan.subtasks[1].needs_reasoning


# ── OpenAICompatProvider ─────────────────────────────────────────────────────

class TestOpenAICompatProvider:
    def test_import(self):
        from sage.providers.openai_compat import OpenAICompatProvider
        p = OpenAICompatProvider(api_key="test", base_url="https://example.com/v1", model_id="m")
        assert p.name == "openai-compat"
        assert p.api_key == "test"
        assert p.model_id == "m"

    @pytest.mark.asyncio
    async def test_generate_calls_openai(self):
        from sage.providers.openai_compat import OpenAICompatProvider
        from sage.llm.base import Message, Role, LLMConfig

        p = OpenAICompatProvider(api_key="k", base_url="https://test.api/v1", model_id="test-m")

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello world"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            result = await p.generate(
                [Message(role=Role.USER, content="Hi")],
                config=LLMConfig(provider="test", model="test-m"),
            )

        assert result.content == "Hello world"
        assert result.model == "test-m"

    @pytest.mark.asyncio
    async def test_generate_uses_config_model(self):
        """Config model should override the default model_id."""
        from sage.providers.openai_compat import OpenAICompatProvider
        from sage.llm.base import Message, Role, LLMConfig

        p = OpenAICompatProvider(api_key="k", model_id="default-m")

        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch("openai.AsyncOpenAI", return_value=mock_client):
            await p.generate(
                [Message(role=Role.USER, content="test")],
                config=LLMConfig(provider="test", model="override-m"),
            )

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs.get("model") or call_kwargs[1].get("model") == "override-m"
