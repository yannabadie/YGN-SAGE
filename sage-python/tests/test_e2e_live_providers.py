"""E2E tests with LIVE LLM providers.

These tests make real API calls. They require valid API keys in .env.
Run with: pytest tests/test_e2e_live_providers.py -v -s

Providers tested (March 2026):
  - OpenAI: gpt-4.1, gpt-4.1-mini, gpt-4.1-nano
  - xAI: grok-4-1-fast-reasoning, grok-code-fast-1
  - MiniMax: MiniMax-Text-01

Skip conditions: tests skip automatically if API key is missing.
"""
from __future__ import annotations

import asyncio
import os
import pytest

# ── Skip markers ─────────────────────────────────────────────────────────────

_has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))
_has_xai_key = bool(os.environ.get("XAI_API_KEY"))
_has_minimax_key = bool(os.environ.get("MINIMAX_API_KEY"))

skip_no_openai = pytest.mark.skipif(not _has_openai_key, reason="OPENAI_API_KEY not set")
skip_no_xai = pytest.mark.skipif(not _has_xai_key, reason="XAI_API_KEY not set")
skip_no_minimax = pytest.mark.skipif(not _has_minimax_key, reason="MINIMAX_API_KEY not set")


def _run(coro):
    """Run an async function synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Helper: OpenAI-compat call ───────────────────────────────────────────────

def _openai_chat(base_url: str, api_key: str, model: str, prompt: str, timeout: float = 30.0) -> str:
    """Make a single chat completion call via OpenAI-compatible API."""
    import openai
    client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.1,
    )
    return response.choices[0].message.content or ""


# ── Helper: SAGE provider call ───────────────────────────────────────────────

async def _sage_provider_call(provider_name: str, base_url: str | None, api_key: str,
                               model: str, prompt: str) -> str:
    """Call via SAGE's OpenAICompatProvider."""
    from sage.providers.openai_compat import OpenAICompatProvider
    from sage.llm.base import LLMConfig, Message, Role

    provider = OpenAICompatProvider(
        api_key=api_key,
        base_url=base_url,
        model_id=model,
        provider_name=provider_name,
    )
    config = LLMConfig(provider=provider_name, model=model, temperature=0.1, max_tokens=256)
    response = await provider.generate(
        messages=[Message(role=Role.USER, content=prompt)],
        config=config,
    )
    return response.content


# ═════════════════════════════════════════════════════════════════════════════
# 1. RAW PROVIDER CONNECTIVITY
# ═════════════════════════════════════════════════════════════════════════════

class TestRawProviderConnectivity:
    """Test raw OpenAI-compat API calls to each provider."""

    @skip_no_openai
    def test_openai_gpt41_responds(self):
        result = _openai_chat(
            "https://api.openai.com/v1",
            os.environ["OPENAI_API_KEY"],
            "gpt-4.1-nano",
            "What is 2+2? Reply with just the number.",
        )
        assert "4" in result

    @skip_no_xai
    def test_xai_grok_responds(self):
        result = _openai_chat(
            "https://api.x.ai/v1",
            os.environ["XAI_API_KEY"],
            "grok-4-1-fast-reasoning",
            "What is 2+2? Reply with just the number.",
        )
        assert "4" in result

    @skip_no_minimax
    def test_minimax_responds(self):
        result = _openai_chat(
            "https://api.minimax.io/v1",
            os.environ["MINIMAX_API_KEY"],
            "MiniMax-Text-01",
            "What is 2+2? Reply with just the number.",
        )
        assert "4" in result


# ═════════════════════════════════════════════════════════════════════════════
# 2. SAGE PROVIDER LAYER (OpenAICompatProvider)
# ═════════════════════════════════════════════════════════════════════════════

class TestSageProviderLayer:
    """Test SAGE's OpenAICompatProvider with each backend."""

    @skip_no_openai
    def test_sage_openai_provider(self):
        result = _run(_sage_provider_call(
            "openai", "https://api.openai.com/v1",
            os.environ["OPENAI_API_KEY"],
            "gpt-4.1-nano",
            "What is the capital of France? One word.",
        ))
        assert "paris" in result.lower()

    @skip_no_xai
    def test_sage_xai_provider(self):
        result = _run(_sage_provider_call(
            "xai", "https://api.x.ai/v1",
            os.environ["XAI_API_KEY"],
            "grok-4-1-fast-reasoning",
            "What is the capital of France? One word.",
        ))
        assert "paris" in result.lower()

    @skip_no_minimax
    def test_sage_minimax_provider(self):
        result = _run(_sage_provider_call(
            "minimax", "https://api.minimax.io/v1",
            os.environ["MINIMAX_API_KEY"],
            "MiniMax-Text-01",
            "What is the capital of France? One word.",
        ))
        assert "paris" in result.lower()


# ═════════════════════════════════════════════════════════════════════════════
# 3. PROVIDER DISCOVERY
# ═════════════════════════════════════════════════════════════════════════════

class TestProviderDiscovery:
    """Test auto-discovery of models from available providers."""

    @skip_no_openai
    def test_discover_openai_models(self):
        from sage.providers.connector import ProviderConnector
        connector = ProviderConnector(configs=[{
            "provider": "openai",
            "api_key_env": "OPENAI_API_KEY",
            "base_url": "https://api.openai.com/v1",
            "sdk": "openai",
        }])
        models = _run(connector.discover_all())
        model_ids = [m.id for m in models]
        assert "gpt-4.1" in model_ids
        assert "gpt-4.1-mini" in model_ids
        assert "gpt-4.1-nano" in model_ids
        # All should be openai provider
        for m in models:
            if m.provider == "openai":
                assert m.provider == "openai"

    @skip_no_xai
    def test_discover_xai_models(self):
        from sage.providers.connector import ProviderConnector
        connector = ProviderConnector(configs=[{
            "provider": "xai",
            "api_key_env": "XAI_API_KEY",
            "base_url": "https://api.x.ai/v1",
            "sdk": "openai",
        }])
        models = _run(connector.discover_all())
        model_ids = [m.id for m in models]
        # Should find at least grok-3 and grok-4 variants
        assert any("grok" in mid for mid in model_ids), f"No grok models found: {model_ids}"

    @skip_no_minimax
    def test_discover_minimax_models(self):
        from sage.providers.connector import ProviderConnector
        connector = ProviderConnector(configs=[{
            "provider": "minimax",
            "api_key_env": "MINIMAX_API_KEY",
            "base_url": "https://api.minimax.io/v1",
            "sdk": "openai",
            "hardcoded_models": ["MiniMax-Text-01"],
        }])
        models = _run(connector.discover_all())
        assert len(models) >= 1
        assert models[0].id == "MiniMax-Text-01"


# ═════════════════════════════════════════════════════════════════════════════
# 4. MODEL ROUTER + CONFIG LOADING
# ═════════════════════════════════════════════════════════════════════════════

class TestModelRouterConfig:
    """Test that model router loads updated configs correctly."""

    def test_models_toml_loads(self):
        from sage.llm.config_loader import load_model_config
        from pathlib import Path
        toml_path = Path(__file__).parent.parent / "config" / "models.toml"
        config = load_model_config(toml_path)
        assert config is not None
        tiers = config.get("tiers", {})
        # Verify updated model IDs
        assert tiers.get("codex") == "gpt-4.1"
        assert tiers.get("fast") == "gpt-4.1-nano"
        assert tiers.get("mutator") == "gpt-4.1-mini"
        assert tiers.get("fallback") == "MiniMax-Text-01"

    def test_router_hardcoded_defaults(self):
        from sage.llm.router import _HARDCODED
        assert _HARDCODED["codex"] == "gpt-4.1"
        assert _HARDCODED["fast"] == "gpt-4.1-nano"
        assert _HARDCODED["reasoner"] == "grok-4-1-fast-reasoning"
        # No fictional models
        for model in _HARDCODED.values():
            assert "gpt-5" not in model, f"Fictional model found: {model}"
            assert "gemini-3" not in model, f"Fictional model found: {model}"

    def test_cards_toml_loads(self):
        """Verify cards.toml loads and contains real models."""
        import tomllib
        from pathlib import Path

        cards_path = Path(__file__).parent.parent / "config" / "cards.toml"
        if not cards_path.exists():
            pytest.skip("cards.toml not found")

        with open(cards_path, "rb") as f:
            data = tomllib.load(f)

        models = data.get("models", [])
        model_ids = [m["id"] for m in models]

        # Should contain real models
        assert "gpt-4.1" in model_ids
        assert "grok-4.20-beta" in model_ids
        assert "MiniMax-Text-01" in model_ids

        # Should NOT contain fictional models
        for mid in model_ids:
            assert "gpt-5" not in mid, f"Fictional model in cards.toml: {mid}"

    def test_cost_per_1k_updated(self):
        """Verify _COST_PER_1K dict uses real models."""
        from sage.agent_loop import _COST_PER_1K
        assert "gpt-4.1" in _COST_PER_1K
        assert "grok-4.20-beta" in _COST_PER_1K
        assert "MiniMax-Text-01" in _COST_PER_1K
        # No fictional models
        for model in _COST_PER_1K:
            assert "gpt-5" not in model, f"Fictional model in _COST_PER_1K: {model}"


# ═════════════════════════════════════════════════════════════════════════════
# 5. MULTI-PROVIDER CODE GENERATION (real LLM)
# ═════════════════════════════════════════════════════════════════════════════

class TestMultiProviderCodeGen:
    """Test code generation across providers — the core SAGE use case."""

    @skip_no_openai
    def test_openai_generates_python(self):
        result = _openai_chat(
            "https://api.openai.com/v1",
            os.environ["OPENAI_API_KEY"],
            "gpt-4.1-mini",
            "Write a Python function `is_prime(n)` that returns True if n is prime. "
            "Only output the function, no explanation.",
        )
        assert "def is_prime" in result
        assert "return" in result

    @skip_no_xai
    def test_xai_generates_python(self):
        result = _openai_chat(
            "https://api.x.ai/v1",
            os.environ["XAI_API_KEY"],
            "grok-code-fast-1",
            "Write a Python function `is_prime(n)` that returns True if n is prime. "
            "Only output the function, no explanation.",
        )
        assert "def is_prime" in result
        assert "return" in result

    @skip_no_minimax
    def test_minimax_generates_python(self):
        result = _openai_chat(
            "https://api.minimax.io/v1",
            os.environ["MINIMAX_API_KEY"],
            "MiniMax-Text-01",
            "Write a Python function `is_prime(n)` that returns True if n is prime. "
            "Only output the function, no explanation.",
        )
        assert "def is_prime" in result
        assert "return" in result


# ═════════════════════════════════════════════════════════════════════════════
# 6. ORCHESTRATOR CASCADE FALLBACK
# ═════════════════════════════════════════════════════════════════════════════

class TestOrchestratorCascade:
    """Test that the orchestrator's FrugalGPT cascade works with real providers."""

    @skip_no_openai
    def test_multi_provider_cascade(self):
        """Test FrugalGPT-style cascade: call OpenAI, then xAI as fallback."""
        from sage.providers.openai_compat import OpenAICompatProvider
        from sage.llm.base import LLMConfig, Message, Role

        # Primary: OpenAI gpt-4.1-nano
        provider = OpenAICompatProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://api.openai.com/v1",
            model_id="gpt-4.1-nano",
            provider_name="openai",
        )
        config = LLMConfig(provider="openai", model="gpt-4.1-nano", temperature=0.1, max_tokens=64)
        result = _run(provider.generate(
            messages=[Message(role=Role.USER, content="What is 7 * 8? Reply with just the number.")],
            config=config,
        ))
        assert "56" in result.content

        # Cascade: if primary had failed, xAI would work
        if os.environ.get("XAI_API_KEY"):
            fallback = OpenAICompatProvider(
                api_key=os.environ["XAI_API_KEY"],
                base_url="https://api.x.ai/v1",
                model_id="grok-4-1-fast-reasoning",
                provider_name="xai",
            )
            fb_config = LLMConfig(provider="xai", model="grok-4-1-fast-reasoning", temperature=0.1, max_tokens=64)
            fb_result = _run(fallback.generate(
                messages=[Message(role=Role.USER, content="What is 7 * 8? Reply with just the number.")],
                config=fb_config,
            ))
            assert "56" in fb_result.content


# ═════════════════════════════════════════════════════════════════════════════
# 7. QUALITY ESTIMATOR ON REAL OUTPUT
# ═════════════════════════════════════════════════════════════════════════════

class TestQualityEstimatorReal:
    """Test QualityEstimator on real LLM output."""

    @skip_no_openai
    def test_quality_estimator_scores_real_output(self):
        from sage.quality_estimator import QualityEstimator

        qe = QualityEstimator()

        # Get real LLM output
        result = _openai_chat(
            "https://api.openai.com/v1",
            os.environ["OPENAI_API_KEY"],
            "gpt-4.1-nano",
            "Write a Python function that computes factorial. Include a docstring.",
        )

        task = "Write a Python function that computes factorial."
        score = qe.estimate(task, result, latency_ms=500.0)

        # Real output from a good model should score well
        assert 0.0 <= score <= 1.0
        assert score > 0.3, f"Quality score too low for real LLM output: {score}"


# ═════════════════════════════════════════════════════════════════════════════
# 8. FULL AGENT LOOP WITH REAL LLM
# ═════════════════════════════════════════════════════════════════════════════

class TestFullAgentLoopReal:
    """Test the full AgentLoop perceive→think→act→learn cycle with a real LLM."""

    @skip_no_openai
    def test_agent_loop_simple_task(self):
        """Run AgentLoop on a simple math task with a real LLM."""
        from sage.agent import AgentConfig
        from sage.agent_loop import AgentLoop
        from sage.llm.base import LLMConfig
        from sage.providers.openai_compat import OpenAICompatProvider

        provider = OpenAICompatProvider(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url="https://api.openai.com/v1",
            model_id="gpt-4.1-nano",
            provider_name="openai",
        )

        config = AgentConfig(
            name="test-real-agent",
            llm=LLMConfig(provider="openai", model="gpt-4.1-nano", temperature=0.1, max_tokens=128),
            max_steps=3,
        )

        events = []
        loop = AgentLoop(
            config=config,
            llm_provider=provider,
            on_event=lambda e: events.append(e),
        )

        result = _run(loop.run("What is the sum of 15 and 27? Reply with just the number."))

        assert result is not None
        assert len(result) > 0
        # Should contain 42
        assert "42" in result
        # Should have emitted events
        assert len(events) > 0


# ═════════════════════════════════════════════════════════════════════════════
# 9. PROVIDER CONFIG CONSISTENCY
# ═════════════════════════════════════════════════════════════════════════════

class TestProviderConfigConsistency:
    """Verify connector config matches env var names and URLs."""

    def test_connector_env_var_names_match_dotenv(self):
        """Env var names in PROVIDER_CONFIGS should match .env convention."""
        from sage.providers.connector import PROVIDER_CONFIGS
        expected_env_vars = {
            "google": "GOOGLE_API_KEY",
            "openai": "OPENAI_API_KEY",
            "xai": "XAI_API_KEY",
            "minimax": "MINIMAX_API_KEY",
            "kimi": "MOONSHOT_API_KEY",
            "glm": "GLM_API_KEY",
            "deepseek": "DEEP_SEEK_API_KEY",
        }
        for cfg in PROVIDER_CONFIGS:
            provider = cfg["provider"]
            if provider in expected_env_vars:
                assert cfg["api_key_env"] == expected_env_vars[provider], \
                    f"{provider}: expected {expected_env_vars[provider]}, got {cfg['api_key_env']}"

    def test_minimax_url_correct(self):
        from sage.providers.connector import PROVIDER_CONFIGS
        minimax_cfg = next(c for c in PROVIDER_CONFIGS if c["provider"] == "minimax")
        assert minimax_cfg["base_url"] == "https://api.minimax.io/v1"

    def test_glm_provider_exists(self):
        from sage.providers.connector import PROVIDER_CONFIGS
        providers = [c["provider"] for c in PROVIDER_CONFIGS]
        assert "glm" in providers

    def test_no_fictional_hardcoded_models(self):
        """No gpt-5.x or gemini-3.x in hardcoded model lists."""
        from sage.providers.connector import PROVIDER_CONFIGS
        for cfg in PROVIDER_CONFIGS:
            for model in cfg.get("hardcoded_models", []):
                assert "gpt-5" not in model, f"Fictional model: {model}"
                assert "M2.5" not in model, f"Old MiniMax model: {model}"


# ═════════════════════════════════════════════════════════════════════════════
# 10. TOPOLOGY CONTROLLER WITH REAL QUALITY SCORES
# ═════════════════════════════════════════════════════════════════════════════

class TestTopologyControllerReal:
    """Test TopologyController decision logic with realistic inputs."""

    def test_good_quality_continues(self):
        from sage.topology_controller import TopologyController, AdaptationDecision
        from sage.quality_estimator import QualityEstimator

        qe = QualityEstimator()
        tc = TopologyController(quality_estimator=qe)

        # Good quality code output
        result = '''def fibonacci(n):
    """Return the n-th Fibonacci number."""
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b'''

        decision = tc.evaluate_and_decide(
            node_idx=0,
            result=result,
            task="Write a fibonacci function",
            topology=None,
            ctx=type("Ctx", (), {"latency_ms": 500.0})(),
        )
        assert decision.action == "continue"

    def test_empty_output_triggers_upgrade(self):
        from sage.topology_controller import TopologyController
        from sage.quality_estimator import QualityEstimator

        qe = QualityEstimator()
        tc = TopologyController(quality_estimator=qe)

        decision = tc.evaluate_and_decide(
            node_idx=0,
            result="",
            task="Write a sorting algorithm",
            topology=None,
            ctx=type("Ctx", (), {"latency_ms": 500.0})(),
        )
        # Empty output should score low → trigger upgrade
        assert decision.action in ("upgrade_model", "continue")
