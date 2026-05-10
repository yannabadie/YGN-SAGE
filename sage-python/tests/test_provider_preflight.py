"""Tests for provider preflight artifact accounting."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT_PATH = (
    Path(__file__).parent.parent / "scripts" / "provider_preflight.py"
).resolve()
_SPEC = importlib.util.spec_from_file_location("provider_preflight", _SCRIPT_PATH)
provider_preflight = importlib.util.module_from_spec(_SPEC)
sys.modules["provider_preflight"] = provider_preflight
assert _SPEC.loader is not None
_SPEC.loader.exec_module(provider_preflight)


def test_cost_from_usage_prefers_provider_reported_cost() -> None:
    usage = {"input_tokens": 10, "output_tokens": 5, "cost_usd": 0.0123}

    cost = provider_preflight._cost_from_usage(
        usage,
        "deepseek",
        "deepseek-v4-flash",
        lookup_cost_per_token=lambda provider, model: (1.0, 1.0),
    )

    assert cost == pytest.approx(0.0123)


def test_cost_from_usage_dict_uses_per_token_rates_without_second_division() -> None:
    usage = {"input_tokens": 1000, "output_tokens": 250}

    cost = provider_preflight._cost_from_usage(
        usage,
        "deepseek",
        "deepseek-v4-flash",
        lookup_cost_per_token=lambda provider, model: (0.000001, 0.000002),
    )

    assert cost == pytest.approx(0.0015)


def test_cost_from_usage_object_shape() -> None:
    usage = SimpleNamespace(input_tokens=100, output_tokens=20)

    cost = provider_preflight._cost_from_usage(
        usage,
        "google",
        "gemini-2.5-flash",
        lookup_cost_per_token=lambda provider, model: (0.0000001, 0.0000004),
    )

    assert cost == pytest.approx(0.000018)


def test_provider_result_labels_preflight_as_liveness_only() -> None:
    result = provider_preflight.ProviderResult(
        provider="openai",
        model_id="gpt-5.5-pro",
        status="ok",
    )

    as_dict = result.as_dict()

    assert as_dict["evidence_scope"] == "liveness_only"


def test_xai_grok_code_fast_warns_before_retirement() -> None:
    warnings = provider_preflight._preflight_warnings("xai", "grok-code-fast-1")

    assert warnings
    assert "2026-05-15T19:00:00Z" in warnings[0]


@pytest.mark.asyncio
async def test_non_exact_smoke_output_warns_but_stays_liveness_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeResponse:
        content = "ok plus explanation"
        usage = {}

    class FakeProvider:
        async def generate(self, **_kwargs):
            return FakeResponse()

    class FakePydanticProvider:
        @classmethod
        def for_sage_provider(cls, *_args, **_kwargs):
            return FakeProvider()

    monkeypatch.setitem(
        sys.modules,
        "sage.providers.pydantic_ai_provider",
        SimpleNamespace(
            PydanticAIProvider=FakePydanticProvider,
            _lookup_cost_per_token=lambda _provider, _model: (0.0, 0.0),
        ),
    )

    result = await provider_preflight._test_one(
        "minimax", "MiniMax-M2.7", "key", timeout=1.0
    )

    assert result.status == "ok"
    assert result.evidence_scope == "liveness_only"
    assert any("not exactly 'ok'" in warning for warning in result.warnings)
