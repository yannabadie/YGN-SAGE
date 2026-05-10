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
