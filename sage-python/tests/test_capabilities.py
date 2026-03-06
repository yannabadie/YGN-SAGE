"""Tests for CapabilityMatrix and ProviderCapabilities."""
from __future__ import annotations

import pytest
from sage.providers.capabilities import CapabilityMatrix, ProviderCapabilities


def test_matrix_registers_provider():
    matrix = CapabilityMatrix()
    caps = ProviderCapabilities(
        provider="google", structured_output=True, tool_role=True,
        file_search=True, grounding=True,
    )
    matrix.register(caps)
    assert matrix.get("google").structured_output is True


def test_matrix_check_requirement():
    matrix = CapabilityMatrix()
    matrix.register(ProviderCapabilities(provider="google", tool_role=True))
    matrix.register(ProviderCapabilities(provider="xai", tool_role=False))
    compatible = matrix.providers_for(tool_role=True)
    assert "google" in compatible
    assert "xai" not in compatible


def test_matrix_hard_fail_on_missing():
    matrix = CapabilityMatrix()
    matrix.register(ProviderCapabilities(provider="mock", tool_role=False))
    with pytest.raises(ValueError, match="No provider supports"):
        matrix.require(tool_role=True)


def test_matrix_require_returns_compatible():
    matrix = CapabilityMatrix()
    matrix.register(ProviderCapabilities(provider="google", file_search=True))
    matrix.register(ProviderCapabilities(provider="xai", file_search=False))
    result = matrix.require(file_search=True)
    assert result == ["google"]
