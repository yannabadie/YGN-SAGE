"""Tests for CapabilityMatrix auto-population."""
from sage.providers.capabilities import ProviderCapabilities, CapabilityMatrix


def test_known_google_capabilities():
    caps = ProviderCapabilities.for_provider("google")
    assert caps.file_search is True
    assert caps.structured_output is True
    assert caps.tool_role is True
    assert caps.grounding is True

def test_known_openai_capabilities():
    caps = ProviderCapabilities.for_provider("openai")
    assert caps.structured_output is True
    assert caps.tool_role is True
    assert caps.file_search is False

def test_known_deepseek_capabilities():
    caps = ProviderCapabilities.for_provider("deepseek")
    assert caps.structured_output is True
    assert caps.tool_role is True

def test_known_codex_capabilities():
    caps = ProviderCapabilities.for_provider("codex")
    assert caps.structured_output is False
    assert caps.system_prompt is False

def test_unknown_provider_defaults():
    caps = ProviderCapabilities.for_provider("unknown-llm")
    assert caps.provider == "unknown-llm"
    assert caps.system_prompt is True
    assert caps.file_search is False
    assert caps.structured_output is False

def test_matrix_populate_and_query():
    matrix = CapabilityMatrix()
    matrix.populate_from_providers(["google", "openai", "deepseek"])

    assert len(matrix.providers_for(file_search=True)) == 1
    assert "google" in matrix.providers_for(file_search=True)

    structured = matrix.providers_for(structured_output=True)
    assert "google" in structured
    assert "openai" in structured
    assert "deepseek" in structured

def test_matrix_require_raises_for_impossible():
    matrix = CapabilityMatrix()
    matrix.populate_from_providers(["openai", "deepseek"])
    import pytest
    with pytest.raises(ValueError, match="No provider supports"):
        matrix.require(file_search=True)  # Neither has file_search

def test_matrix_populate_is_idempotent():
    matrix = CapabilityMatrix()
    matrix.populate_from_providers(["google"])
    matrix.populate_from_providers(["google", "openai"])
    assert len(matrix._providers) == 2
