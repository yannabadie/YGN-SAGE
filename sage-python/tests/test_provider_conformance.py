"""Provider conformance: every provider must expose capabilities()."""
from __future__ import annotations

import pytest


def test_openai_compat_capabilities():
    """OpenAICompatProvider must declare its limitations."""
    from sage.providers.openai_compat import OpenAICompatProvider

    provider = OpenAICompatProvider(
        api_key="test", base_url="http://fake", model_id="test"
    )
    caps = provider.capabilities()
    assert isinstance(caps, dict)
    assert caps["tool_role"] is False
    assert caps["file_search"] is False
    assert "structured_output" in caps


def test_google_provider_capabilities():
    """GoogleProvider must declare its capabilities."""
    from sage.llm.google import GoogleProvider

    provider = GoogleProvider(api_key="test-key")
    caps = provider.capabilities()
    assert isinstance(caps, dict)
    assert caps["tool_role"] is True
    assert caps["file_search"] is True
    assert caps["structured_output"] is True


def test_codex_provider_capabilities():
    """CodexProvider must declare its capabilities."""
    from sage.llm.codex import CodexProvider

    provider = CodexProvider()
    caps = provider.capabilities()
    assert isinstance(caps, dict)
    assert caps["system_prompt"] is False  # Codex skips system messages
    assert "tool_role" in caps
