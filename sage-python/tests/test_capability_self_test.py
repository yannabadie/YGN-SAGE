"""Test that capability matrix uses runtime adapter capabilities, not static lies."""
import pytest
from sage.providers.capabilities import CapabilityMatrix, ProviderCapabilities


def test_runtime_overrides_static():
    """Provider-reported capabilities must override static claims."""
    matrix = CapabilityMatrix()

    # Simulate: static says structured_output=True, runtime says False
    static_caps = ProviderCapabilities.for_provider("deepseek")
    assert static_caps.structured_output is True  # static lie

    # Register with runtime-reported capabilities
    runtime_caps = ProviderCapabilities(
        provider="deepseek",
        structured_output=False,  # truth from openai_compat
        tool_role=False,
    )
    matrix.register(runtime_caps)

    # Matrix should reflect runtime truth
    assert matrix.get("deepseek").structured_output is False


def test_register_from_adapter():
    """CapabilityMatrix.register_from_adapter() should use adapter.capabilities()."""
    matrix = CapabilityMatrix()
    matrix.register_from_adapter("test_provider", {
        "structured_output": False,
        "tool_role": True,
        "system_prompt": True,
    })
    caps = matrix.get("test_provider")
    assert caps.structured_output is False
    assert caps.tool_role is True


def test_populate_from_providers_uses_adapter_over_static():
    """populate_from_providers with adapter dict should trust adapter.capabilities()."""
    from sage.providers.openai_compat import OpenAICompatProvider

    matrix = CapabilityMatrix()
    # Create a real OpenAICompatProvider for deepseek (no API call — capabilities() is static)
    adapter = OpenAICompatProvider(
        api_key="dummy",
        base_url="https://api.deepseek.com",
        provider_name="deepseek",
    )
    # Verify adapter reports False for structured_output (runtime truth)
    assert adapter.capabilities()["structured_output"] is False

    # Static _KNOWN_CAPABILITIES says True — adapter should win
    matrix.populate_from_providers(["deepseek"], adapters={"deepseek": adapter})
    assert matrix.get("deepseek").structured_output is False


def test_populate_from_providers_fallback_to_static():
    """populate_from_providers without adapter falls back to static claims."""
    matrix = CapabilityMatrix()
    matrix.populate_from_providers(["google"])
    # Google's static caps are correct (structured_output=True)
    assert matrix.get("google").structured_output is True


def test_all_openai_compat_providers_report_false_structured_output():
    """xAI, DeepSeek, MiniMax, Kimi adapters all correctly report structured_output=False."""
    from sage.providers.openai_compat import OpenAICompatProvider

    providers = [
        ("xai", "https://api.x.ai/v1"),
        ("deepseek", "https://api.deepseek.com"),
        ("minimax", "https://api.minimaxi.chat/v1"),
        ("kimi", "https://api.moonshot.ai/v1"),
    ]
    for provider_name, base_url in providers:
        adapter = OpenAICompatProvider(
            api_key="dummy",
            base_url=base_url,
            provider_name=provider_name,
        )
        caps = adapter.capabilities()
        assert caps["structured_output"] is False, (
            f"{provider_name} adapter claims structured_output=True "
            f"but runtime returns False"
        )
