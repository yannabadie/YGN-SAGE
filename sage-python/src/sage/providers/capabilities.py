"""Semantic capability matrix — hard-fail when required features missing."""
from __future__ import annotations

from dataclasses import dataclass


_KNOWN_CAPABILITIES: dict[str, dict[str, bool]] = {
    "google": {
        "structured_output": True, "tool_role": True, "file_search": True,
        "grounding": True, "system_prompt": True, "streaming": True,
    },
    "openai": {
        "structured_output": True, "tool_role": True, "file_search": False,
        "grounding": False, "system_prompt": True, "streaming": True,
    },
    "codex": {
        "structured_output": False, "tool_role": False, "file_search": False,
        "grounding": False, "system_prompt": False, "streaming": False,
    },
    "xai": {
        "structured_output": True, "tool_role": True, "file_search": False,
        "grounding": False, "system_prompt": True, "streaming": False,
    },
    "deepseek": {
        "structured_output": True, "tool_role": True, "file_search": False,
        "grounding": False, "system_prompt": True, "streaming": False,
    },
    "minimax": {
        "structured_output": True, "tool_role": True, "file_search": False,
        "grounding": False, "system_prompt": True, "streaming": False,
    },
    "kimi": {
        "structured_output": True, "tool_role": True, "file_search": False,
        "grounding": False, "system_prompt": True, "streaming": False,
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
