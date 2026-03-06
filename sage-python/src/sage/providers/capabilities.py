"""Semantic capability matrix — hard-fail when required features missing."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderCapabilities:
    provider: str
    structured_output: bool = False
    tool_role: bool = False
    file_search: bool = False
    grounding: bool = False
    system_prompt: bool = True
    streaming: bool = False


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
