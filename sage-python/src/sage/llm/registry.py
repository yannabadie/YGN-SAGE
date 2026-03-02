"""Registry for LLM providers."""
from __future__ import annotations

from typing import Any


class LLMRegistry:
    """Central registry for LLM provider classes."""

    def __init__(self) -> None:
        self._providers: dict[str, Any] = {}

    def register(self, name: str, provider_cls: Any) -> None:
        """Register a provider class by name."""
        self._providers[name] = provider_cls

    def get(self, name: str) -> Any:
        """Get a provider class by name. Raises KeyError if not found."""
        if name not in self._providers:
            raise KeyError(f"Unknown LLM provider: {name!r}. Available: {list(self._providers)}")
        return self._providers[name]

    def list_providers(self) -> list[str]:
        """List all registered provider names."""
        return list(self._providers.keys())


# Global registry instance
default_registry = LLMRegistry()
