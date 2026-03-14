"""Phase modules for the structured perceive -> think -> act -> learn agent loop.

Each phase is extracted from agent_loop.py's run() method to improve
maintainability while preserving identical behavior.

Phase functions can be imported from their modules or from this package:
    from sage.phases.perceive import perceive
    from sage.phases import perceive  # also works (lazy import)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LoopContext:
    """Shared state passed between perceive/think/act/learn phases."""

    task: str
    messages: list[dict[str, Any]]
    step: int = 0
    done: bool = False
    result_text: str = ""
    cost: float = 0.0
    routing_decision: Any = None
    tool_calls: list[Any] = field(default_factory=list)
    has_tool_calls: bool = False
    guardrail_results: list[Any] = field(default_factory=list)
    is_code_task: bool = False
    validation_level: str = "default"
    topology_result: str | None = None


# Lazy re-exports to avoid circular imports (phase modules import agent_loop).
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "perceive": ("sage.phases.perceive", "perceive"),
    "think": ("sage.phases.think", "think"),
    "act": ("sage.phases.act", "act"),
    "learn_step": ("sage.phases.learn", "learn_step"),
    "learn_final": ("sage.phases.learn", "learn_final"),
    # Also allow importing the learn module itself
    "learn": ("sage.phases", "learn"),
}


def __getattr__(name: str) -> Any:  # noqa: ANN401
    if name in _LAZY_IMPORTS:
        mod_path, attr = _LAZY_IMPORTS[name]
        if name == "learn":
            # Import the learn module
            import importlib
            return importlib.import_module("sage.phases.learn")
        import importlib
        mod = importlib.import_module(mod_path)
        return getattr(mod, attr)
    raise AttributeError(f"module 'sage.phases' has no attribute {name!r}")


__all__ = [
    "LoopContext",
    "perceive",
    "think",
    "act",
    "learn_step",
    "learn_final",
]
