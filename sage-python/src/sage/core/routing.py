"""Rust SystemRouter Python wrapper (Phase 8 boundary)."""

from __future__ import annotations

from typing import Any

import sage_core


class RustSystemRouter:
    """Thin façade over ``sage_core.SystemRouter``."""

    def __init__(self, registry: Any) -> None:
        self._inner = sage_core.SystemRouter(registry)

    def route(self, task: str, budget: float = 1.0) -> Any:
        return self._inner.route(task, budget)

    def route_constrained(self, task: str, constraints: Any) -> Any:
        return self._inner.route_constrained(task, constraints)
