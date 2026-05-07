"""Rust ToolExecutor Python wrapper (Phase 8 boundary)."""

from __future__ import annotations

from typing import Any

import sage_core


class RustToolExecutor:
    """Thin façade over ``sage_core.ToolExecutor``."""

    def __init__(self) -> None:
        self._inner = sage_core.ToolExecutor()

    def validate(self, code: str) -> Any:
        return self._inner.validate(code)

    def validate_and_execute(self, code: str) -> Any:
        return self._inner.validate_and_execute(code)
