"""Runtime safety gates for dynamic tool execution."""

from __future__ import annotations

import importlib
import os
from typing import Any

UNSAFE_PY_SUBPROCESS_ENV = "SAGE_UNSAFE_PY_SUBPROCESS"

_TRUE_VALUES = {"1", "true", "yes", "on"}


def env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def unsafe_py_subprocess_enabled() -> bool:
    """Return whether legacy Python subprocess fallback is explicitly allowed."""
    return env_truthy(UNSAFE_PY_SUBPROCESS_ENV)


def load_tool_executor_or_raise() -> type[Any]:
    """Load sage_core.ToolExecutor or raise a clear fail-closed ImportError."""
    try:
        sage_core = importlib.import_module("sage_core")
        tool_executor = getattr(sage_core, "ToolExecutor")
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "Dynamic Python tool creation requires sage_core.ToolExecutor "
            "(Rust tree-sitter validator + Wasm sandbox). The legacy Python "
            "subprocess fallback is disabled by default because it provides "
            "timeout isolation only and does not enforce the ADR-013 sandbox "
            "contract. Build and install sage-core with maturin, or set "
            f"{UNSAFE_PY_SUBPROCESS_ENV}=1 for local development only."
        ) from exc

    return tool_executor
