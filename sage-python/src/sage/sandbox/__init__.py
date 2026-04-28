"""Sandbox system for isolated tool execution.

`errors` is always importable. Other names are lazy via __getattr__ so
`from sage.sandbox.errors import SandboxUnavailable` works even when
`sage.sandbox.isolated_executor` is unavailable.
"""
from typing import Any

from sage.sandbox.errors import SandboxUnavailable

__all__ = [
    "SandboxUnavailable",
    "SandboxManager",
    "SandboxConfig",
    "execute_isolated",
    "BWRAP_AVAILABLE",
]


def __getattr__(name: str) -> Any:
    if name in {"SandboxManager", "SandboxConfig"}:
        from sage.sandbox.manager import SandboxConfig, SandboxManager

        return {"SandboxManager": SandboxManager, "SandboxConfig": SandboxConfig}[name]
    if name in {"execute_isolated", "BWRAP_AVAILABLE"}:
        from sage.sandbox.isolated_executor import BWRAP_AVAILABLE, execute_isolated

        return {"execute_isolated": execute_isolated, "BWRAP_AVAILABLE": BWRAP_AVAILABLE}[
            name
        ]
    raise AttributeError(f"module 'sage.sandbox' has no attribute {name!r}")
