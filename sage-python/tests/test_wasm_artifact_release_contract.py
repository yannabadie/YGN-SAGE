"""Sandbox release-grade contract tests — Phase 4 (AUDITRUST.md).

Ensures no wheel ships without a real, working Wasm artifact.
"""

from __future__ import annotations

import os

import pytest


def test_embedded_wasm_available_in_release_contract():
    """When SAGE_RELEASE_CONTRACT=1, the embedded Wasm must be available."""
    import sage_core

    if os.environ.get("SAGE_RELEASE_CONTRACT") != "1":
        pytest.skip("release-only contract")

    assert sage_core.embedded_wasm_available() is True, (
        "embedded_wasm_available() returned False under SAGE_RELEASE_CONTRACT=1. "
        "Release wheels must bundle a real RustPython wasm artifact."
    )


def test_embedded_wasm_available_is_bool():
    """embedded_wasm_available() must return a concrete bool (not None)."""
    import sage_core

    available = sage_core.embedded_wasm_available()
    assert isinstance(available, bool), (
        f"embedded_wasm_available() returned {type(available).__name__}, "
        f"expected bool."
    )


def test_tool_executor_validate_accepts_safe_code():
    """ToolExecutor.validate must accept safe Python code (Wasm runtime)."""
    import sage_core

    executor = sage_core.ToolExecutor()
    result = executor.validate("print('ok')")
    assert result.valid, f"validate() rejected safe code: {result}"


def test_tool_executor_validate_rejects_dangerous_code():
    """ToolExecutor.validate must reject code with dangerous imports."""
    import sage_core

    executor = sage_core.ToolExecutor()
    result = executor.validate("import os; os.system('rm -rf /')")
    assert not result.valid, (
        "validate() accepted code with os.system — sandbox must reject"
    )
