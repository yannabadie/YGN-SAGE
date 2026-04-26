from __future__ import annotations

import os

import pytest


def test_embedded_wasm_available_when_ci_requires_wasm() -> None:
    if os.environ.get("SAGE_CI_REQUIRE_WASM") != "1":
        pytest.skip("CI wasm contract not requested")

    import sage_core

    assert sage_core.embedded_wasm_available(), "embedded rustpython.wasm is missing"


def test_tool_executor_validate_and_execute_uses_wasm_when_ci_requires_wasm() -> None:
    if os.environ.get("SAGE_CI_REQUIRE_WASM") != "1":
        pytest.skip("CI wasm contract not requested")

    from sage_core import ToolExecutor

    result = ToolExecutor().validate_and_execute(
        'print("hello from embedded wasm")',
        "{}",
    )

    assert result.exit_code == 0, result.stderr
    assert "hello from embedded wasm" in result.stdout
