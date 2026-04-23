"""Regression test for the A0c fix (2026-04-23, ALIRE2 §6).

Tool.execute() must NOT leak Python tracebacks into the model-visible
ToolResult.output string. Prior behaviour appended ``traceback.format_exc()``
to the output, which reached the agent loop + LLM prompt and leaked file
paths / module names / host directory structure. The fix keeps the
exception type + message (useful for agent steering) and routes the full
traceback to the operator log via ``log.exception``.
"""
from __future__ import annotations

import pytest

from sage.llm.base import ToolDef
from sage.tools.base import Tool, ToolResult


@pytest.mark.asyncio
async def test_tool_execute_exception_omits_traceback() -> None:
    """Model-visible output must not contain raw traceback markers.

    A raised exception inside the handler should produce a ToolResult
    whose ``output`` is exactly ``Error: <type>: <message>`` — no
    ``Traceback (most recent call last):``, no ``File "..."``, no
    stack-frame lines.
    """

    async def _raises(**_kwargs: object) -> str:
        raise ValueError("something specific went wrong")

    tool = Tool(
        spec=ToolDef(name="demo", description="d", parameters={}),
        handler=_raises,
    )

    result = await tool.execute({})

    assert isinstance(result, ToolResult)
    assert result.is_error is True
    assert result.output == "Error: ValueError: something specific went wrong"
    # Explicit guards against the old behaviour (post-fix must stay clean).
    assert "Traceback" not in result.output
    assert 'File "' not in result.output
    assert "line " not in result.output
    # Exception type survives — agents often steer on it.
    assert "ValueError" in result.output


@pytest.mark.asyncio
async def test_tool_execute_exception_logs_full_traceback(caplog) -> None:
    """Operators still see the full traceback in logs — just not the model.

    We assert the log record contains the exception info (pytest's caplog
    captures exc_info via log.exception). The exact traceback text is
    environment-dependent, so we just assert the exception type and
    message appear in the log record.
    """
    import logging

    async def _raises(**_kwargs: object) -> str:
        raise RuntimeError("operator-visible detail")

    tool = Tool(
        spec=ToolDef(name="demo", description="d", parameters={}),
        handler=_raises,
    )

    with caplog.at_level(logging.ERROR, logger="sage.tools.base"):
        result = await tool.execute({})

    # Model-visible side
    assert result.output == "Error: RuntimeError: operator-visible detail"
    # Operator-visible side: at least one log record with exc_info.
    tool_records = [r for r in caplog.records if r.name == "sage.tools.base"]
    assert tool_records, "expected at least one log record from sage.tools.base"
    assert any(r.exc_info is not None for r in tool_records), (
        "exception info should be attached to the log record"
    )


@pytest.mark.asyncio
async def test_tool_execute_success_still_returns_output() -> None:
    """Baseline: successful execution is unchanged by the redaction fix."""

    async def _ok(x: int) -> str:
        return f"got {x}"

    tool = Tool(
        spec=ToolDef(name="demo", description="d", parameters={}),
        handler=_ok,
    )
    result = await tool.execute({"x": 42})
    assert result.is_error is False
    assert result.output == "got 42"
