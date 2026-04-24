"""AUDIT3 #17 / A14 — opt-in Pydantic validation on ToolResult.output."""
from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from sage.llm.base import ToolDef
from sage.tools.base import Tool, ToolResult


class FileReadResult(BaseModel):
    path: str
    size: int
    content: str


class Outer(BaseModel):
    name: str
    inner: FileReadResult


async def _noop_handler(**_: object) -> str:
    return ""


def _make_tool(output_schema: type[BaseModel] | None = None) -> Tool:
    spec = ToolDef(name="t", description="", parameters={})
    return Tool(spec=spec, handler=_noop_handler, output_schema=output_schema)


def test_tool_without_schema_leaves_validated_none():
    tool = _make_tool()
    assert tool.output_schema is None

    result = ToolResult(output="arbitrary text")
    assert result.validated is None


def test_valid_json_matching_schema_returns_instance():
    result = ToolResult(output='{"path":"x.txt","size":42,"content":"hi"}')
    parsed = result.validate_output(FileReadResult)

    assert isinstance(parsed, FileReadResult)
    assert parsed.path == "x.txt"
    assert parsed.size == 42
    assert result.validated is parsed


def test_invalid_json_returns_none_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SAGE_TOOLRESULT_VALIDATE", raising=False)
    result = ToolResult(output="not-json")

    assert result.validate_output(FileReadResult) is None
    assert result.validated is None


def test_schema_violation_raises_validation_error():
    # Valid JSON, wrong schema (missing fields).
    result = ToolResult(output='{"path":"x.txt"}')

    with pytest.raises(ValidationError):
        result.validate_output(FileReadResult)


def test_strict_env_promotes_json_decode_to_error(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SAGE_TOOLRESULT_VALIDATE", "1")
    result = ToolResult(output="definitely-not-json")

    with pytest.raises(ValueError, match="not valid JSON"):
        result.validate_output(FileReadResult)


def test_nested_schemas_work():
    payload = '{"name":"outer","inner":{"path":"p","size":1,"content":"c"}}'
    result = ToolResult(output=payload)
    parsed = result.validate_output(Outer)

    assert isinstance(parsed, Outer)
    assert isinstance(parsed.inner, FileReadResult)
    assert parsed.inner.path == "p"


def test_tool_accepts_output_schema_kwarg():
    tool = _make_tool(output_schema=FileReadResult)
    assert tool.output_schema is FileReadResult
