from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sage.tools.gap_detector import CreationTicket


_LLM_RESPONSE = """\
```python
import json

def tool_adder(args):
    result = args.get("a", 0) + args.get("b", 0)
    print(json.dumps({"output": result}))
```

```python
tool_adder({"a": 1, "b": 2})
tool_adder({"a": 0, "b": 0})
tool_adder({"a": -1, "b": 1})
```
"""


def _ticket(name: str = "adder") -> CreationTicket:
    return CreationTicket(
        task="add two numbers",
        gap_description=f"Tool '{name}' not found in registry",
        required_interface='{"a": int, "b": int}',
        context="",
        created_at=0,
        tool_name_hint=name,
    )


def _forge(registry: MagicMock, approval_callback=None):
    from sage.tools.forge import ToolForge

    llm = AsyncMock()
    response = MagicMock()
    response.content = _LLM_RESPONSE
    llm.generate.return_value = response

    return ToolForge(
        registry=registry,
        llm_provider=llm,
        llm_config=MagicMock(),
        approval_callback=approval_callback,
    )


@pytest.mark.asyncio
async def test_process_ticket_denied_approval_does_not_mark_source():
    registry = MagicMock()
    forge = _forge(registry, approval_callback=lambda _name, _spec: False)

    with patch("sage.tools.meta.create_python_tool") as mock_create:
        mock_create.run = AsyncMock(return_value="Tool 'tool_adder' created")
        result = await forge.process_ticket(_ticket())

    assert result is None
    mock_create.run.assert_not_called()
    registry.mark_source.assert_not_called()


@pytest.mark.asyncio
async def test_process_ticket_allowed_approval_marks_source_with_approver():
    registry = MagicMock()
    approvals: list[tuple[str, str]] = []

    def allow_approval(ticket_name: str, tool_spec_text: str) -> bool:
        approvals.append((ticket_name, tool_spec_text))
        return True

    forge = _forge(registry, approval_callback=allow_approval)

    with patch("sage.tools.meta.create_python_tool") as mock_create:
        mock_create.run = AsyncMock(return_value="Tool 'tool_adder' created")
        result = await forge.process_ticket(_ticket())

    assert result == "tool_adder"
    assert approvals
    assert approvals[0][0] == "adder"
    assert "def tool_adder" in approvals[0][1]
    registry.mark_source.assert_called_once_with(
        "tool_adder",
        "forged",
        approved_by="allow_approval",
    )


@pytest.mark.asyncio
async def test_require_approval_without_callback_raises(monkeypatch):
    monkeypatch.setenv("SAGE_TOOLFORGE_REQUIRE_APPROVAL", "1")
    registry = MagicMock()
    forge = _forge(registry)

    with pytest.raises(RuntimeError, match="approval_callback"):
        await forge.process_ticket(_ticket())

    registry.mark_source.assert_not_called()


@pytest.mark.asyncio
async def test_approve_all_env_approves_without_callback(monkeypatch, caplog):
    monkeypatch.setenv("SAGE_TOOLFORGE_APPROVE_ALL", "1")

    import sage.tools.forge as forge_module

    monkeypatch.setattr(forge_module, "_APPROVE_ALL_WARNED", False)

    registry = MagicMock()
    forge = _forge(registry)

    with patch("sage.tools.meta.create_python_tool") as mock_create:
        mock_create.run = AsyncMock(return_value="Tool 'tool_adder' created")
        with caplog.at_level(logging.WARNING, logger="sage.tools.forge"):
            result = await forge.process_ticket(_ticket())

    assert result == "tool_adder"
    registry.mark_source.assert_called_once_with(
        "tool_adder",
        "forged",
        approved_by="env:approve_all",
    )
    assert any("SAGE_TOOLFORGE_APPROVE_ALL=1" in r.message for r in caplog.records)
