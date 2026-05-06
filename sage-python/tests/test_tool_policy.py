"""ToolPolicy capability gate tests — Phase 1.5 (cycle-13 K).

Per cgpro DESIGN_LOCKED 2026-05-06 Commit 4: 15 tests T1-T15 covering
every ToolCapability tier × grant scenario, registration-time rejection
of unresolved tools, runtime gate via Tool.execute, bypass-factory
ContextVar inheritance, and the audit CLI.

The autouse `_grant_all_tool_capabilities_in_tests` fixture in
`conftest.py` grants ALL tiers and softens the resolver for the rest
of the test suite. This file exercises the STRICT contract — tests
reset the policy to `ToolPolicy.default()` (or a more restrictive
scope) and re-import the production resolver where needed.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from sage.llm.base import ToolDef
from sage.policy import (
    ToolCapability,
    ToolPolicy,
    ToolPolicyDeclarationError,
    set_current_tool_policy,
)
from sage.policy.tool_policy import _CURRENT_POLICY
from sage.tools.base import Tool
from sage.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _noop_handler(**_kwargs: Any) -> str:
    return "ok"


def _make_tool(name: str, capability: ToolCapability | str | None = None) -> Tool:
    """Construct a Tool with explicit capability for the test scope."""
    return Tool(
        spec=ToolDef(name=name, description="test", parameters={}),
        handler=_noop_handler,
        capability=capability,
    )


@pytest.fixture
def scoped_policy():
    """Yield a callable that sets a scoped ToolPolicy and restores on exit.

    Bypasses the autouse permissive fixture by directly manipulating
    `_CURRENT_POLICY` for the duration of the test.
    """
    tokens: list[Any] = []

    def _set(policy: ToolPolicy) -> ToolPolicy:
        tokens.append(_CURRENT_POLICY.set(policy))
        return policy

    yield _set

    for token in reversed(tokens):
        try:
            _CURRENT_POLICY.reset(token)
        except (ValueError, LookupError):
            pass


# ---------------------------------------------------------------------------
# T1-T11 — capability tier × grant scenarios.
# ---------------------------------------------------------------------------


def test_t1_pure_allowed_without_explicit_grant(scoped_policy):
    """Default policy `{pure}` allows pure tool execution."""
    scoped_policy(ToolPolicy.default())
    tool = _make_tool("t1_pure", capability=ToolCapability.PURE)
    result = asyncio.run(tool.execute({}))
    assert result.is_error is False
    assert result.output == "ok"


def test_t2_read_local_denied_by_default(scoped_policy):
    """Default policy `{pure}` denies a read_local tool at runtime."""
    scoped_policy(ToolPolicy.default())
    tool = _make_tool("t2_read", capability=ToolCapability.READ_LOCAL)
    result = asyncio.run(tool.execute({}))
    assert result.is_error is True
    assert "ToolPolicyDenied" in result.output
    assert "read_local" in result.output


def test_t3_read_local_allowed_with_grant(scoped_policy):
    """Granting read_local lets the read_local tool through."""
    scoped_policy(ToolPolicy.default().grant("read_local"))
    tool = _make_tool("t3_read", capability=ToolCapability.READ_LOCAL)
    result = asyncio.run(tool.execute({}))
    assert result.is_error is False


def test_t4_write_local_denied_by_default(scoped_policy):
    scoped_policy(ToolPolicy.default())
    tool = _make_tool("t4_write", capability=ToolCapability.WRITE_LOCAL)
    result = asyncio.run(tool.execute({}))
    assert result.is_error is True
    assert "write_local" in result.output


def test_t5_write_local_allowed_with_grant(scoped_policy):
    scoped_policy(ToolPolicy.default().grant("write_local"))
    tool = _make_tool("t5_write", capability=ToolCapability.WRITE_LOCAL)
    result = asyncio.run(tool.execute({}))
    assert result.is_error is False


def test_t6_network_denied_by_default(scoped_policy):
    scoped_policy(ToolPolicy.default())
    tool = _make_tool("t6_net", capability=ToolCapability.NETWORK)
    result = asyncio.run(tool.execute({}))
    assert result.is_error is True
    assert "network" in result.output


def test_t7_network_allowed_with_env_grant(scoped_policy, monkeypatch):
    """SAGE_TOOL_GRANTS=network grants network only — not other tiers."""
    monkeypatch.setenv("SAGE_TOOL_GRANTS", "network")
    scoped_policy(ToolPolicy.from_environment())

    net_tool = _make_tool("t7_net", capability=ToolCapability.NETWORK)
    result = asyncio.run(net_tool.execute({}))
    assert result.is_error is False, "network grant should allow network tool"

    # write_local NOT granted — denied.
    write_tool = _make_tool("t7_write", capability=ToolCapability.WRITE_LOCAL)
    result_write = asyncio.run(write_tool.execute({}))
    assert result_write.is_error is True, "network grant must NOT also grant write_local"


def test_t8_subprocess_denied_by_default(scoped_policy):
    scoped_policy(ToolPolicy.default())
    tool = _make_tool("t8_sub", capability=ToolCapability.SUBPROCESS)
    result = asyncio.run(tool.execute({}))
    assert result.is_error is True


def test_t9_subprocess_allowed_with_grant(scoped_policy):
    scoped_policy(ToolPolicy.default().grant("subprocess"))
    tool = _make_tool("t9_sub", capability=ToolCapability.SUBPROCESS)
    result = asyncio.run(tool.execute({}))
    assert result.is_error is False


def test_t10_dangerous_denied_without_exact_dangerous_grant(scoped_policy):
    """No hierarchical implicit closure — granting subprocess + network
    + read_local + write_local does NOT imply dangerous."""
    scoped_policy(
        ToolPolicy.default()
        .grant("subprocess")
        .grant("network")
        .grant("read_local")
        .grant("write_local")
    )
    tool = _make_tool("t10_dangerous", capability=ToolCapability.DANGEROUS)
    result = asyncio.run(tool.execute({}))
    assert result.is_error is True
    assert "dangerous" in result.output


def test_t11_dangerous_allowed_with_exact_grant(scoped_policy):
    scoped_policy(ToolPolicy.default().grant("dangerous"))
    tool = _make_tool("t11_dangerous", capability=ToolCapability.DANGEROUS)
    result = asyncio.run(tool.execute({}))
    assert result.is_error is False


# ---------------------------------------------------------------------------
# T12 — Registry resolution: manifest hit + unknown rejection.
# ---------------------------------------------------------------------------


def test_t12_registry_resolution_allows_manifest_and_rejects_unknown_unlabeled(
    monkeypatch,
):
    """The strict resolver accepts manifest entries, rejects unknown unlabeled.

    The autouse fixture monkey-patches `resolve_tool_capability` to a
    permissive variant. This test reverts that patch back to the strict
    semantics for the duration of this test only.
    """
    import sage.policy.manifest as _manifest_mod
    from sage.policy.manifest import _BUILTIN_TOOL_CAPABILITIES, _CLASS_CAPABILITY_DEFAULTS

    def _strict_resolve(tool: Any) -> ToolCapability:
        explicit = getattr(tool, "capability", None)
        if explicit is not None:
            if isinstance(explicit, ToolCapability):
                return explicit
            if isinstance(explicit, str):
                return ToolCapability(explicit)
        spec = getattr(tool, "spec", None)
        spec_name = getattr(spec, "name", None) if spec else None
        if isinstance(spec_name, str) and spec_name in _BUILTIN_TOOL_CAPABILITIES:
            return _BUILTIN_TOOL_CAPABILITIES[spec_name]
        cls_name = type(tool).__name__
        if cls_name in _CLASS_CAPABILITY_DEFAULTS:
            return _CLASS_CAPABILITY_DEFAULTS[cls_name]
        raise ToolPolicyDeclarationError(f"unresolved: {spec_name or cls_name}")

    monkeypatch.setattr(_manifest_mod, "resolve_tool_capability", _strict_resolve)

    reg = ToolRegistry()

    # Manifest hit: read_file.
    known = _make_tool("read_file")
    reg.register(known)
    assert known.capability == ToolCapability.READ_LOCAL
    assert reg.get("read_file") is known

    # Unknown unlabeled tool: rejected.
    unknown = _make_tool("unknown_xyz_t12")
    with pytest.raises(ToolPolicyDeclarationError):
        reg.register(unknown)
    assert reg.get("unknown_xyz_t12") is None


# ---------------------------------------------------------------------------
# T13 — Tool.execute returns is_error on policy denial (no handler call).
# ---------------------------------------------------------------------------


def test_t13_tool_execute_returns_error_result_on_policy_denial(scoped_policy):
    """When the policy denies, the handler is NOT called.

    A handler that would always raise can be safely registered and
    invoked under a denying policy: the gate intercepts before the
    handler runs, so no exception escapes.
    """
    scoped_policy(ToolPolicy.default())  # only pure granted

    handler_called = False

    async def _trap(**_kwargs: Any) -> str:
        nonlocal handler_called
        handler_called = True
        raise RuntimeError("handler should not have run")

    tool = Tool(
        spec=ToolDef(name="t13_trap", description="", parameters={}),
        handler=_trap,
        capability=ToolCapability.NETWORK,
    )
    result = asyncio.run(tool.execute({}))
    assert result.is_error is True
    assert "ToolPolicyDenied" in result.output
    assert handler_called is False, "Handler must NOT be invoked when policy denies"


# ---------------------------------------------------------------------------
# T14 — AgentLoop and bypass factory share effective policy via ContextVar.
# ---------------------------------------------------------------------------


def test_t14_bypass_factory_inherits_effective_policy_via_contextvar(scoped_policy):
    """ContextVar propagation: an asyncio task spawned in the same
    context inherits the parent's policy.

    This is the production contract: `pipeline_v2/execute.py
    create_bypass_agent_loop` spins up a per-run agent loop via
    `asyncio.create_task` (or equivalent), and the new task copies
    the parent context per PEP 567. So the bypass loop's
    Tool.execute calls hit the same `_CURRENT_POLICY` value.
    """
    scoped_policy(ToolPolicy.default())  # only pure

    async def _parent() -> tuple[bool, bool]:
        # Direct execution in parent context.
        net_tool = _make_tool("t14_net", capability=ToolCapability.NETWORK)
        parent_result = await net_tool.execute({})

        # Spawn a child task — should inherit the parent's policy.
        async def _child() -> bool:
            child_tool = _make_tool("t14_net_child", capability=ToolCapability.NETWORK)
            child_result = await child_tool.execute({})
            return child_result.is_error

        child_denied = await asyncio.create_task(_child())
        return parent_result.is_error, child_denied

    parent_denied, child_denied = asyncio.run(_parent())
    assert parent_denied is True, "Parent under default {pure} should deny network"
    assert child_denied is True, "Child task must inherit denial via ContextVar"


# ---------------------------------------------------------------------------
# T15 — toolpolicy_audit CLI lists capabilities, sources, allow/deny.
# ---------------------------------------------------------------------------


def test_t15_toolpolicy_audit_cli_lists_capabilities_and_effective_grants(
    scoped_policy, capsys
):
    """The audit CLI emits a JSON payload with effective grants, total
    tools, by_capability counts, and per-tool allow/deny status."""
    scoped_policy(ToolPolicy.default().grant("read_local"))

    from sage.ops import toolpolicy_audit

    rc = toolpolicy_audit.main(["--json"])
    captured = capsys.readouterr()
    assert rc == 0

    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert sorted(payload["effective_grants"]) == ["pure", "read_local"]
    assert payload["total_tools"] > 0
    assert "by_capability" in payload

    # At least one entry is read_file (manifest entry, READ_LOCAL).
    read_file = next(
        (e for e in payload["entries"] if e["name"] == "read_file"),
        None,
    )
    assert read_file is not None, "Expected 'read_file' in audit entries"
    assert read_file["capability"] == "read_local"
    assert read_file["source"] == "manifest"
    assert read_file["allowed"] is True

    # A dangerous-tier entry is denied under {pure, read_local}.
    bash_entry = next(
        (e for e in payload["entries"] if e["name"] == "bash"),
        None,
    )
    assert bash_entry is not None
    assert bash_entry["capability"] == "dangerous"
    assert bash_entry["allowed"] is False
