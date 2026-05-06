"""Built-in tool capability manifest + resolver.

Phase 1.5 cycle-13 K (cgpro DESIGN_LOCKED 2026-05-06): per the migration
strategy, built-in tools that predate the `capability=` parameter are
declared in this internal manifest. External / new tools that don't pass
an explicit `capability=` MUST fail registration with
`ToolPolicyDeclarationError` — no default-tag-dangerous fallback (cgpro
trap: that creates an illusion of security).

This module is INTERNAL — do not import from outside `sage.policy`.

Resolution order (per `resolve_tool_capability`):
  1. Explicit `tool.capability` set by constructor (`Tool(capability=...)`)
     or `Tool.define(capability=...)` — wins.
  2. Spec-name lookup in `_BUILTIN_TOOL_CAPABILITIES` — covers the 13
     built-in modules at Phase 1.5 ship time.
  3. Class-level default in `_CLASS_CAPABILITY_DEFAULTS` — covers
     dynamic-class cases (e.g. `AgentTool` defaults to `dangerous`).
  4. None of the above → raise `ToolPolicyDeclarationError`.
"""
from __future__ import annotations

from typing import Any

from sage.policy.errors import ToolPolicyDeclarationError
from sage.policy.tool_policy import ToolCapability


# ---------------------------------------------------------------------------
# Built-in manifest — by spec.name (the LLM-facing tool name).
# ---------------------------------------------------------------------------
#
# Phase 1.5 ship-time inventory. New tools added to the codebase MUST
# either pass `capability=...` at construction or be added here. cgpro
# DESIGN trap: AgentTool defaults to `dangerous` because it delegates
# to arbitrary `agent.run(...)` — its concrete instances inherit the
# class default unless they pass an explicit override.

_BUILTIN_TOOL_CAPABILITIES: dict[str, ToolCapability] = {
    # File-system reads (typed_repo.py + memory_tools.py).
    "read_file": ToolCapability.READ_LOCAL,
    "list_files": ToolCapability.READ_LOCAL,
    "search_files": ToolCapability.READ_LOCAL,
    "search_repo": ToolCapability.READ_LOCAL,
    "git_diff": ToolCapability.READ_LOCAL,
    # File-system writes (typed_repo.py).
    "write_file": ToolCapability.WRITE_LOCAL,
    "edit_file": ToolCapability.WRITE_LOCAL,
    "create_file": ToolCapability.WRITE_LOCAL,
    "delete_file": ToolCapability.WRITE_LOCAL,
    "apply_patch": ToolCapability.WRITE_LOCAL,
    # Subprocess (typed_repo.py: pytest invocation under sandbox).
    "run_tests": ToolCapability.SUBPROCESS,
    # Network / remote (exocortex_tools.py + context7_tools.py).
    "search_exocortex": ToolCapability.NETWORK,
    "refresh_knowledge": ToolCapability.NETWORK,
    "lookup_library_docs": ToolCapability.NETWORK,
    "context7_query": ToolCapability.NETWORK,
    "web_search": ToolCapability.NETWORK,
    "web_fetch": ToolCapability.NETWORK,
    # Memory tools (memory_tools.py) — read/write within local SQLite
    # tier. Treated as read_local for retrieval, write_local for store/
    # update/delete.
    "search_memory": ToolCapability.READ_LOCAL,
    "retrieve_context": ToolCapability.READ_LOCAL,
    "summarize_context": ToolCapability.READ_LOCAL,
    "filter_context": ToolCapability.READ_LOCAL,
    "search_causal_chain": ToolCapability.READ_LOCAL,
    "store_memory": ToolCapability.WRITE_LOCAL,
    "update_memory": ToolCapability.WRITE_LOCAL,
    "delete_memory": ToolCapability.WRITE_LOCAL,
    # Subprocess / sandboxed exec.
    "execute_python": ToolCapability.SUBPROCESS,
    "execute_code": ToolCapability.SUBPROCESS,
    # Agent management (agent_mgmt.py): creating / calling sub-agents
    # delegates to arbitrary `agent.run(...)` — class-default-dangerous
    # for AgentTool concretes; the create/call wrappers themselves are
    # also dangerous because they enable that delegation.
    "create_agent": ToolCapability.DANGEROUS,
    "call_agent": ToolCapability.DANGEROUS,
    "list_active_agents": ToolCapability.READ_LOCAL,
    # ToolForge / dynamic tool synthesis.
    "create_python_tool": ToolCapability.DANGEROUS,
    "create_bash_tool": ToolCapability.DANGEROUS,
    # Dangerous: raw shell, agent recursion, raw exec.
    "bash": ToolCapability.DANGEROUS,
    "execute_bash": ToolCapability.DANGEROUS,
    "execute_raw": ToolCapability.DANGEROUS,
    "sage_recurse": ToolCapability.DANGEROUS,
}


# ---------------------------------------------------------------------------
# Class-level defaults — by class name.
# ---------------------------------------------------------------------------
#
# Used when a tool's `spec.name` doesn't match the manifest but the
# concrete class has a known default. cgpro DESIGN: AgentTool maps to
# `dangerous` because it forwards to `agent.run(...)` which can read,
# write, and call network. Tool subclasses without an explicit override
# inherit this.

_CLASS_CAPABILITY_DEFAULTS: dict[str, ToolCapability] = {
    "AgentTool": ToolCapability.DANGEROUS,
}


def resolve_tool_capability(tool: Any) -> ToolCapability:
    """Resolve a `Tool`-like instance to its declared capability.

    Resolution order:
      1. `tool.capability` if set and not None.
      2. Built-in manifest lookup by `tool.spec.name`.
      3. Class-level default by `type(tool).__name__`.
      4. Raise `ToolPolicyDeclarationError`.

    The function is duck-typed (operates on any object exposing the
    expected attributes) so it can be used both by `Registry.register`
    (which has access to a real `Tool`) and by audit / introspection
    code paths that might pass a record-shaped object.
    """
    explicit = getattr(tool, "capability", None)
    if explicit is not None:
        if isinstance(explicit, ToolCapability):
            return explicit
        if isinstance(explicit, str):
            try:
                return ToolCapability(explicit)
            except ValueError as exc:
                raise ToolPolicyDeclarationError(
                    f"Tool {_describe_tool(tool)} has invalid `capability` value "
                    f"{explicit!r}: {exc}"
                ) from exc

    spec_name = _safe_spec_name(tool)
    if spec_name and spec_name in _BUILTIN_TOOL_CAPABILITIES:
        return _BUILTIN_TOOL_CAPABILITIES[spec_name]

    class_name = type(tool).__name__
    if class_name in _CLASS_CAPABILITY_DEFAULTS:
        return _CLASS_CAPABILITY_DEFAULTS[class_name]

    raise ToolPolicyDeclarationError(
        f"Tool {_describe_tool(tool)} has no resolvable capability declaration. "
        f"Pass `capability=` at construction (e.g. `Tool(spec=..., capability="
        f"\"pure\")`), add a built-in manifest entry, or set a class default. "
        f"There is intentionally no default-tag-dangerous fallback (per cgpro "
        f"DESIGN: that creates an illusion of security)."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_spec_name(tool: Any) -> str | None:
    """Return `tool.spec.name` if present, else None. Never raises."""
    spec = getattr(tool, "spec", None)
    if spec is None:
        return None
    name = getattr(spec, "name", None)
    if isinstance(name, str) and name:
        return name
    return None


def _describe_tool(tool: Any) -> str:
    """Best-effort human-readable identifier for an unresolved tool."""
    spec_name = _safe_spec_name(tool)
    class_name = type(tool).__name__
    if spec_name:
        return f"{spec_name!r} (class {class_name})"
    return f"<{class_name} instance>"
