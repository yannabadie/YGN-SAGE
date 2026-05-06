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
# Built-in manifest — keyed by (spec.name, expected handler module prefix).
# ---------------------------------------------------------------------------
#
# Phase 1.5 ship-time inventory. New tools added to the codebase MUST
# either pass `capability=...` at construction or be added here.
#
# Phase 1.5b (cgpro VERIFY 2026-05-06 EDIT_REQUIRED): the manifest is
# keyed BOTH by `spec.name` AND by the trusted `handler.__module__`
# prefix the handler MUST originate from. Without the module check, an
# external tool could spoof a built-in name (e.g. construct a
# `Tool(name="read_file", handler=untrusted_callable)`) and inherit
# the built-in's READ_LOCAL capability. The module check anchors the
# manifest hit to the trusted tool factory's module surface — an
# external impostor whose handler lives in `__main__` or some
# unrelated module fails resolution and is rejected at registration.
#
# Map shape: `(spec.name, expected_handler_module_prefix) -> ToolCapability`.
# A handler at `<expected_prefix>` OR `<expected_prefix>.<sub>` matches.
# A handler from an unrelated module does NOT match — even if the
# `spec.name` is identical.

_BUILTIN_TOOL_CAPABILITIES: dict[tuple[str, str], ToolCapability] = {
    # File-system reads (typed_repo.py).
    ("read_file", "sage.tools.typed_repo"): ToolCapability.READ_LOCAL,
    ("list_files", "sage.tools.typed_repo"): ToolCapability.READ_LOCAL,
    ("search_repo", "sage.tools.typed_repo"): ToolCapability.READ_LOCAL,
    ("git_diff", "sage.tools.typed_repo"): ToolCapability.READ_LOCAL,
    # File-system writes (typed_repo.py).
    ("apply_patch", "sage.tools.typed_repo"): ToolCapability.WRITE_LOCAL,
    # Subprocess (typed_repo.py: pytest invocation under sandbox).
    ("run_tests", "sage.tools.typed_repo"): ToolCapability.SUBPROCESS,
    # Network / remote (exocortex_tools.py + context7_tools.py).
    ("search_exocortex", "sage.tools.exocortex_tools"): ToolCapability.NETWORK,
    ("refresh_knowledge", "sage.tools.exocortex_tools"): ToolCapability.NETWORK,
    ("lookup_library_docs", "sage.tools.context7_tools"): ToolCapability.NETWORK,
    ("context7_query", "sage.tools.context7_tools"): ToolCapability.NETWORK,
    # Memory tools (memory_tools.py) — read/write within local SQLite
    # tier. Treated as read_local for retrieval, write_local for store/
    # update/delete.
    ("search_memory", "sage.tools.memory_tools"): ToolCapability.READ_LOCAL,
    ("retrieve_context", "sage.tools.memory_tools"): ToolCapability.READ_LOCAL,
    ("summarize_context", "sage.tools.memory_tools"): ToolCapability.READ_LOCAL,
    ("filter_context", "sage.tools.memory_tools"): ToolCapability.READ_LOCAL,
    ("search_causal_chain", "sage.tools.memory_tools"): ToolCapability.READ_LOCAL,
    ("store_memory", "sage.tools.memory_tools"): ToolCapability.WRITE_LOCAL,
    ("update_memory", "sage.tools.memory_tools"): ToolCapability.WRITE_LOCAL,
    ("delete_memory", "sage.tools.memory_tools"): ToolCapability.WRITE_LOCAL,
    # Agent management (agent_mgmt.py): creating / calling sub-agents
    # delegates to arbitrary `agent.run(...)` — class-default-dangerous
    # for AgentTool concretes; the create/call wrappers themselves are
    # also dangerous because they enable that delegation.
    ("create_agent", "sage.tools.agent_mgmt"): ToolCapability.DANGEROUS,
    ("call_agent", "sage.tools.agent_mgmt"): ToolCapability.DANGEROUS,
    ("list_active_agents", "sage.tools.agent_mgmt"): ToolCapability.READ_LOCAL,
    # ToolForge / dynamic tool synthesis.
    ("create_python_tool", "sage.tools.forge"): ToolCapability.DANGEROUS,
    ("create_bash_tool", "sage.tools.forge"): ToolCapability.DANGEROUS,
    # Dangerous: raw shell, agent recursion.
    ("bash", "sage.tools.builtin"): ToolCapability.DANGEROUS,
    ("execute_bash", "sage.tools.builtin"): ToolCapability.DANGEROUS,
    ("sage_recurse", "sage.tools.sage_recurse"): ToolCapability.DANGEROUS,
}


def _module_matches(handler_module: str, expected_prefix: str) -> bool:
    """Trusted-module match: exact, or `expected_prefix` + dot-submodule.

    Examples:
        _module_matches("sage.tools.typed_repo", "sage.tools.typed_repo")  # True
        _module_matches("sage.tools.typed_repo.helpers", "sage.tools.typed_repo")  # True
        _module_matches("__main__", "sage.tools.typed_repo")  # False  (spoof)
        _module_matches("sage.tools.typed_repo_evil", "sage.tools.typed_repo")  # False
    """
    if not handler_module:
        return False
    if handler_module == expected_prefix:
        return True
    return handler_module.startswith(expected_prefix + ".")


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
    if spec_name:
        # Phase 1.5b anti-spoof: the manifest hit requires BOTH the
        # spec.name AND the handler's `__module__` to match a trusted
        # entry. A tool whose spec.name happens to match `read_file`
        # but whose handler lives in `__main__` (or any other module)
        # is NOT a manifest hit — it falls through to class default
        # then raises ToolPolicyDeclarationError if unresolved.
        handler = getattr(tool, "_handler", None)
        handler_module = getattr(handler, "__module__", "") if handler else ""
        for (entry_name, expected_prefix), capability in _BUILTIN_TOOL_CAPABILITIES.items():
            if entry_name != spec_name:
                continue
            if _module_matches(handler_module, expected_prefix):
                return capability
            # Name match but module mismatch — possible spoof attempt.
            # Don't return early; let the class-default / raise paths
            # handle it. A future iteration could log a warning here.

    class_name = type(tool).__name__
    if class_name in _CLASS_CAPABILITY_DEFAULTS:
        return _CLASS_CAPABILITY_DEFAULTS[class_name]

    raise ToolPolicyDeclarationError(
        f"Tool {_describe_tool(tool)} has no resolvable capability declaration. "
        f"Pass `capability=` at construction (e.g. `Tool(spec=..., capability="
        f"\"pure\")`), add a built-in manifest entry, or set a class default. "
        f"There is intentionally no default-tag-dangerous fallback (per cgpro "
        f"DESIGN: that creates an illusion of security). NB: the manifest "
        f"now requires BOTH spec.name AND the handler's __module__ to match "
        f"a trusted entry (Phase 1.5b anti-spoof, cgpro 2026-05-06)."
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
