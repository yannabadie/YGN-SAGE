"""Built-in tool capability manifest + resolver.

Phase 1.5c cycle-13 K (cgpro VERIFY 2026-05-06 EDIT_REQUIRED): the
built-in manifest is DOCUMENTATION / AUDIT ONLY since Phase 1.5c.
Python documents `function.__module__` and `function.__qualname__` as
WRITABLE attributes of user-defined functions, so any handler-metadata-
based trust anchor is forgeable: an attacker can mutate
`fake._handler.__module__` and `fake._handler.__qualname__` to mimic a
trusted factory's signature exactly. Phase 1.5c resolves this by
removing the manifest from the runtime path entirely.

The runtime resolver `resolve_tool_capability` chain (Phase 1.5c):
  1. Explicit `tool.capability` set by constructor
     (`Tool(capability=...)`) or `@Tool.define(capability=...)` —
     this is REQUIRED for every built-in factory at the construction
     site (see `sage.tools.{typed_repo, exocortex_tools, context7_tools,
     memory_tools, agent_mgmt, builtin, sage_recurse, meta}` post-1.5c).
  2. Class-level default in `_CLASS_CAPABILITY_DEFAULTS` — covers
     dynamic-class cases (e.g. `AgentTool` defaults to `dangerous`).
  3. None of the above → raise `ToolPolicyDeclarationError`.

This module is INTERNAL — do not import from outside `sage.policy`.

The `_BUILTIN_TOOL_CAPABILITIES` dict below is the authoritative
documentation of what capability each built-in tool name carries; the
audit CLI (`sage.ops.toolpolicy_audit`) reads it for inventory. To
change a built-in's capability you MUST update both:
  - the `capability=...` argument at the factory call site, AND
  - the corresponding entry in `_BUILTIN_TOOL_CAPABILITIES`,
so the audit CLI and the runtime resolver stay aligned.
"""
from __future__ import annotations

from typing import Any

from sage.policy.errors import ToolPolicyDeclarationError
from sage.policy.tool_policy import ToolCapability


# ---------------------------------------------------------------------------
# Built-in manifest — DOCUMENTATION / AUDIT ONLY (Phase 1.5c).
# ---------------------------------------------------------------------------
#
# Phase 1.5c (cgpro VERIFY 2026-05-06 EDIT_REQUIRED): the manifest is
# NO LONGER consulted by `resolve_tool_capability` at runtime. Python's
# `function.__module__` and `function.__qualname__` are writable
# attributes — they cannot serve as a trust anchor for accepting an
# unlabeled tool, because a hostile caller can set
# `fake._handler.__module__ = "sage.tools.typed_repo"` and bypass any
# (name, module) match.
#
# The manifest is kept here for two purposes:
#   (1) Audit (`sage.ops.toolpolicy_audit`) reads it to enumerate the
#       built-in inventory and report each tool's expected capability.
#   (2) Documentation: a single place where a reader can find out
#       "what capability does the canonical `read_file` factory carry?".
#
# To CHANGE a built-in tool's capability, you MUST update both:
#   - The `capability=...` argument at the factory's call site
#     (e.g. `@Tool.define(name="read_file", capability=...)` in
#     `sage.tools.typed_repo._build_read_file_tool`).
#   - The corresponding entry in this manifest (so the audit CLI
#     reports the same value).
#
# Map shape: `(spec.name, expected_handler_module_prefix) -> ToolCapability`.

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

    # Phase 1.5c (cgpro VERIFY 2026-05-06 EDIT_REQUIRED): the resolver
    # runtime path NO LONGER consults the built-in manifest. Python
    # documents `function.__module__` and `function.__qualname__` as
    # writable attributes, so any handler-metadata-based trust anchor
    # is forgeable: an attacker can set `fake._handler.__module__ =
    # "sage.tools.typed_repo"` and bypass any (name, module) match.
    # The manifest is kept as documentation / audit reference (Tool
    # of name X is *expected* to live in module Y) but is not
    # consulted to resolve capability for an unlabeled tool.
    #
    # Resolution order (Phase 1.5c):
    #   1. Explicit `tool.capability` — required for built-ins, set
    #      at the factory's `Tool.define(capability=...)` /
    #      `Tool(spec=..., capability=...)` call site.
    #   2. Class-level default (currently: AgentTool -> dangerous).
    #   3. Raise `ToolPolicyDeclarationError`.

    class_name = type(tool).__name__
    if class_name in _CLASS_CAPABILITY_DEFAULTS:
        return _CLASS_CAPABILITY_DEFAULTS[class_name]

    raise ToolPolicyDeclarationError(
        f"Tool {_describe_tool(tool)} has no resolvable capability declaration. "
        f"Pass `capability=` at construction (e.g. `Tool(spec=..., capability="
        f"\"pure\")` or `@Tool.define(name=..., capability=ToolCapability.X)`), "
        f"or set a class default in `_CLASS_CAPABILITY_DEFAULTS`. "
        f"There is intentionally no default-tag-dangerous fallback AND no "
        f"runtime manifest lookup (per cgpro 2026-05-06 EDIT_REQUIRED Phase "
        f"1.5c: handler metadata `__module__` / `__qualname__` are writable "
        f"in Python and cannot serve as a trust anchor). The built-in "
        f"manifest at `_BUILTIN_TOOL_CAPABILITIES` is documentation only — "
        f"it is consulted by `sage.ops.toolpolicy_audit` for inventory but "
        f"NOT by `Registry.register` to blanche unlabeled tools."
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
