"""ToolPolicy exception hierarchy.

Per cgpro DESIGN_LOCKED Q4: registration-time failure is hard, runtime
failure is convertible to a `ToolResult(is_error=True)` so the agent can
reason about the denial as a tool-call result rather than crashing.
"""
from __future__ import annotations


class ToolPolicyError(Exception):
    """Base class for ToolPolicy failures.

    Public — agents and operator code may catch this to handle either
    declaration or runtime denial in one place.
    """


class ToolPolicyDeclarationError(ToolPolicyError):
    """A tool was registered without a resolvable capability declaration.

    Raised from `Registry.register(tool)` when the resolver finds no
    explicit `capability=` argument, no built-in manifest entry, and no
    class-level default. cgpro DESIGN trap: NO default-tag-dangerous
    fallback is provided — that creates an illusion of security.
    """


class ToolPolicyDenied(ToolPolicyError):
    """Runtime denial: a tool was invoked with a capability not in the
    effective grant set.

    `Tool.execute(...)` catches this internally and returns
    `ToolResult(is_error=True, output="Error: ToolPolicyDenied: ...")`
    so the agent observes a failed tool call (and the OracleStack records
    the run as `trainable=False`). Operator code that bypasses
    `Tool.execute` (e.g. introspection paths) sees the bare exception.
    """
