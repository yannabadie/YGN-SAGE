"""Tool capability policy — Phase 1.5 (cycle-13 K).

Per cgpro DESIGN_LOCKED 2026-05-06 on conv `cgpro_phase15_toolpolicy_20260506`:
this module is the runtime tool-capability policy enforcement layer that
gates tool execution against an effective grant set BEFORE PyPI publication.

Public surface (re-exports):
  - `ToolCapability`   — Enum of 6 tiers: pure / read_local / write_local /
                         network / subprocess / dangerous.
  - `ToolPolicy`       — Immutable effective-grant container with
                         `default()`, `grant()`, `assert_allowed()`.
  - `ToolPolicyError`         — Common base.
  - `ToolPolicyDenied`        — Runtime denial (gate refused execution).
  - `ToolPolicyDeclarationError` — Registration denial (tool has no
                                   resolvable capability).
  - `get_effective_tool_policy` / `set_current_tool_policy` — ContextVar
                                   accessors for ambient policy state.

Private (do not import):
  - `_BUILTIN_TOOL_CAPABILITIES`   in `manifest.py` — the migration map
                                   for built-in tools that predate Phase 1.5.
  - `_CLASS_CAPABILITY_DEFAULTS`   in `manifest.py` — class-level fallbacks
                                   (e.g. AgentTool defaults to `dangerous`).
  - `resolve_tool_capability`      in `manifest.py` — internal resolver
                                   used by `Registry.register` and friends.

Phase 1.5 SCOPE (per cgpro DESIGN):
  - Declarative gate at registration + runtime, NOT behavior verification.
    A `pure` tool can still lie at runtime; declaration is a contract,
    not a proof. Behavior verification is future work (sandbox/WASI/AST
    confinement).
  - Default effective grant is `pure` ONLY. Even `read_local` requires an
    explicit grant — strictness over convenience for the research-preview
    PyPI install path.
  - No hierarchical implicit closure: granting `dangerous` does NOT also
    grant `network`. Each tier is granted exactly.
"""
from __future__ import annotations

from sage.policy.errors import (
    ToolPolicyDeclarationError,
    ToolPolicyDenied,
    ToolPolicyError,
)
from sage.policy.tool_policy import (
    ToolCapability,
    ToolPolicy,
    get_effective_tool_policy,
    set_current_tool_policy,
)

__all__ = [
    "ToolCapability",
    "ToolPolicy",
    "ToolPolicyError",
    "ToolPolicyDenied",
    "ToolPolicyDeclarationError",
    "get_effective_tool_policy",
    "set_current_tool_policy",
]
