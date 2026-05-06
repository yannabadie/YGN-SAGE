"""ToolCapability enum + ToolPolicy effective-grant container.

Per cgpro DESIGN_LOCKED 2026-05-06 (conv cgpro_phase15_toolpolicy_20260506):
- 6-tier capability vocabulary, single label per tool.
- Effective grants = {pure} ∪ TOML ∪ env(SAGE_TOOL_GRANTS) ∪ programmatic.
- No hierarchical implicit closure: each tier granted exactly.
- ContextVar-based ambient policy for thread-safe scoping.
"""
from __future__ import annotations

import enum
import logging
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from sage.policy.errors import ToolPolicyDenied

log = logging.getLogger(__name__)


class ToolCapability(str, enum.Enum):
    """The 6 capability tiers per Phase 1.5 DESIGN.

    Single label per tool — represents the maximum-safe summary, not a
    proof. Combinations (e.g. network + write_local) MUST classify as
    `dangerous` per the DESIGN trap rule.
    """

    PURE = "pure"
    READ_LOCAL = "read_local"
    WRITE_LOCAL = "write_local"
    NETWORK = "network"
    SUBPROCESS = "subprocess"
    DANGEROUS = "dangerous"


_VALID_CAPABILITY_VALUES: frozenset[str] = frozenset(c.value for c in ToolCapability)


def _coerce_capability(value: Any) -> ToolCapability:
    """Accept ToolCapability or its string value; reject everything else.

    Used at the public boundary `Tool(..., capability=...)` so callers
    can pass the enum or the raw string interchangeably.
    """
    if isinstance(value, ToolCapability):
        return value
    if isinstance(value, str) and value in _VALID_CAPABILITY_VALUES:
        return ToolCapability(value)
    raise ValueError(
        f"Invalid ToolCapability value: {value!r}. "
        f"Must be one of {sorted(_VALID_CAPABILITY_VALUES)}."
    )


def _parse_grants_csv(raw: str) -> set[ToolCapability]:
    """Parse a comma-separated grant list, raising on unknown values."""
    if not raw:
        return set()
    out: set[ToolCapability] = set()
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.add(_coerce_capability(token))
    return out


def _grants_from_env() -> set[ToolCapability]:
    raw = os.environ.get("SAGE_TOOL_GRANTS", "")
    return _parse_grants_csv(raw)


def _grants_from_toml(path: Path | None = None) -> set[ToolCapability]:
    """Read grants from `~/.sage/tool_policy.toml`.

    Schema:
        [tool_policy]
        grants = ["read_local", "network"]

    Missing file → empty set (no error). Malformed file → log warning
    and return empty (operator can fix without breaking boot).
    """
    target = path if path is not None else Path.home() / ".sage" / "tool_policy.toml"
    if not target.is_file():
        return set()
    try:
        # Python 3.11+ stdlib tomllib; fall back to tomli if 3.10.
        try:
            import tomllib  # type: ignore[import-not-found]
        except ImportError:  # pragma: no cover
            import tomli as tomllib  # type: ignore[import-not-found,no-redef]

        with target.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception as exc:
        log.warning("ToolPolicy TOML load failed at %s: %s — ignoring file.", target, exc)
        return set()

    section = data.get("tool_policy", {})
    grants_raw = section.get("grants", [])
    if not isinstance(grants_raw, list):
        log.warning(
            "ToolPolicy TOML at %s has tool_policy.grants of type %s; expected list. Ignoring.",
            target,
            type(grants_raw).__name__,
        )
        return set()
    out: set[ToolCapability] = set()
    for entry in grants_raw:
        if not isinstance(entry, str):
            continue
        try:
            out.add(_coerce_capability(entry))
        except ValueError as exc:
            log.warning("ToolPolicy TOML at %s: %s", target, exc)
    return out


@dataclass(frozen=True)
class ToolPolicy:
    """Immutable effective-grant container.

    Construct via `ToolPolicy.default()` (returns `{pure}` only) and
    extend via `policy.grant(tier)` which returns a NEW policy. The
    immutability is intentional: scoped policies travel through
    `ContextVar` and concurrent runs MUST NOT mutate each other's view.
    """

    grants: frozenset[ToolCapability] = field(default_factory=frozenset)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> "ToolPolicy":
        """The default effective grant set: `{pure}` only.

        Every other tier requires an explicit grant via TOML, env, or
        programmatic call.
        """
        return cls(grants=frozenset({ToolCapability.PURE}))

    @classmethod
    def from_environment(
        cls,
        *,
        toml_path: Path | None = None,
        extra_grants: Iterable[ToolCapability | str] | None = None,
    ) -> "ToolPolicy":
        """Compose grants = {pure} ∪ TOML ∪ env(SAGE_TOOL_GRANTS) ∪ extra.

        Used at boot time to materialise an effective policy from all
        configured sources. `extra_grants` is the programmatic channel.
        """
        grants: set[ToolCapability] = {ToolCapability.PURE}
        grants |= _grants_from_toml(toml_path)
        grants |= _grants_from_env()
        if extra_grants:
            for entry in extra_grants:
                grants.add(_coerce_capability(entry))
        return cls(grants=frozenset(grants))

    # ------------------------------------------------------------------
    # Mutation (returns new instance)
    # ------------------------------------------------------------------

    def grant(self, tier: ToolCapability | str) -> "ToolPolicy":
        """Return a NEW policy with `tier` added to the effective grants."""
        return ToolPolicy(grants=frozenset(self.grants | {_coerce_capability(tier)}))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def allows(self, capability: ToolCapability | str) -> bool:
        """Return True iff `capability` is in the effective grant set."""
        return _coerce_capability(capability) in self.grants

    def assert_allowed(self, capability: ToolCapability | str, *, tool_name: str = "<unknown>") -> None:
        """Raise `ToolPolicyDenied` iff `capability` is not granted.

        Tools with declared `dangerous` capability are rejected unless
        the policy holds the EXACT `dangerous` grant — no implicit
        closure from any other tier.
        """
        cap = _coerce_capability(capability)
        if cap not in self.grants:
            raise ToolPolicyDenied(
                f"Tool {tool_name!r} requires capability {cap.value!r} but the "
                f"effective ToolPolicy only grants {sorted(g.value for g in self.grants)}. "
                f"To grant: set SAGE_TOOL_GRANTS={cap.value} or use "
                f"`policy.grant({cap.value!r})` programmatically."
            )


# ---------------------------------------------------------------------------
# Ambient policy (ContextVar)
# ---------------------------------------------------------------------------

_CURRENT_POLICY: ContextVar[ToolPolicy] = ContextVar(
    "sage_tool_policy_current",
    default=ToolPolicy.default(),
)


def get_effective_tool_policy() -> ToolPolicy:
    """Return the currently-installed `ToolPolicy` for this context.

    Defaults to `ToolPolicy.default()` (= `{pure}`-only) when no policy
    has been explicitly set for the current ContextVar scope.
    """
    return _CURRENT_POLICY.get()


def set_current_tool_policy(policy: ToolPolicy) -> Any:
    """Install `policy` as the effective policy for this context.

    Returns the ContextVar Token from `set()` so callers that scope a
    policy locally (e.g. a per-run boot path) can `reset()` it on exit.
    """
    return _CURRENT_POLICY.set(policy)
