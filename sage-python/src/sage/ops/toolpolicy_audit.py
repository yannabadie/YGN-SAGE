"""ToolPolicy audit CLI.

Cycle-13 K Phase 1.5 Commit 4 (cgpro DESIGN_LOCKED 2026-05-06): list
every tool the boot path knows about (built-in manifest entries +
runtime-registered tools, if a registry is reachable), their resolved
ToolCapability declaration, the source of declaration, the effective
grant set, and per-tool allow/deny status.

Usage:
  python -m sage.ops.toolpolicy_audit              # human-readable summary
  python -m sage.ops.toolpolicy_audit --json       # machine-readable
  python -m sage.ops.toolpolicy_audit --strict     # exit 1 on unresolved

The CLI does NOT execute any tool handlers — it only inspects
declaration / resolution / policy. Per cgpro DESIGN trap 7:
"Le CLI audit doit inspecter déclaration/résolution et grants
effectifs sans exécuter les handlers."
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from typing import Any

from sage.policy.errors import ToolPolicyDeclarationError
from sage.policy.manifest import (
    _BUILTIN_TOOL_CAPABILITIES,
    _CLASS_CAPABILITY_DEFAULTS,
)
from sage.policy.tool_policy import (
    ToolCapability,
    ToolPolicy,
    get_effective_tool_policy,
)

log = logging.getLogger("sage.ops.toolpolicy_audit")


@dataclass
class _ToolAuditEntry:
    """One row of the audit report."""

    name: str
    capability: str  # the resolved enum value, OR "<unresolved>"
    source: str  # "manifest" | "class_default" | "explicit" | "<unresolved>"
    allowed: bool


@dataclass
class _AuditReport:
    """Structured audit output."""

    effective_grants: list[str]
    total_tools: int
    by_capability: dict[str, int]
    unresolved: list[str]
    entries: list[_ToolAuditEntry]
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "effective_grants": sorted(self.effective_grants),
            "total_tools": self.total_tools,
            "by_capability": self.by_capability,
            "unresolved": sorted(self.unresolved),
            "entries": [asdict(e) for e in self.entries],
        }


def build_report(policy: ToolPolicy | None = None) -> _AuditReport:
    """Audit the built-in tool manifest against `policy`.

    `policy=None` reads the ambient policy via `get_effective_tool_policy`.
    The audit only inspects the static built-in manifest at this stage —
    a future extension could pass a live `ToolRegistry` to extend the
    inventory with runtime-registered tools.
    """
    if policy is None:
        policy = get_effective_tool_policy()

    entries: list[_ToolAuditEntry] = []
    by_capability: dict[str, int] = {}
    unresolved: list[str] = []

    # Built-in name → capability map.
    for name, cap in sorted(_BUILTIN_TOOL_CAPABILITIES.items()):
        entries.append(
            _ToolAuditEntry(
                name=name,
                capability=cap.value,
                source="manifest",
                allowed=policy.allows(cap),
            )
        )
        by_capability[cap.value] = by_capability.get(cap.value, 0) + 1

    # Class-level defaults are reported separately (their concrete
    # instances may not register at every boot, but the operator
    # should know the default is on file).
    for class_name, cap in sorted(_CLASS_CAPABILITY_DEFAULTS.items()):
        entries.append(
            _ToolAuditEntry(
                name=f"<class:{class_name}>",
                capability=cap.value,
                source="class_default",
                allowed=policy.allows(cap),
            )
        )

    return _AuditReport(
        effective_grants=[g.value for g in policy.grants],
        total_tools=len(entries),
        by_capability=by_capability,
        unresolved=unresolved,
        entries=entries,
        ok=len(unresolved) == 0,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit YGN-SAGE ToolPolicy declarations and grants.")
    parser.add_argument("--strict", action="store_true", help="Exit 1 on unresolved tools.")
    parser.add_argument("--json", action="store_true", help="Print structured JSON.")
    args = parser.parse_args(argv)

    report = build_report()

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print("YGN-SAGE ToolPolicy audit")
        print(f"  Effective grants : {sorted(report.effective_grants)}")
        print(f"  Total tools      : {report.total_tools}")
        print("  By capability    :")
        for cap, count in sorted(report.by_capability.items()):
            print(f"    {cap}: {count}")
        if report.unresolved:
            print(f"  Unresolved ({len(report.unresolved)}):")
            for name in sorted(report.unresolved):
                print(f"    {name}")
        denied = [e for e in report.entries if not e.allowed]
        if denied:
            print(f"  Denied under current policy ({len(denied)}):")
            for entry in denied:
                print(f"    {entry.name} ({entry.capability}, source={entry.source})")
        print(f"  Result: {'OK' if report.ok else 'UNRESOLVED'}")

    if args.strict and not report.ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
