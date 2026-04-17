"""Sprint 6 decision helper.

Reads the ablation JSON produced by scripts/run_swebench_ablation.py,
loads each per-config report, and prints a single recommendation:

    GATE A -> v1.0 Release Candidate
    GATE B -> Training revival
    GATE C -> Narrow improvements (keep iterating)

The thresholds mirror docs/ROADMAP_SPRINT6_DECISION.md so the doc and
the code do not drift. Exit code: 0 for A, 1 for B, 2 for C, 3 on
malformed input.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


GATE_A_THRESHOLD = 0.35
GATE_B_THRESHOLD = 0.20


def _load_rate(report_path: Path) -> float | None:
    """Extract pass_rate / resolved_rate from a bench report JSON."""
    if not report_path.exists():
        return None
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    for key in ("pass_rate", "resolved_rate", "accuracy"):
        if key in data:
            return float(data[key])
    # Nested "summary" block used by some bench reports.
    if "summary" in data and isinstance(data["summary"], dict):
        for key in ("pass_rate", "resolved_rate"):
            if key in data["summary"]:
                return float(data["summary"][key])
    return None


def decide(ablation: dict) -> tuple[str, str, int]:
    """Return (gate_label, rationale, exit_code)."""
    results = ablation.get("results", [])
    if not results:
        return ("ERROR", "no 'results' array in ablation JSON", 3)

    rates: dict[str, float] = {}
    for r in results:
        cfg = r.get("config", "?")
        rate = _load_rate(Path(r.get("report_path", "")))
        if rate is not None:
            rates[cfg] = rate

    if "full" not in rates:
        return ("ERROR", "no 'full' config report found or unreadable", 3)

    full = rates["full"]
    bare = rates.get("bare")
    delta = (full - bare) if bare is not None else None

    # Gate A
    if full >= GATE_A_THRESHOLD:
        reason = f"full={full:.1%} >= {GATE_A_THRESHOLD:.0%}"
        if delta is not None:
            reason += f" (delta vs bare {delta:+.1%})"
        return ("A", reason, 0)

    # Gate B
    if full < GATE_B_THRESHOLD:
        reason = f"full={full:.1%} < {GATE_B_THRESHOLD:.0%} — architecture not the bottleneck"
        return ("B", reason, 1)

    # Gate C
    reason = f"full={full:.1%} in [{GATE_B_THRESHOLD:.0%}, {GATE_A_THRESHOLD:.0%}) — iterate"
    if delta is not None:
        reason += f" (delta vs bare {delta:+.1%})"
    # Identify dominant ablated component for the Gate C recommendation.
    drops = []
    for cfg in ("no_sage_recurse", "no_toolforge", "no_topology"):
        if cfg in rates:
            drops.append((cfg, full - rates[cfg]))
    if drops:
        drops.sort(key=lambda kv: kv[1], reverse=True)
        top = drops[0]
        reason += f"; biggest drop from ablating {top[0]} ({top[1]:+.1%})"
    return ("C", reason, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ablation_json", type=Path,
                        help="Path to the ablation summary JSON.")
    args = parser.parse_args()

    try:
        ablation = json.loads(args.ablation_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Could not read ablation JSON: {exc}", file=sys.stderr)
        sys.exit(3)

    gate, reason, exit_code = decide(ablation)
    print(f"GATE {gate}: {reason}")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
