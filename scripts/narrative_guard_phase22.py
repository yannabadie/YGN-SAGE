#!/usr/bin/env python3
"""Cycle-13 K Phase 2.2 narrative guard.

Fails closed on stale Phase 2.1 / transition-seam / stage-count vocabulary
inside the Phase 2.2 touch surface.

Usage:
    python scripts/narrative_guard_phase22.py

Allowed hits must carry an inline explanation marker:
    # narrative-guard: allow <reason>
    <!-- narrative-guard: allow <reason> -->

Do not use broad file-level allowlists.

Lock source: cgpro `cgpro_phase22_test_rewrite_20260506` DESIGN_LOCK
2026-05-06. The script body (regex + glob list + allow-marker syntax)
is taken verbatim from the cgpro response. Only the placement moved
from `.tmp/` (cgpro's recommendation) to `scripts/` (this repo's
gitignore excludes `.tmp/`).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]

TARGET_GLOBS: tuple[str, ...] = (
    "docs/adr/ADR-015-*.md",
    "docs/adr/ADR-016-*.md",
    "sage-python/src/sage/pipeline.py",
    "sage-python/src/sage/pipeline_v2/*.py",
    "sage-python/tests/test_pipeline_v2_phase_a_wrappers.py",

    # Phase 2.2 27-file stage seam rewrite scope.
    "sage-python/tests/observability/test_pipeline_spans.py",
    "sage-python/tests/test_e2e_campaign.py",
    "sage-python/tests/test_f7_task_system_forward.py",
    "sage-python/tests/test_model_assigner_top3_logging.py",
    "sage-python/tests/test_online_evolution.py",
    "sage-python/tests/test_oracle_stack.py",
    "sage-python/tests/test_oxiz_pipeline.py",
    "sage-python/tests/test_pillar_logging.py",
    "sage-python/tests/test_pipeline.py",
    "sage-python/tests/test_pipeline_adaptation.py",
    "sage-python/tests/test_pipeline_bandit_causality.py",
    "sage-python/tests/test_pipeline_budget.py",
    "sage-python/tests/test_pipeline_bypass.py",
    "sage-python/tests/test_pipeline_bypass_structural_isolation.py",
    "sage-python/tests/test_pipeline_governance.py",
    "sage-python/tests/test_pipeline_periodic_save_epoch_preflight.py",
    "sage-python/tests/test_pipeline_topology_flags.py",
    "sage-python/tests/test_pipeline_topology_skip_guardrails_decoupling.py",
    "sage-python/tests/test_pipeline_v2_bandit_attribution_invariant.py",
    "sage-python/tests/test_pipeline_v2_control_surface_fields.py",
    "sage-python/tests/test_pipeline_v2_fix_c_budget_tier_no_controller.py",
    "sage-python/tests/test_pipeline_v2_oracle_gate_invariant.py",
    "sage-python/tests/test_pipeline_v2_phase_a_wrappers.py",
    "sage-python/tests/test_run_frame.py",
    "sage-python/tests/test_speculative_routing.py",
    "sage-python/tests/test_system_hint.py",
    "sage-python/tests/test_topology_learn.py",

    # Public narrative surfaces. AGENTS.md is intentionally NOT in this
    # list: it is gitignored at HEAD `96155232` (no tracked AGENTS.md
    # file in the repo) so a fail-closed missing-glob would never pass
    # in a fresh clone. Per cgpro `cgpro_phase22_test_rewrite_20260506`
    # Stage A pre-commit HARD_STOP 2026-05-06.
    ".claude/rules/architecture.md",
    "CLAUDE.md",
    "README.md",
)

STALE_PATTERN = re.compile(
    r"("
    r"Phase A|"
    r"Phase B|"
    r"Phase 2\.1|"
    r"placeholder|"
    r"do\s+NOT\s+move|"
    r"helper ownership migration is Phase C|"
    r"5[-\s]stage|"
    r"5\s+stage|"
    r"six[-\s]stage|"
    r"6\s+stage|"
    r"stage seam|"
    r"transition seam|"
    r"delegator"
    r")",
    re.IGNORECASE,
)

ALLOW_MARKER = re.compile(r"narrative-guard:\s*allow\s+\S+", re.IGNORECASE)


def iter_target_files() -> Iterable[Path]:
    seen: set[Path] = set()
    for pattern in TARGET_GLOBS:
        matches = sorted(ROOT.glob(pattern))
        if not matches:
            # Fail closed: an expected file disappeared or the glob is stale.
            yield ROOT / pattern
            continue
        for path in matches:
            if path not in seen:
                seen.add(path)
                yield path


def main() -> int:
    failures: list[str] = []
    missing: list[str] = []

    for path in iter_target_files():
        rel = path.relative_to(ROOT) if path.is_absolute() and path.exists() else path
        if not path.exists():
            missing.append(str(rel))
            continue

        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not STALE_PATTERN.search(line):
                continue
            if ALLOW_MARKER.search(line):
                continue
            failures.append(f"{rel}:{line_no}: {line.strip()}")

    if missing:
        print("NARRATIVE_GUARD_FAIL: missing expected files/globs:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)

    if failures:
        print("NARRATIVE_GUARD_FAIL: stale/unexplained narrative terms:", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)

    if missing or failures:
        print(
            "\nFix the wording, or add an inline "
            "`narrative-guard: allow <reason>` marker for a genuinely historical reference.",
            file=sys.stderr,
        )
        return 1

    print("NARRATIVE_GUARD_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
