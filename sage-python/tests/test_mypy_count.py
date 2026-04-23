"""Regression gate for type: ignore comment count.

Tracks the number of ``# type: ignore`` comments in the sage source tree
and fails if the count increases beyond the established ceiling. This
prevents new type-safety escapes from creeping in.
"""

from __future__ import annotations

import re
from pathlib import Path

# Maximum allowed type: ignore comments (regression ceiling).
# Wave 1+2 cleanup removed 8 fixable ignores from 20 original.
# Remaining are all in "skip" categories (third-party / unfixable):
#   11 = third-party imports (a2a x6, sentence-transformers, sage_core x2, openai, google-genai)
#    5 = pipeline.py typing (verify_provider_assignment, ProviderSpec, _emit, TopologyGraph,
#        TopologyRunner, TopologyExecutor assignments)
#    3 = ssl private API, pipeline_stages dict arg-type, providers openai SDK arg-type
#    1 = evolution cli sage_core import
# NOTE: concurrent linter activity may add new third-party ignores.
# Raised from 20 to 23 after QualityEstimator rewrite (3 new sage_core imports).
# Raised from 23 to 25 after orchestrator cleanup (2 new pipeline sage_core imports).
# Raised from 25 to 27: +1 FrugalGPT cascade (pipeline.py sage_core import),
# +1 apps_bench.py datasets import, +1 livecodebench_bench.py datasets import,
# -1 orchestrator.py deleted (net +2).
# Raised from 27 to 29: +1 Path 6 TopologyEdge import (pipeline.py),
# +1 FrugalGPT cascade retry TopologyExecutor import (pipeline.py).
# Raised from 29 to 36: +5 a2a_server.py new server imports (import-untyped),
# +2 quality_estimator.py Rust imports (return type).
# Raised from 36 to 41 (2026-04-23, audit-aware catch-up): the ceiling
# drifted between commit 6ef60cf and the 2026-04-23 bench chain as
# cross-cutting work landed. Net +5 across:
#   +2 bench/swebench_ca_patch.py (UTF-8/CRLF wrappers on stdlib
#     write_text / run_evaluation.open — #[attr-defined]/method-assign)
#   +1 bench/swebench_patch_repair.py (_extract_patch misc)
#   +1 bench/swebench_bench.py ssl private API (assignment)
#   +1 topology_controller.py _RustTopologyControllerImpl sentinel.
# All five are in skip categories (ssl private API, stdlib
# monkey-patch, third-party or Rust bindings). No new ignores from the
# 2026-04-23 bench/sandbox chain itself — the gen-log commit's one
# ignore was retired in the same pass via setattr.
_MAX_TYPE_IGNORES = 41

_SAGE_SRC = Path(__file__).resolve().parent.parent / "src" / "sage"
_PATTERN = re.compile(r"#\s*type:\s*ignore")


def _count_type_ignores() -> list[tuple[str, int, str]]:
    """Return list of (relative_path, line_number, line_text) for all type: ignore hits."""
    hits: list[tuple[str, int, str]] = []
    for py_file in sorted(_SAGE_SRC.rglob("*.py")):
        try:
            lines = py_file.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(lines, start=1):
            if _PATTERN.search(line):
                rel = py_file.relative_to(_SAGE_SRC)
                hits.append((str(rel), i, line.strip()))
    return hits


def test_type_ignore_count_does_not_increase() -> None:
    """Regression gate: type: ignore count must not exceed ceiling."""
    hits = _count_type_ignores()
    count = len(hits)

    # Print for visibility in CI output
    print(f"\n[type: ignore audit] Found {count} comments (ceiling: {_MAX_TYPE_IGNORES})")
    for path, lineno, text in hits:
        print(f"  {path}:{lineno}  {text}")

    assert count <= _MAX_TYPE_IGNORES, (
        f"type: ignore count ({count}) exceeds ceiling ({_MAX_TYPE_IGNORES}). "
        f"Fix the new ignores or raise the ceiling with justification."
    )
