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
_MAX_TYPE_IGNORES = 20

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
