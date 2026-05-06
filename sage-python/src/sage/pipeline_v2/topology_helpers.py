"""Cycle-13 K Phase 2.1 Step A1 — topology helper module.

cgpro DESIGN_LOCKED 2026-05-06 (`cgpro_phase21_facade_rewrite_20260506`)
extracted the smallest topology helper from `sage.pipeline` so the façade
rewrite can shrink `pipeline.py` toward < 300 LOC without losing the
mockable surface that ~6 test files rely on.

Per cgpro Q3: this module is the future home of all topology-construction
helpers (TemplateStore-backed creation, candidate parsing, structure
logging, budget/cache, single-node fallback). Stage 2 flow stays in
`pipeline_v2/select_topology.py` lisible.

Logger uses ``sage.pipeline`` per cgpro Q7 trap "logger name drift" —
modules carved out of `pipeline.py` keep the legacy logger name so
trace-grep continuity is preserved.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("sage.pipeline")


def build_topology_from_hint(hint: str) -> Any | None:
    """Create a topology from a template hint using Rust TemplateStore.

    No hardcoded prompts — nodes use their role-based defaults.
    The runner builds system prompts from each node's role field.

    Returns ``None`` when the Rust TemplateStore is unavailable
    (sage_core not installed, e.g. CI subset, type-check pass)
    or when the hint does not resolve to a known template.
    """
    try:
        from sage_core import PyTemplateStore  # type: ignore[import-not-found]

        store = PyTemplateStore()
        return store.create(hint, "")
    except (ImportError, ValueError):
        return None
