"""Phase A placeholder for bandit-attribution lifecycle helpers.

Per cgpro 2026-05-05 DESIGN lock + ADR-015 §"Module boundaries":
the bandit-attribution lifecycle (`_emit_bandit_attribution_mismatch`,
`_record_bandit_outcome_checked`, `_clear_bandit_decision`,
`_cancel_bandit_decision`) is the home of invariant 6 (Bandit
attribution singleton settle). Phase A does NOT move these — they
remain as methods of `CognitiveOrchestrationPipeline` in
`sage.pipeline`. Helper ownership migration is Phase C territory.

This module exists so the package layout matches ADR-015 from
day one. Future Phase C may move ownership; until then the file
is intentionally empty.
"""
from __future__ import annotations

__all__: list[str] = []
