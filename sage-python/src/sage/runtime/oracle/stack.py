from __future__ import annotations

import logging
from typing import Any, Mapping

from sage.runtime.oracle._oracles import (
    _exact_oracle,
    _formal_oracle,
    _llm_judge_oracle,
    _spec_oracle,
    _tool_oracle,
)
from sage.runtime.oracle.config import OracleConfig
from sage.runtime.oracle.verdict import EvidenceRef, OracleVerdict

log = logging.getLogger(__name__)

_HIERARCHY = (
    ("exact", _exact_oracle),
    ("tool", _tool_oracle),
    ("formal", _formal_oracle),
    ("spec", _spec_oracle),
    ("llm_judge", _llm_judge_oracle),
)


def _abstain(view: Any, *, reason_codes: tuple[str, ...]) -> OracleVerdict:
    return OracleVerdict(
        trainable=False,
        verdict_source="abstain",
        quality_label="unknown",
        score=None,
        confidence=1.0,
        reason_codes=reason_codes,
        evidence=(EvidenceRef(run_id=view.run_id),) if hasattr(view, "run_id") else (),
    )


def evaluate(
    view: Any,
    *,
    final_output: str,
    bench_result: Mapping[str, Any] | None = None,
    config: OracleConfig | None = None,
) -> OracleVerdict:
    """Walk the hierarchy: first confident trainable verdict wins."""
    cfg = config or OracleConfig()
    low_confidence: list[OracleVerdict] = []

    for source_name, oracle_fn in _HIERARCHY:
        if source_name not in cfg.allowed_sources:
            continue
        try:
            candidate = oracle_fn(
                view,
                final_output=final_output,
                bench_result=bench_result,
                config=cfg,
            )
        except Exception as exc:  # noqa: BLE001 - oracle failures abstain by design
            log.warning(
                "oracle %s raised %s; falling through hierarchy",
                source_name,
                exc,
            )
            continue
        if candidate is None:
            continue
        if candidate.confidence < cfg.min_confidence:
            low_confidence.append(candidate)
            continue
        if candidate.trainable:
            return candidate
        low_confidence.append(candidate)

    if low_confidence:
        return _abstain(view, reason_codes=("hierarchy_low_confidence_only",))
    return _abstain(view, reason_codes=("hierarchy_exhausted",))
