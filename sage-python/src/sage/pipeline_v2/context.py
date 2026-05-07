"""`PipelineContext` dataclass — canonical source of truth.

The canonical `PipelineContext` dataclass body lives here.
`sage.pipeline` re-exports it as a 1-line import so the legacy
`from sage.pipeline import PipelineContext` keeps resolving.

`PipelineContext.__module__` is forced back to ``"sage.pipeline"``
after the dataclass body so:

  - existing tests / bench / dashboard / observability consumers
    that compare ``PipelineContext.__module__ == "sage.pipeline"``
    keep passing byte-identical
  - `repr(ctx)` still shows the legacy module path
  - pickle support is unchanged for any consumer pickling
    `PipelineContext` instances against the legacy path

Module-level constants (`BUDGET_EXCEEDED_RESULT`,
`_BANDIT_ATTRIBUTION_REASON_CODES`, etc.) stay in `sage.pipeline` —
they are not context-side state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from sage.pipeline import BanditAttributionState
    from sage.pipeline_stages import DAGFeatures
    from sage.runtime.oracle import OracleVerdict


@dataclass
class PipelineContext:
    """State that flows through the 6 pipeline stages."""

    task: str
    budget: float = 5.0
    domain: str = ""
    system: int = 0
    task_dag: Any = None
    dag_features: "DAGFeatures | None" = None
    topology: Any = None
    topology_id: str = ""
    assignments: dict[int, str] = field(default_factory=dict)
    provider_hints: dict[int, str] = field(default_factory=dict)  # node_idx -> provider_name
    result: str = ""
    latency_ms: float = 0.0
    cost: float = 0.0
    bandit_decision_id: str = ""
    bandit_model_id: str = ""
    bandit_template: str = ""
    bandit_context: list[float] = field(default_factory=list)
    executed_model_id: str = ""
    executed_template: str = ""
    executed_model_ids: list[str] = field(default_factory=list)
    bandit_attribution_state: "BanditAttributionState" = "skipped"
    verification_passed: bool = True
    axis_hint: str = ""  # MASBENCH axis hint for topology selection
    tool_call_count: int = 0
    tool_turn_count: int = 0
    executed_commands: list[str] = field(default_factory=list)
    executed_tools: list[str] = field(default_factory=list)
    cost_tracker: Any = None
    oracle_verdict: "OracleVerdict | None" = None
    bench_result: "Mapping[str, Any] | None" = None


# cgpro round-3 garde-fou: preserve `PipelineContext.__module__ ==
# "sage.pipeline"` so existing assertions in tests / bench /
# dashboards / observability continue to pass. The dataclass body
# physically lives here, but its declared module path is the legacy
# `sage.pipeline` for byte-identical observable behavior.
PipelineContext.__module__ = "sage.pipeline"


__all__ = ["PipelineContext"]
