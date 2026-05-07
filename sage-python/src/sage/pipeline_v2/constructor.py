"""`CognitiveOrchestrationPipeline.__init__` body — Phase 2.2 Stage D3.

Cycle-13 K Phase 2.2 Stage D3 (cgpro `cgpro_phase22_test_rewrite_20260506`,
2026-05-07): the `__init__` body lives here so `pipeline.py` can shrink
under the 300 raw-line hard gate. The public class-level
`CognitiveOrchestrationPipeline.__init__` signature is preserved
unchanged — the method becomes a thin LOCAL-import wrapper that calls
`initialize_pipeline(self, **kwargs)`.

Per cgpro Q3b lock: keep imports local inside the function body to
avoid reintroducing the partial-init / circular-import trap. The
`OracleConfig` and `_resolve_task_budget_usd` references resolve
through `sage.pipeline` at call time, not at module load.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sage.pipeline import CognitiveOrchestrationPipeline


log = logging.getLogger("sage.pipeline")


def initialize_pipeline(
    self: "CognitiveOrchestrationPipeline",
    *,
    router: Any,
    engine: Any,
    assigner: Any,
    provider_pool: Any,
    bandit: Any = None,
    quality_estimator: Any = None,
    event_bus: Any = None,
    llm_provider: Any = None,
    llm_config: Any = None,
    prm: Any = None,
    controller: Any = None,
    smmu: Any = None,
    consolidator: Any = None,
    working_memory: Any = None,
    episodic_memory: Any = None,
    semantic_memory: Any = None,
    memory_agent: Any = None,
    causal_memory: Any = None,
    tool_forge: Any = None,
    tool_registry: Any = None,
    harness_config: Any = None,
    agent_loop: Any = None,
    budget_usd: float | None = None,
    oracle_config: Any | None = None,
    llm_tier: str = "",
) -> None:
    """Body of `CognitiveOrchestrationPipeline.__init__`.

    Behavior preserved byte-identically from the pre-Phase-2.2 inline
    constructor. Only structural change: lifted out of the class so
    pipeline.py meets the < 300 raw-line target.
    """
    from sage.pipeline import _resolve_task_budget_usd
    from sage.pipeline_v2.memory_gate import build_write_gate
    from sage.runtime.oracle.config import OracleConfig

    self.router = router
    self.engine = engine
    self.assigner = assigner
    self.provider_pool = provider_pool
    self.bandit = bandit
    self.quality_estimator = quality_estimator
    self.event_bus = event_bus
    self.llm_provider = llm_provider
    self.llm_config = llm_config
    self.prm = prm
    self.controller = controller
    self.tool_registry = tool_registry
    self._rust_registry = None  # Set by boot if Rust ModelRegistry available
    self._rust_router = None  # Set by boot if Rust SystemRouter available
    self._smmu = smmu
    self.consolidator = consolidator
    self.working_memory = working_memory
    self.episodic_memory = episodic_memory
    # T2 phase 0/1 (cgpro 2026-04-29): forward the other 3 memory
    # backends to per-node agent loops so write-gate skips can target
    # real backends instead of "memory_backend_unwired".
    self.semantic_memory = semantic_memory
    self.memory_agent = memory_agent
    self.causal_memory = causal_memory
    self.tool_forge = tool_forge
    self.harness_config = harness_config  # Meta-Harness: loaded from config/harness.json at boot
    self._harness_patcher = None
    if harness_config:
        try:
            from sage.meta_harness.patcher import HarnessPatcher

            self._harness_patcher = HarnessPatcher(harness_config)
            log.info(
                "Meta-Harness config '%s' loaded: %s",
                harness_config.id,
                harness_config.description,
            )
        except ImportError:
            log.debug("meta_harness module not available, skipping harness config")
    self._agent_loop = agent_loop
    self._task_count = 0
    self.budget_usd = _resolve_task_budget_usd(budget_usd)
    self._llm_tier = llm_tier
    self._oracle_config = oracle_config or OracleConfig()

    # G-series audit fix (2026-04-19 docs/audits/2026-04-18-astropy-14995-*):
    # RustCompositeWriteGate was built, exported, but never called at
    # runtime (investigation confirmed 0 runtime call sites). Memory
    # writes in phases/act.py and _record_to_memory here all skipped
    # the 5-signal salience check.
    #
    # Weights: w_confidence=0.0 because AgentLoop has no per-turn
    # confidence signal — redistributing that 0.25 to novelty (+0.10)
    # and relevance (+0.15) keeps the composite summing to 1.0 and
    # leans on signals that ARE available (task text + content text).
    # Not a heuristic tweak: an honest statement that this engine cannot
    # produce the "confidence" input the research paper assumed.
    #
    # Gate is REBUILT per-task in `run()` (not reset in-place) so the
    # Rust class — which has no `reset_task()` method yet — doesn't need
    # an ABI bump. `_gate_config` holds the construction args; `write_gate`
    # is swapped out per task.
    self._gate_config = dict(
        threshold=0.35,
        w_confidence=0.0,
        w_novelty=0.40,
        w_reliability=0.20,
        w_recency=0.10,
        w_relevance=0.30,
    )
    self.write_gate = build_write_gate(self)
