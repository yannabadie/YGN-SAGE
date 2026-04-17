"""LEARN phase: per-step stats, output guardrails, final completion event.

Extracted from agent_loop.py run() — the LEARN block within the loop
and the post-loop output guardrail + final completion event.
"""
from __future__ import annotations

import logging
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from sage.agent_loop import AgentLoop

log = logging.getLogger(__name__)


async def learn_step(loop: AgentLoop) -> None:
    """Execute the in-loop LEARN phase: memory stats, evolution grid snapshot.

    Called at the end of each step when tool calls were made (loop continues).
    """
    from sage.agent_loop import LoopPhase

    aio = loop._compute_aio()
    wall = time.perf_counter() - loop.start_time
    learn_meta: dict[str, Any] = {
        "aio_ratio": aio,
        "events": loop.working_memory.event_count(),
        "wall_time_s": round(wall, 1),
        "cost_usd": round(loop.total_cost_usd, 4),
    }

    # Sub-agent pool status (if wired)
    if loop.agent_pool and hasattr(loop.agent_pool, "list_agents"):
        learn_meta["sub_agents"] = loop.agent_pool.list_agents()

    # Semantic memory stats (if wired)
    if loop.semantic_memory:
        learn_meta["semantic_entities"] = loop.semantic_memory.entity_count()

    loop._emit(LoopPhase.LEARN, **learn_meta)


async def learn_final(task: str, result_text: str, loop: AgentLoop) -> str:
    """Execute the post-loop LEARN phase: output guardrails, final event.

    Called once after the main loop exits.
    Returns the final result text.
    """
    from sage.agent_loop import LoopPhase

    # Expose state for QualityEstimator signals
    loop._last_avr_iterations = loop._s2_avr_retries

    # Output guardrail check
    if loop.guardrail_pipeline and result_text and not loop._skip_guardrails:
        try:
            output_results = await loop.guardrail_pipeline.check_all(
                output=result_text,
                context={
                    "cost_usd": loop.total_cost_usd,
                    "steps": loop.step_count,
                },
            )
            for r in output_results:
                loop._emit(
                    LoopPhase.LEARN,
                    guardrail="output",
                    guardrail_passed=r.passed,
                    guardrail_reason=r.reason,
                )
        except Exception as e:
            log.warning("Output guardrail error: %s", e)

    # Final completion event (includes response text for dashboard)
    final_meta: dict[str, Any] = {
        "result": "complete",
        "steps": loop.step_count,
        "cost_usd": round(loop.total_cost_usd, 4),
        "response_text": result_text or "",
        "task": task,
    }
    if loop.semantic_memory:
        final_meta["semantic_entities"] = loop.semantic_memory.entity_count()
    loop._emit(LoopPhase.LEARN, **final_meta)

    # Restore validation_level if S3->S2 degradation occurred (multi-run safety)
    loop.config.validation_level = loop._original_validation_level

    if result_text:
        return result_text
    # Salvage the last real assistant content if the loop exited without a
    # final answer (e.g. max_steps hit mid-tool-calling). Prior behavior
    # returned "Agent finished at step N", which SWE-bench and similar
    # benches happily packaged as a fake "patch" and masked the failure.
    try:
        events = getattr(loop.working_memory, "_events", None) or []
        for event in reversed(events):
            etype, content = (event[0], event[1]) if len(event) >= 2 else ("", "")
            if etype == "ASSISTANT" and isinstance(content, str) and content.strip():
                return content
    except (AttributeError, IndexError):
        pass
    return f"[sage: agent exited after {loop.step_count} steps with no content]"
