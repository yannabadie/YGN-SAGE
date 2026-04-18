"""THINK phase: LLM call, cost estimation, S3 PRM validation, CEGAR repair.

Extracted from agent_loop.py run() — the THINK block within the while loop.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from sage.llm.base import Message, Role, ToolDef

if TYPE_CHECKING:
    from sage.agent_loop import AgentLoop

log = logging.getLogger(__name__)


@dataclass
class _ThinkResult:
    """Return bundle from the think phase."""

    content: str = ""
    response: Any = None
    brake: bool = False
    loop_action: str = "proceed"  # "proceed", "continue", "break"


def _extract_step_cost(response: Any, model_name: str, content: str) -> float:
    """Single-LLM-call cost in USD. Prefers the provider's own figure.

    LiteLLM populates ``response._hidden_params["response_cost"]`` by default;
    ``LiteLLMProvider._finalize_response`` copies it into
    ``LLMResponse.usage["cost_usd"]``. When present we use it verbatim —
    the token-count × local cost table fallback only runs when the provider
    returned no cost (e.g. an abstaining provider, a mocked response in
    tests, or a model id missing from the hidden-params cost catalogue).

    Until 2026-04-18 the fallback was always used and silently floored to $0
    whenever the model id was missing from ``_COST_PER_1K``, which is why
    every v5* bench run reported ``_cost_usd=0.0`` even when the providers
    were metering correctly.
    """
    from sage.agent_loop import _COST_PER_1K, _estimate_tokens
    from sage.constants import DEFAULT_COST_PER_1K

    usage = getattr(response, "usage", None) or {}
    provider_cost = usage.get("cost_usd") if isinstance(usage, dict) else None
    if provider_cost is not None and provider_cost > 0:
        return float(provider_cost)

    actual_total = usage.get("total_tokens") if isinstance(usage, dict) else None
    tokens = _estimate_tokens(content, actual_count=actual_total)
    cost_per_k = _COST_PER_1K.get(model_name, DEFAULT_COST_PER_1K)
    return (tokens / 1000) * cost_per_k


async def think(
    task: str,
    messages: list[Message],
    system_prompt: str,
    tool_defs: list[ToolDef],
    loop: AgentLoop,
) -> _ThinkResult:
    """Execute the THINK phase of the agent loop.

    Handles:
    - Topology-aware multi-agent execution (first step only)
    - LLM inference call
    - Cost estimation and token counting
    - Entropy calculation and CGRS self-braking
    - MEM1 rolling internal state generation
    - System 3 validation (Z3 PRM) with CEGAR repair and S3->S2 degradation

    Returns a _ThinkResult with the content, response, brake flag, and loop_action.
    loop_action is "continue" if the step should be retried (S3 retry/degrade),
    "break" if topology result was used, or "proceed" to continue to ACT/LEARN.
    """
    from sage.agent_loop import LoopPhase, _estimate_tokens, _text_entropy, _COST_PER_1K

    # Topology-aware execution: delegate to TopologyRunner for multi-node
    if loop.step_count == 1:
        topology_result = await loop._run_topology(task)
        if topology_result is not None:
            # Multi-agent topology executed — use its result directly
            loop.working_memory.add_event("ASSISTANT", topology_result)
            return _ThinkResult(
                content=topology_result,
                loop_action="break",
            )

    model_name = loop.config.llm.model
    loop._emit(LoopPhase.THINK, model=model_name)

    t0 = time.perf_counter()
    # ExoCortex: passive grounding removed per Sprint 3 evidence.
    # Use active tool (search_exocortex) instead — agent invokes when needed.

    response = await loop._llm.generate(
        messages=messages,
        tools=tool_defs if tool_defs else None,
        config=loop.config.llm,
    )
    inference_ms = (time.perf_counter() - t0) * 1000
    loop.total_inference_time += inference_ms / 1000

    content = response.content or ""

    step_cost = _extract_step_cost(response, model_name, content)
    loop.total_cost_usd += step_cost

    # Entropy for CGRS self-braking
    entropy = _text_entropy(content)
    brake = False
    if loop.metacognition:
        loop.metacognition.record_output_entropy(entropy)
        brake = loop.metacognition.should_brake()

    # Emit THINK results (including content for dashboard real-time display)
    loop._emit(
        LoopPhase.THINK,
        model=model_name,
        content=content,
        latency_ms=round(inference_ms, 1),
        cost_usd=round(loop.total_cost_usd, 4),
        entropy=round(entropy, 3),
        brake=brake,
    )

    # MEM1: generate rolling internal state every step
    if loop.memory_compressor and content:
        await loop.memory_compressor.generate_internal_state(
            f"[Step {loop.step_count}] {content[:300]}"
        )

    # System 3 validation (Z3 PRM) -- hard gating with CEGAR repair
    if loop.config.validation_level >= 3 and content:
        r_path, details = loop.prm.calculate_r_path(content)
        loop._emit(LoopPhase.THINK, r_path=r_path, details=details)

        if r_path < 0.0 and "error" in details:
            loop._s3_retries += 1
            if loop._s3_retries <= loop._max_s3_retries:
                messages.append(Message(
                    role=Role.USER,
                    content=(
                        "SYSTEM: Your reasoning lacks formal assertions. "
                        "Use <think> tags with Z3 assertions:\n"
                        "- assert bounds(addr, limit)\n"
                        "- assert loop(var)\n"
                        "- assert arithmetic(expr, expected)\n"
                        '- assert invariant("precondition", "postcondition")\n'
                        "Include at least one formal assertion per reasoning step."
                    ),
                ))
                return _ThinkResult(
                    content=content,
                    response=response,
                    brake=brake,
                    loop_action="continue",
                )

            # Max retries reached — attempt CEGAR repair
            inv_feedback = getattr(loop.prm.kg, "_last_invariant_feedback", [])
            repaired = await loop._cegar_repair(content, details, inv_feedback)
            if repaired is not None:
                content = repaired
                loop._s3_retries = 0
            else:
                # CEGAR failed — degrade to S2 (NOT accept unverified)
                log.warning(
                    "S3 verification failed after CEGAR repair — "
                    "degrading to S2 AVR."
                )
                loop._emit(
                    LoopPhase.THINK,
                    s3_degradation=True,
                    reason="CEGAR repair failed",
                )
                loop.config.validation_level = 2
                loop._s3_degraded = True  # Prevent S2->S3 re-escalation
                loop._s3_retries = 0
                loop._s2_avr_retries = 0
                return _ThinkResult(
                    content=content,
                    response=response,
                    brake=brake,
                    loop_action="continue",  # Re-enter loop with S2 validation
                )
        else:
            loop._s3_retries = 0  # Reset on success

    return _ThinkResult(
        content=content,
        response=response,
        brake=brake,
        loop_action="proceed",
    )
