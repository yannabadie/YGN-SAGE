"""PERCEIVE phase: routing, input guardrails, memory context injection, code-task detection.

Extracted from agent_loop.py run() — lines 419-521 of the original.

C4b (2026-04-22): `perceive()` now accepts either `str` (legacy path,
what `AgentLoop.run()` calls after the C4 early-rendering dispatch
in `AgentSystem.run()`) OR a `TaskInput` (direct-caller path for
future consumers that want layered composition without round-tripping
through a rendered string). When a `TaskInput` is passed:
  * The `prompt` field becomes the USER message (what routing /
    guardrails / working memory see).
  * `instructions` (if non-empty) is injected into the system prompt
    as a `## Workflow` section, layered after the tool-affordance
    block and before the validation-level augmentation.
  * `tools_filter` (if not None) overrides `loop.config.tools` for
    `get_tool_defs` + `describe_for_prompt`. This lets chat REPLs
    and future benches restrict the toolset per-turn without
    mutating the underlying agent config.
For string callers, perceive wraps the input in a minimal TaskInput
with all the new fields at their zero-values — behavior is
byte-identical to the pre-C4b path.
"""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from sage.input.types import TaskInput
from sage.llm.base import Message, Role, ToolDef

if TYPE_CHECKING:
    from sage.agent_loop import AgentLoop

log = logging.getLogger(__name__)


class _PerceiveResult:
    """Return bundle from the perceive phase."""

    __slots__ = ("messages", "system_prompt", "tool_defs", "blocked_reason")

    def __init__(
        self,
        messages: list[Message],
        system_prompt: str,
        tool_defs: list[ToolDef],
        blocked_reason: str | None = None,
    ):
        self.messages = messages
        self.system_prompt = system_prompt
        self.tool_defs = tool_defs
        self.blocked_reason = blocked_reason


def _coerce_to_task_input(task: "str | TaskInput") -> TaskInput:
    """Wrap a raw string into a minimal TaskInput for unified handling.

    Pre-C4b behavior is preserved byte-for-byte because the wrapped
    TaskInput has `instructions=""`, `tools_filter=None`, and
    `response_format=TEXT` — so every downstream branch that checks
    those fields falls into the same path a string caller used to
    take.
    """
    if isinstance(task, TaskInput):
        return task
    return TaskInput(prompt=task, source="direct")


async def perceive(
    task: "str | TaskInput", loop: AgentLoop
) -> _PerceiveResult:
    """Execute the PERCEIVE phase of the agent loop.

    Handles:
    - Metacognitive routing (S1/S2/S3 classification)
    - Input guardrail checks (may block early)
    - System prompt augmentation (S2 edge-case hints, S3 Z3 instructions)
    - Tool-affordance injection
    - Per-source workflow section (C4b) from `task_input.instructions`
    - Semantic memory context injection
    - S-MMU context injection
    - Working memory initialization

    Returns a _PerceiveResult with messages, system_prompt, tool_defs,
    and optionally a blocked_reason if input guardrails blocked the task.
    """
    from sage.agent_loop import LoopPhase, _is_code_task

    # C4b: coerce inputs to a single shape so every branch below
    # operates on the same fields. String callers keep byte-identical
    # behavior via a wrapper with zero-value fields.
    task_input = _coerce_to_task_input(task)
    task_str = task_input.prompt

    perceive_meta: dict[str, Any] = {"task": task_str, "agent": loop.config.name}

    # Metacognitive routing (if wired)
    if loop._skip_routing:
        perceive_meta["system"] = 2
        perceive_meta["routing_source"] = "ablation_forced_s2"
    elif loop.metacognition:
        profile = await loop.metacognition.assess_complexity_async(task_str)
        decision = loop.metacognition.route(profile)
        perceive_meta["system"] = decision.system
        perceive_meta["routed_tier"] = decision.llm_tier
        perceive_meta["use_z3"] = decision.use_z3
        perceive_meta["validation_level"] = decision.validation_level
        perceive_meta["complexity"] = round(profile.complexity, 2)
        perceive_meta["uncertainty"] = round(profile.uncertainty, 2)
        if profile.reasoning:
            perceive_meta["routing_reason"] = profile.reasoning
        perceive_meta["routing_source"] = (
            "llm" if profile.reasoning != "heuristic" else "heuristic"
        )

    loop._emit(LoopPhase.PERCEIVE, **perceive_meta)

    # Input guardrail check
    if loop.guardrail_pipeline and not loop._skip_guardrails:
        try:
            input_results = await loop.guardrail_pipeline.check_all(
                input=task_str, context={"step": 0, "agent": loop.config.name}
            )
            for r in input_results:
                loop._emit(
                    LoopPhase.PERCEIVE,
                    guardrail=(
                        r.__class__.__name__
                        if hasattr(r, "__class__")
                        else "input"
                    ),
                    guardrail_passed=r.passed,
                    guardrail_reason=r.reason,
                )
            if loop.guardrail_pipeline.any_blocked(input_results):
                blocked = [r for r in input_results if not r.passed]
                return _PerceiveResult(
                    messages=[],
                    system_prompt="",
                    tool_defs=[],
                    blocked_reason=f"Blocked by guardrail: {blocked[0].reason}",
                )
        except Exception as e:
            log.warning("Input guardrail error: %s", e)

    # Build system prompt with validation-level augmentation
    system_prompt = loop.config.system_prompt

    # C4b: prefer the TaskInput's tools_filter when the caller
    # supplied one (e.g. chat mode's read-only allowlist). Fall back
    # to the agent's static config.tools (e.g. per-node topology
    # filtering from agent_loop_factory) so topology verifier /
    # formatter / aggregator nodes keep their restricted toolset.
    tool_names = (
        task_input.tools_filter
        if task_input.tools_filter is not None
        else (loop.config.tools if loop.config.tools else None)
    )

    # Inject tool affordance block (2026-04-21 audit fix). The agent
    # already gets structured tool schemas via ``tool_defs`` below, but
    # on some benches (SWE-bench in particular) the user-facing prompt
    # only named execute_bash so the agent never discovered other tools
    # like search_exocortex. Auto-injecting a ``## Available Tools``
    # section at the system-prompt level fixes this for every caller
    # without each bench having to list tools by hand. See
    # ``docs/audits/2026-04-21-exocortex-swebench-usage.md``.
    tool_affordance = loop._tools.describe_for_prompt(tool_names)
    if tool_affordance:
        system_prompt = f"{system_prompt}\n\n{tool_affordance}"

    # C4b: source-specific workflow section. When a TaskInput carries
    # `instructions` (e.g. SWE-bench's "Mandatory Workflow" block),
    # inject it under a dedicated heading. String callers get an
    # empty `instructions` field (wrapper default) so this is a no-op
    # for them and pre-C4b byte-identity is preserved.
    if task_input.instructions:
        system_prompt = (
            f"{system_prompt}\n\n## Workflow\n\n{task_input.instructions}"
        )

    if loop.config.validation_level >= 3:
        system_prompt += (
            "\n\nCRITICAL: Use <think>...</think> tags for formal reasoning. "
            "Include Z3-verifiable assertions in your reasoning steps:\n"
            "- assert bounds(address, limit) — prove memory safety\n"
            "- assert loop(variable) — prove loop termination\n"
            "- assert arithmetic(expression, expected) — prove arithmetic correctness\n"
            '- assert invariant("precondition", "postcondition") — prove logical invariants\n'
            "Your reasoning is verified by Z3 SMT solver. "
            "Steps with proven assertions score 1.0. Steps without score 0.0."
        )
    elif loop.config.validation_level >= 2:
        system_prompt += (
            "\n\nUse step-by-step reasoning to solve this task. "
            "Show your work clearly."
        )
        if _is_code_task(task_str):
            system_prompt += (
                "\n\nIMPORTANT edge cases to handle: "
                'empty inputs ([], ""), negative numbers, zero values, '
                "single-element collections, very large numbers, "
                "boolean-is-int in Python (type(True)==int), "
                "floating-point precision (use abs(a-b)<epsilon). "
                "Return ONLY the complete function in a ```python block."
            )

    messages: list[Message] = [
        Message(role=Role.SYSTEM, content=system_prompt),
        Message(role=Role.USER, content=task_str),
    ]
    loop.working_memory.add_event("USER", task_str)
    # C4b: same tool-filter precedence as the affordance block above.
    tool_defs = loop._tools.get_tool_defs(tool_names)

    # Semantic memory context injection (one-time, before loop)
    if (
        loop.semantic_memory
        and not loop._cb_semantic.should_skip()
        and not loop._skip_memory
    ):
        try:
            sem_context = loop.semantic_memory.get_context_for(task_str)
            if sem_context and loop._relevance_gate.is_relevant(task_str, sem_context):
                messages.insert(
                    1,
                    Message(
                        role=Role.SYSTEM,
                        content=f"Relevant knowledge from previous interactions:\n{sem_context}",
                    ),
                )
            loop._cb_semantic.record_success()
        except Exception as e:
            loop._cb_semantic.record_failure(e)

    # S-MMU context injection (graph-based retrieval from compacted chunks)
    if not loop._cb_smmu.should_skip() and not loop._skip_memory:
        try:
            from sage.memory.smmu_context import retrieve_smmu_context

            smmu_context = retrieve_smmu_context(loop.working_memory)
            if smmu_context and loop._relevance_gate.is_relevant(task_str, smmu_context):
                messages.insert(
                    min(2, len(messages)),  # After system + semantic, before user
                    Message(role=Role.SYSTEM, content=smmu_context),
                )
            loop._cb_smmu.record_success()
        except Exception as e:
            loop._cb_smmu.record_failure(e)

    return _PerceiveResult(
        messages=messages,
        system_prompt=system_prompt,
        tool_defs=tool_defs,
    )
