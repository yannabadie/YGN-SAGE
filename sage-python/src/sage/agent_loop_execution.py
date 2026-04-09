"""Tool execution helpers for AgentLoop.

Extracted from agent_loop.py _run_legacy() to reduce file size.
These are standalone functions that take the tool registry and sandbox
as parameters instead of relying on self.X attribute access.
"""
from __future__ import annotations

import logging
from typing import Any

from sage.llm.base import Message, Role

log = logging.getLogger(__name__)


async def execute_tool_call(
    tc: Any,
    tool_registry: Any,
    emit_fn: Any,
) -> str:
    """Execute a single tool call with argument validation.

    Parameters
    ----------
    tc : ToolCall
        The tool call object (has .name, .arguments, .id).
    tool_registry : ToolRegistry
        Registry to look up the tool by name.
    emit_fn : callable
        Callable to emit LoopPhase.ACT events (for TOOL_GAP detection).

    Returns
    -------
    str
        The tool output string, or an error message.
    """
    from sage.agent_loop import LoopPhase

    tool = tool_registry.get(tc.name)
    if tool is None:
        # Emit TOOL_GAP for ToolForge gap detection
        emit_fn(
            LoopPhase.ACT,
            tool_gap=True,
            tool_name=tc.name,
            tool_args=tc.arguments,
        )
        return f"Error: Unknown tool '{tc.name}'"
    kwargs = tc.arguments
    if not isinstance(kwargs, dict):
        log.warning("Tool '%s' received non-dict arguments: %s", tc.name, type(kwargs))
        return (
            f"Error: Tool '{tc.name}' received invalid arguments "
            f"(expected dict, got {type(kwargs).__name__})"
        )
    try:
        result = await tool.execute(kwargs.copy())
        return result.output
    except (RuntimeError, ValueError, TimeoutError) as e:
        log.error("Tool '%s' execution failed: %s", tc.name, e)
        return f"Error executing tool '{tc.name}': {type(e).__name__}: {e}"


async def execute_tool_calls_and_record(
    response: Any,
    content: str,
    messages: list[Message],
    tool_registry: Any,
    working_memory: Any,
    causal_memory: Any,
    cb_causal: Any,
    skip_memory: bool,
    step_count: int,
    emit_fn: Any,
    max_messages: int,
) -> None:
    """Execute all tool calls from a response, record in memory, trim messages.

    This handles:
    - Appending the assistant message
    - Executing each tool call and appending TOOL messages
    - Recording causal edges (tool -> output) in causal memory
    - Trimming messages to max_messages

    Mutates *messages* in place.
    """
    from sage.agent_loop import LoopPhase

    messages.append(
        Message(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=response.tool_calls or None,
        )
    )

    for tc in response.tool_calls:
        emit_fn(LoopPhase.ACT, tool=tc.name, args=tc.arguments)
        output = await execute_tool_call(tc, tool_registry, emit_fn)
        working_memory.add_event("TOOL", f"{tc.name} -> {output}")
        messages.append(Message(
            role=Role.TOOL, content=output,
            tool_call_id=tc.id, name=tc.name,
        ))
        # Causal edge: tool invocation "triggered" its output entity
        if causal_memory and not cb_causal.should_skip() and not skip_memory:
            try:
                tool_entity = f"tool:{tc.name}"
                output_entity = f"result:{tc.name}:{step_count}"
                causal_memory.add_entity(tool_entity)
                causal_memory.add_entity(output_entity)
                causal_memory.add_causal_edge(tool_entity, output_entity, cause_type="triggered")
                cb_causal.record_success()
            except (RuntimeError, AttributeError) as exc:
                cb_causal.record_failure(exc)

    if len(messages) > max_messages:
        messages[:] = messages[:2] + messages[-(max_messages - 2):]


def schedule_from_topology(current_topology: Any) -> list[dict]:
    """Use Rust TopologyExecutor to get node execution order.

    Returns list of node specs for topology-aware execution.
    Falls back to empty list if executor unavailable.
    """
    if not current_topology:
        return []
    try:
        from sage_core import TopologyExecutor as PyTopologyExecutor  # noqa: E402
        executor = PyTopologyExecutor(current_topology)
        schedule: list[dict] = []
        while not executor.is_done():
            ready = executor.next_ready(current_topology)
            if not ready:
                break
            for idx in ready:
                node = current_topology.get_node(idx)
                schedule.append({
                    "index": idx,
                    "role": node.role,
                    "model_id": node.model_id,
                    "system": node.system,
                })
                executor.mark_completed(idx)
        return schedule
    except (ImportError, RuntimeError):
        return []


async def run_topology(
    task: str,
    current_topology: Any,
    llm_provider: Any,
    llm_config: Any,
    emit_fn: Any,
) -> str | None:
    """Execute multi-node topology via TopologyRunner.

    Returns the result string if topology has >1 node, or None to fall
    through to standard single-LLM execution.
    """
    from sage.agent_loop import LoopPhase

    if not current_topology:
        return None

    schedule = schedule_from_topology(current_topology)
    if len(schedule) <= 1:
        return None  # Single node or empty -- use standard path

    try:
        from sage.topology.runner import TopologyRunner
        from sage_core import TopologyExecutor as PyTopologyExecutor  # noqa: E402
        executor = PyTopologyExecutor(current_topology)
        runner = TopologyRunner(
            graph=current_topology,
            executor=executor,
            llm_provider=llm_provider,
            llm_config=llm_config,
        )
        result = await runner.run(task)
        emit_fn(
            LoopPhase.THINK,
            topology_execution="multi_agent",
            node_count=len(schedule),
        )
        return result
    except (ImportError, RuntimeError, TimeoutError) as e:
        log.warning("TopologyRunner failed (%s), falling back to single-LLM", e)
        return None


async def legacy_think_step(
    messages: list[Message],
    tool_defs: list,
    loop: Any,
) -> tuple[Any, str, bool]:
    """LLM call + cost estimation + entropy + MEM1 for legacy loop.

    Returns (response, content, brake).
    """
    from sage.agent_loop import (
        LoopPhase, _estimate_tokens, _text_entropy, _load_cost_table,
        _COST_PER_1K,
    )
    from sage.constants import DEFAULT_COST_PER_1K

    model_name = loop.config.llm.model
    loop._emit(LoopPhase.THINK, model=model_name)

    import time
    t0 = time.perf_counter()
    response = await loop._llm.generate(
        messages=messages,
        tools=tool_defs if tool_defs else None,
        config=loop.config.llm,
    )
    inference_ms = (time.perf_counter() - t0) * 1000
    loop.total_inference_time += inference_ms / 1000

    content = response.content or ""

    usage = getattr(response, "usage", None) or {}
    actual_total = usage.get("total_tokens") if isinstance(usage, dict) else None
    tokens = _estimate_tokens(content, actual_count=actual_total)
    _load_cost_table()
    cost_per_k = _COST_PER_1K.get(model_name, DEFAULT_COST_PER_1K)
    step_cost = (tokens / 1000) * cost_per_k
    loop.total_cost_usd += step_cost

    entropy = _text_entropy(content)
    brake = False
    if loop.metacognition:
        loop.metacognition.record_output_entropy(entropy)
        brake = loop.metacognition.should_brake()

    loop._emit(
        LoopPhase.THINK,
        model=model_name,
        content=content,
        latency_ms=round(inference_ms, 1),
        cost_usd=round(loop.total_cost_usd, 4),
        entropy=round(entropy, 3),
        brake=brake,
    )

    if loop.memory_compressor and content:
        await loop.memory_compressor.generate_internal_state(
            f"[Step {loop.step_count}] {content[:300]}"
        )

    return response, content, brake


async def cegar_repair(
    content: str,
    prm_details: Any,
    invariant_feedback: list[str],
    system_prompt: str,
    llm_provider: Any,
    llm_config: Any,
    prm: Any,
) -> str | None:
    """Attempt CEGAR repair of failed S3 verification.

    Extracts failed clauses from PRM details and invariant feedback,
    builds a targeted repair prompt, and makes a single LLM call.
    Returns repaired content if PRM passes, or None if repair fails.
    """
    repair_prompt = (
        "SYSTEM: Your formal verification FAILED. "
        "Do NOT regenerate from scratch -- fix the specific failures below.\n\n"
        f"Verification error: {prm_details}\n"
    )
    if invariant_feedback:
        repair_prompt += (
            "\nFailed invariant clauses:\n"
            + "\n".join(f"- {f}" for f in invariant_feedback)
            + "\n"
        )
    repair_prompt += (
        "\nFix your reasoning by adding the missing formal assertions. "
        "Use <think> tags with Z3 assertions for each step."
    )

    messages = [
        Message(role=Role.SYSTEM, content=system_prompt),
        Message(role=Role.ASSISTANT, content=content),
        Message(role=Role.USER, content=repair_prompt),
    ]

    try:
        response = await llm_provider.generate(
            messages=messages,
            config=llm_config,
        )
        repaired = response.content or ""
        if not repaired:
            return None

        # Verify the repair
        r_path, details = prm.calculate_r_path(repaired)
        if r_path >= 0.0 or "error" not in details:
            log.info("CEGAR repair succeeded: r_path=%.2f", r_path)
            return repaired
        else:
            log.warning("CEGAR repair failed: %s", details)
            return None
    except (RuntimeError, TimeoutError) as e:
        log.warning("CEGAR repair LLM call failed: %s", e)
        return None


async def run_legacy_s3(
    content: str,
    messages: list[Message],
    loop: Any,
) -> tuple[str, str]:
    """Run S3 (Z3 PRM) validation within _run_legacy.

    Returns (action, content) where action is 'continue' or 'proceed',
    and content may be updated by CEGAR repair.
    """
    from sage.agent_loop import LoopPhase

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
            return "continue", content

        inv_feedback = getattr(loop.prm.kg, "_last_invariant_feedback", [])
        repaired = await loop._cegar_repair(content, details, inv_feedback)
        if repaired is not None:
            content = repaired
            loop._s3_retries = 0
        else:
            log.warning(
                "S3 verification failed after CEGAR repair -- "
                "degrading to S2 AVR."
            )
            loop._emit(
                LoopPhase.THINK,
                s3_degradation=True,
                reason="CEGAR repair failed",
            )
            loop.config.validation_level = 2
            loop._s3_degraded = True
            loop._s3_retries = 0
            loop._s2_avr_retries = 0
            return "continue", content
    else:
        loop._s3_retries = 0

    return "proceed", content


async def run_legacy_avr(
    content: str,
    messages: list[Message],
    loop: Any,
) -> str:
    """Run S2 AVR validation within _run_legacy.

    Returns 'continue' if the caller should re-enter the loop,
    or 'proceed' to continue to the next phase.

    Parameters
    ----------
    content : str
        The LLM response content to validate.
    messages : list[Message]
        The message list (mutated in place with feedback messages).
    loop : AgentLoop
        The agent loop instance (accesses sandbox_manager, tool_executor,
        guardrail_pipeline, prm, and various counters).
    """
    from sage.agent_loop import (
        LoopPhase, _extract_code_blocks, _strip_markdown_fences,
        _validate_code_syntax, _is_stagnating, _shell_quote,
    )

    code_blocks = _extract_code_blocks(content)

    if code_blocks and loop.sandbox_manager:
        raw_code = code_blocks[-1]
        cleaned_code = _strip_markdown_fences(raw_code)

        _te_rejected = False
        if loop.tool_executor:
            try:
                te_result = loop.tool_executor.validate(cleaned_code)
                if not te_result.valid:
                    te_err = "; ".join(te_result.errors)
                    log.warning("ToolExecutor rejected code: %s", te_err)
                    _te_rejected = True
                    syntax_ok, syntax_err = False, te_err
                else:
                    log.debug("ToolExecutor validated code successfully")
            except (ImportError, RuntimeError) as e:
                log.warning("ToolExecutor failed, falling back to Python: %s", e)

        if not _te_rejected:
            syntax_ok, syntax_err = _validate_code_syntax(cleaned_code)

        if not syntax_ok:
            loop._s2_avr_retries += 1
            loop._avr_error_history.append(syntax_err)

            if _is_stagnating(loop._avr_error_history, window=3):
                log.warning("S2 AVR stagnation detected (same error %d times), forcing escalation.",
                            len(loop._avr_error_history))
                loop._s2_avr_retries = loop._max_s2_avr_retries + 1
            else:
                budget_left = loop._max_s2_avr_retries - loop._s2_avr_retries + 1
                loop._emit(LoopPhase.ACT,
                           validation="s2_avr_fail",
                           avr_iteration=loop._s2_avr_retries,
                           avr_budget_left=budget_left,
                           error_type="syntax",
                           error=syntax_err)
                if loop._s2_avr_retries <= loop._max_s2_avr_retries:
                    log.info("S2 AVR syntax fail (iteration %d/%d): %s",
                             loop._s2_avr_retries, loop._max_s2_avr_retries, syntax_err)
                    messages.append(Message(
                        role=Role.USER,
                        content=(
                            f"SYSTEM [AVR {loop._s2_avr_retries}/{loop._max_s2_avr_retries}]: "
                            f"Syntax error in your code:\n```\n{syntax_err}\n```\n"
                            f"Fix the syntax error and return ONLY corrected Python code "
                            f"in a ```python fenced block."
                        ),
                    ))
                    return "continue"
        else:
            if loop.guardrail_pipeline and not loop._cb_runtime_guard.should_skip() and not loop._skip_guardrails:
                try:
                    runtime_results = await loop.guardrail_pipeline.check_all(
                        input=cleaned_code,
                        context={"step": loop.step_count, "phase": "runtime"}
                    )
                    for r in runtime_results:
                        loop._emit(LoopPhase.ACT,
                                   guardrail="runtime",
                                   guardrail_passed=r.passed,
                                   guardrail_reason=r.reason)
                    loop._cb_runtime_guard.record_success()
                except (RuntimeError, ValueError, TimeoutError) as e:
                    loop._cb_runtime_guard.record_failure(e)

            sandbox = await loop.sandbox_manager.create()
            try:
                result = await sandbox.execute(
                    f"python3 -c {_shell_quote(cleaned_code)}"
                )
                if result.exit_code != 0:
                    stderr_full = (result.stderr or "").strip()
                    stderr_last = stderr_full.split("\n")[-1][:200]
                    stdout_snippet = (result.stdout or "").strip()[:200]
                    runtime_err = f"RuntimeError (exit {result.exit_code}): {stderr_last}"
                    loop._s2_avr_retries += 1
                    loop._avr_error_history.append(runtime_err)

                    if _is_stagnating(loop._avr_error_history, window=3):
                        log.warning("S2 AVR stagnation detected (same runtime error %d times), forcing escalation.",
                                    len(loop._avr_error_history))
                        loop._s2_avr_retries = loop._max_s2_avr_retries + 1
                    else:
                        budget_left = loop._max_s2_avr_retries - loop._s2_avr_retries + 1
                        loop._emit(LoopPhase.ACT,
                                   validation="s2_avr_fail",
                                   avr_iteration=loop._s2_avr_retries,
                                   avr_budget_left=budget_left,
                                   error_type="runtime",
                                   error=runtime_err)
                        if loop._s2_avr_retries <= loop._max_s2_avr_retries:
                            log.info("S2 AVR runtime fail (iteration %d/%d): %s",
                                     loop._s2_avr_retries, loop._max_s2_avr_retries, runtime_err)
                            feedback_parts = [
                                f"SYSTEM [AVR {loop._s2_avr_retries}/{loop._max_s2_avr_retries}]: Code execution failed.",
                            ]
                            if stderr_full:
                                feedback_parts.append(f"Traceback:\n```\n{stderr_full[:500]}\n```")
                            if stdout_snippet:
                                feedback_parts.append(f"Stdout: {stdout_snippet}")
                            feedback_parts.append(
                                "Analyze the error, fix the bug, and return ONLY corrected Python code "
                                "in a ```python fenced block."
                            )
                            messages.append(Message(
                                role=Role.USER,
                                content="\n".join(feedback_parts),
                            ))
                            return "continue"
                else:
                    loop._emit(LoopPhase.ACT,
                               validation="s2_avr_pass",
                               stdout=result.stdout[:200])
                    loop._s2_avr_retries = 0
                    loop._avr_error_history.clear()
            finally:
                await loop.sandbox_manager.destroy(sandbox.id)

    elif not code_blocks and loop.step_count == 1:
        has_reasoning = "<think>" in content or "\n1." in content or "\n- " in content
        if not has_reasoning:
            loop._s2_avr_retries += 1
            if loop._s2_avr_retries <= loop._max_s2_avr_retries:
                log.info("S2 validation: missing reasoning, requesting CoT.")
                messages.append(Message(
                    role=Role.USER,
                    content="SYSTEM: Provide step-by-step reasoning for this task.",
                ))
                return "continue"

    # S2 -> S3 escalation
    if loop._s2_avr_retries > loop._max_s2_avr_retries and loop.config.validation_level == 2 and not loop._s3_degraded:
        log.info("S2 AVR exhausted -- escalating to S3 (formal verification).")
        loop.config.validation_level = 3
        loop._s3_retries = 0
        loop._avr_error_history.clear()
        loop._emit(LoopPhase.THINK, escalation="s2_to_s3",
                   reason="AVR budget exhausted")
        escalation_msg = (
            "SYSTEM: Escalating to formal verification. Use <think> tags "
            "with Z3 assertions (assert bounds, assert loop, assert arithmetic, "
            "assert invariant) for rigorous step-by-step reasoning."
        )
        inv_feedback = getattr(loop.prm.kg, "_last_invariant_feedback", [])
        if inv_feedback:
            escalation_msg += (
                "\n\nPrevious invariant verification failures:\n"
                + "\n".join(f"- {f}" for f in inv_feedback)
            )
        messages.append(Message(
            role=Role.USER,
            content=escalation_msg,
        ))
        return "continue"

    return "proceed"


async def store_episodic_and_entities(
    task: str,
    content: str,
    step_count: int,
    episodic_memory: Any,
    memory_agent: Any,
    semantic_memory: Any,
    causal_memory: Any,
    cb_episodic: Any,
    cb_entity: Any,
    cb_causal: Any,
    skip_memory: bool,
) -> None:
    """Store significant responses in episodic memory and extract entities.

    This handles:
    - Episodic memory storage for long responses
    - Entity extraction via memory_agent
    - Causal edge creation between consecutive extracted entities

    Called after working_memory.add_event("ASSISTANT", content) in the legacy loop.
    """
    # Episodic memory
    if episodic_memory and len(content) > 100 and not cb_episodic.should_skip() and not skip_memory:
        try:
            await episodic_memory.store(
                key=f"step-{step_count}",
                content=content[:500],
                metadata={"task": task, "step": step_count},
            )
            cb_episodic.record_success()
        except (RuntimeError, AttributeError) as e:
            cb_episodic.record_failure(e)

    # Semantic memory: extract entities from response
    if memory_agent and semantic_memory and content and len(content) > 50 and not cb_entity.should_skip() and not skip_memory:
        try:
            extraction = await memory_agent.extract(content[:1000])
            if extraction.entities:
                semantic_memory.add_extraction(extraction)
                # Feed causal memory: consecutive entities form causal edges
                # AMA-Bench (2602.22769): memory without causality fails
                if causal_memory and len(extraction.entities) >= 2 and not cb_causal.should_skip():
                    try:
                        for i in range(len(extraction.entities) - 1):
                            src, tgt = extraction.entities[i], extraction.entities[i + 1]
                            causal_memory.add_entity(src)
                            causal_memory.add_entity(tgt)
                            causal_memory.add_causal_edge(src, tgt, cause_type="enabled")
                        cb_causal.record_success()
                    except (RuntimeError, AttributeError) as exc:
                        cb_causal.record_failure(exc)
            cb_entity.record_success()
        except (RuntimeError, AttributeError) as e:
            cb_entity.record_failure(e)
