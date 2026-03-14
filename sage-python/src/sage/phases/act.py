"""ACT phase: S2 AVR validation, tool execution, S2->S3 escalation.

Extracted from agent_loop.py run() -- the S2 AVR block and tool execution.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from sage.llm.base import Message, Role

if TYPE_CHECKING:
    from sage.agent_loop import AgentLoop

log = logging.getLogger(__name__)


@dataclass
class _ActResult:
    """Return bundle from the act phase."""
    content: str = ""
    result_text: str = ""
    loop_action: str = "proceed"  # "proceed", "continue", "break"
    has_tool_calls: bool = False


async def _run_avr_sandbox(
    cleaned_code: str, content: str, messages: list[Message], loop: AgentLoop,
) -> _ActResult | None:
    """Run AVR sandbox execution. Returns _ActResult if loop action needed, else None."""
    from sage.agent_loop import LoopPhase, _is_stagnating, _shell_quote

    sandbox = await loop.sandbox_manager.create()
    try:
        result = await sandbox.execute(f"python3 -c {_shell_quote(cleaned_code)}")
        if result.exit_code != 0:
            stderr_full = (result.stderr or "").strip()
            stderr_last = stderr_full.split("\n")[-1][:200]
            stdout_snippet = (result.stdout or "").strip()[:200]
            runtime_err = f"RuntimeError (exit {result.exit_code}): {stderr_last}"
            loop._s2_avr_retries += 1
            loop._avr_error_history.append(runtime_err)

            if _is_stagnating(loop._avr_error_history, window=3):
                log.warning("S2 AVR stagnation detected (same runtime error %d times), "
                            "forcing escalation.", len(loop._avr_error_history))
                loop._s2_avr_retries = loop._max_s2_avr_retries + 1
            else:
                budget_left = loop._max_s2_avr_retries - loop._s2_avr_retries + 1
                loop._emit(LoopPhase.ACT, validation="s2_avr_fail",
                           avr_iteration=loop._s2_avr_retries,
                           avr_budget_left=budget_left,
                           error_type="runtime", error=runtime_err)
                if loop._s2_avr_retries <= loop._max_s2_avr_retries:
                    log.info("S2 AVR runtime fail (iteration %d/%d): %s",
                             loop._s2_avr_retries, loop._max_s2_avr_retries, runtime_err)
                    feedback_parts = [
                        f"SYSTEM [AVR {loop._s2_avr_retries}/{loop._max_s2_avr_retries}]: "
                        "Code execution failed.",
                    ]
                    if stderr_full:
                        feedback_parts.append(f"Traceback:\n```\n{stderr_full[:500]}\n```")
                    if stdout_snippet:
                        feedback_parts.append(f"Stdout: {stdout_snippet}")
                    feedback_parts.append(
                        "Analyze the error, fix the bug, and return ONLY corrected "
                        "Python code in a ```python fenced block.")
                    messages.append(Message(role=Role.USER,
                                           content="\n".join(feedback_parts)))
                    return _ActResult(content=content, loop_action="continue")
        else:
            loop._emit(LoopPhase.ACT, validation="s2_avr_pass",
                       stdout=result.stdout[:200])
            loop._s2_avr_retries = 0
            loop._avr_error_history.clear()
    finally:
        await loop.sandbox_manager.destroy(sandbox.id)
    return None


async def act(
    task: str, content: str, response: Any, brake: bool,
    messages: list[Message], loop: AgentLoop,
) -> _ActResult:
    """Execute the ACT phase of the agent loop.

    Handles S2 AVR validation (syntax check, sandbox execution, runtime
    guardrails), S2->S3 escalation, CGRS self-brake, episodic/semantic
    memory storage, tool call execution, and message trimming.
    """
    from sage.agent_loop import (
        LoopPhase, MAX_MESSAGES, _extract_code_blocks, _strip_markdown_fences,
        _validate_code_syntax, _is_stagnating,
    )

    # System 2 validation (Empirical -- AVR: Act-Verify-Refine)
    if loop.config.validation_level == 2 and content and not loop._skip_avr:
        code_blocks = _extract_code_blocks(content)

        if code_blocks and loop.sandbox_manager:
            raw_code = code_blocks[-1]
            cleaned_code = _strip_markdown_fences(raw_code)

            # Prefer Rust ToolExecutor (tree-sitter AST validator) if available
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
                except Exception as e:
                    log.warning("ToolExecutor failed, falling back to Python: %s", e)

            if not _te_rejected:
                syntax_ok, syntax_err = _validate_code_syntax(cleaned_code)

            if not syntax_ok:
                loop._s2_avr_retries += 1
                loop._avr_error_history.append(syntax_err)

                if _is_stagnating(loop._avr_error_history, window=3):
                    log.warning("S2 AVR stagnation detected (same error %d times), "
                                "forcing escalation.", len(loop._avr_error_history))
                    loop._s2_avr_retries = loop._max_s2_avr_retries + 1
                else:
                    budget_left = loop._max_s2_avr_retries - loop._s2_avr_retries + 1
                    loop._emit(LoopPhase.ACT, validation="s2_avr_fail",
                               avr_iteration=loop._s2_avr_retries,
                               avr_budget_left=budget_left,
                               error_type="syntax", error=syntax_err)
                    if loop._s2_avr_retries <= loop._max_s2_avr_retries:
                        log.info("S2 AVR syntax fail (iteration %d/%d): %s",
                                 loop._s2_avr_retries, loop._max_s2_avr_retries, syntax_err)
                        messages.append(Message(role=Role.USER, content=(
                            f"SYSTEM [AVR {loop._s2_avr_retries}/{loop._max_s2_avr_retries}]: "
                            f"Syntax error in your code:\n```\n{syntax_err}\n```\n"
                            f"Fix the syntax error and return ONLY corrected Python code "
                            f"in a ```python fenced block.")))
                        return _ActResult(content=content, loop_action="continue")
            else:
                # Syntax valid -- runtime guardrail + sandbox
                if (loop.guardrail_pipeline
                        and not loop._cb_runtime_guard.should_skip()
                        and not loop._skip_guardrails):
                    try:
                        runtime_results = await loop.guardrail_pipeline.check_all(
                            input=cleaned_code,
                            context={"step": loop.step_count, "phase": "runtime"})
                        for r in runtime_results:
                            loop._emit(LoopPhase.ACT, guardrail="runtime",
                                       guardrail_passed=r.passed,
                                       guardrail_reason=r.reason)
                        loop._cb_runtime_guard.record_success()
                    except Exception as e:
                        loop._cb_runtime_guard.record_failure(e)

                avr_result = await _run_avr_sandbox(cleaned_code, content, messages, loop)
                if avr_result is not None:
                    return avr_result

        elif not code_blocks and loop.step_count == 1:
            has_reasoning = "<think>" in content or "\n1." in content or "\n- " in content
            if not has_reasoning:
                loop._s2_avr_retries += 1
                if loop._s2_avr_retries <= loop._max_s2_avr_retries:
                    log.info("S2 validation: missing reasoning, requesting CoT.")
                    messages.append(Message(role=Role.USER,
                                           content="SYSTEM: Provide step-by-step reasoning for this task."))
                    return _ActResult(content=content, loop_action="continue")

        # S2 -> S3 escalation if max retries exhausted
        if (loop._s2_avr_retries > loop._max_s2_avr_retries
                and loop.config.validation_level == 2 and not loop._s3_degraded):
            log.info("S2 AVR exhausted -- escalating to S3 (formal verification).")
            loop.config.validation_level = 3
            loop._s3_retries = 0
            loop._avr_error_history.clear()
            loop._emit(LoopPhase.THINK, escalation="s2_to_s3",
                       reason="AVR budget exhausted")
            escalation_msg = (
                "SYSTEM: Escalating to formal verification. Use <think> tags "
                "with Z3 assertions (assert bounds, assert loop, assert arithmetic, "
                "assert invariant) for rigorous step-by-step reasoning.")
            inv_feedback = getattr(loop.prm.kg, "_last_invariant_feedback", [])
            if inv_feedback:
                escalation_msg += ("\n\nPrevious invariant verification failures:\n"
                                   + "\n".join(f"- {f}" for f in inv_feedback))
            messages.append(Message(role=Role.USER, content=escalation_msg))
            return _ActResult(content=content, loop_action="continue")

    # CGRS: stop if converged
    if brake:
        log.info("CGRS self-brake triggered -- stopping reasoning loop.")
        loop.working_memory.add_event("ASSISTANT", content)
        messages.append(Message(role=Role.ASSISTANT, content=content))
        return _ActResult(content=content, result_text=content, loop_action="break")

    loop.working_memory.add_event("ASSISTANT", content)

    # Store significant responses in episodic memory (if wired)
    if (loop.episodic_memory and len(content) > 100
            and not loop._cb_episodic.should_skip() and not loop._skip_memory):
        try:
            await loop.episodic_memory.store(
                key=f"step-{loop.step_count}", content=content[:500],
                metadata={"task": task, "step": loop.step_count})
            loop._cb_episodic.record_success()
        except Exception as e:
            loop._cb_episodic.record_failure(e)

    # Semantic memory: extract entities from response
    if (loop.memory_agent and loop.semantic_memory and content and len(content) > 50
            and not loop._cb_entity.should_skip() and not loop._skip_memory):
        try:
            extraction = await loop.memory_agent.extract(content[:1000])
            if extraction.entities:
                loop.semantic_memory.add_extraction(extraction)
            loop._cb_entity.record_success()
        except Exception as e:
            loop._cb_entity.record_failure(e)

    # No tool calls -> final answer
    if not response.tool_calls:
        messages.append(Message(role=Role.ASSISTANT, content=content))
        return _ActResult(content=content, result_text=content,
                          loop_action="break", has_tool_calls=False)

    # === Execute tools ===
    messages.append(Message(role=Role.ASSISTANT, content=content))
    for tc in response.tool_calls:
        loop._emit(LoopPhase.ACT, tool=tc.name, args=tc.arguments)
        output = await loop._execute_tool_call(tc)
        loop.working_memory.add_event("TOOL", f"{tc.name} -> {output}")
        messages.append(Message(role=Role.TOOL, content=output,
                                tool_call_id=tc.id, name=tc.name))

    # Trim messages to prevent unbounded growth
    if len(messages) > MAX_MESSAGES:
        messages[:] = messages[:2] + messages[-(MAX_MESSAGES - 2):]

    return _ActResult(content=content, loop_action="proceed", has_tool_calls=True)
