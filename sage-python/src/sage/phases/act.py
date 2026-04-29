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


def _memory_write_gate_skip_reason(
    *,
    gate_available: bool,
    content_len: int,
    has_tool_calls: bool,
    episodic_wired: bool,
    semantic_wired: bool,
    memory_agent_wired: bool,
    episodic_content_ok: bool,
    semantic_content_ok: bool,
) -> str | None:
    """Classify why write-gate telemetry did not cover all memory paths."""
    if not gate_available and not (episodic_content_ok or semantic_content_ok):
        return "gate_unavailable"
    if not episodic_wired and (not semantic_wired or not memory_agent_wired):
        return "memory_backend_unwired"
    if not episodic_wired:
        return "episodic_backend_unwired"
    if not semantic_wired or not memory_agent_wired:
        return "semantic_backend_unwired"
    if content_len == 0 and has_tool_calls:
        return "tool_only_empty_content"
    if content_len > 0 and not episodic_content_ok and not semantic_content_ok:
        return "content_too_short"
    return None


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
        # A8 Phase 2 (2026-04-24): propagate thinking onto the Message.
        messages.append(
            Message(
                role=Role.ASSISTANT,
                content=content,
                thinking=getattr(response, "thinking", "") or "",
            )
        )
        return _ActResult(content=content, result_text=content, loop_action="break")

    loop.working_memory.add_event("ASSISTANT", content)

    # G-series audit fix (2026-04-19): evaluate the 5-signal composite write
    # gate before each persistent memory write. Gate is shared across nodes
    # of one task so cross-node duplicate sentinels/empties hit exact-dedup.
    # When write_gate is None (legacy/direct callers), we fall back to the
    # old "always allow" behavior so existing tests and offline use keep
    # working.
    gate = loop.write_gate
    gate_task = loop.gate_current_task or task
    gate_tier = loop.gate_source_tier

    def _gate_allows(payload: str) -> bool:
        """Return True if the gate allows the write (or no gate is wired)."""
        if gate is None:
            return True
        try:
            decision = gate.evaluate(
                payload,
                1.0,  # No per-turn confidence signal; w_confidence=0 in config
                task=gate_task,
                source_tier=gate_tier,
                embedding=None,
            )
            # In-run observability: one structured line per gate.evaluate()
            # call so smoke runs can attribute memory writes to pillar
            # behavior. Shared helper ensures the format is stable across all
            # call sites (phases/act.py, tests, future bypass paths).
            try:
                from sage.memory.write_gate import log_write_gate_decision
                log_write_gate_decision(decision, source_tier=gate_tier)
            except Exception:
                pass  # never let logging break the write path
            allowed = bool(getattr(decision, "allowed", True))
            if not allowed:
                log.debug("write_gate blocked memory write: %s",
                          getattr(decision, "reason", "<no reason>"))
            return allowed
        except Exception as exc:
            log.debug("write_gate evaluate raised, allowing write: %s", exc)
            return True

    # Compose a "turn signal" that reflects activity in either prose OR tool
    # use. On tool-heavy turns (SWE-bench, BCB exploration steps), `content`
    # is often short (e.g. "I'll check the file."), with all the work in
    # tool_calls + tool_results. Gating purely on `len(content)` made the
    # write_gate never fire for those turns (Gap 5 diagnosis: Agent #51
    # `0bcb92b`), and the episodic path was skipped entirely — the agent
    # explored, found the issue, and the trace vanished after the task.
    # Fix: append a compact tool-activity marker so the gate can evaluate
    # tool-heavy turns and the episodic store keeps them. Semantic entity
    # extraction still requires real prose (entities from "[tools: bash]"
    # are meaningless) so its threshold is unchanged.
    tool_names = [tc.name for tc in (response.tool_calls or [])]
    tool_call_count = len(response.tool_calls or [])
    has_tool_calls = tool_call_count > 0
    tool_signal = f"[tools: {', '.join(tool_names)}]" if tool_names else ""
    turn_signal = f"{content}\n{tool_signal}".strip()

    # Gap 5 (2026-04-21): when content is too short for either memory path,
    # the gate is never evaluated and observability breaks (see v17 audit
    # docs/audits/2026-04-21-exocortex-swebench-usage.md). Emit an explicit
    # skip log so `grep memory.write_gate` post-run distinguishes "not wired"
    # from "wired but short content". Only fires when gate IS wired; no-op
    # for ungated legacy callers.
    content_len = len(content or "")
    episodic_wired = bool(loop.episodic_memory)
    semantic_wired = bool(loop.semantic_memory)
    memory_agent_wired = bool(loop.memory_agent)
    _episodic_content_ok = episodic_wired and (
        content_len > 100 or bool(tool_names)
    )
    _semantic_content_ok = bool(
        memory_agent_wired and semantic_wired and content and content_len > 50
    )
    skip_reason = _memory_write_gate_skip_reason(
        gate_available=gate is not None,
        content_len=content_len,
        has_tool_calls=has_tool_calls,
        episodic_wired=episodic_wired,
        semantic_wired=semantic_wired,
        memory_agent_wired=memory_agent_wired,
        episodic_content_ok=_episodic_content_ok,
        semantic_content_ok=_semantic_content_ok,
    )
    if skip_reason is not None:
        try:
            from sage.memory.write_gate import log_write_gate_skipped
            log_write_gate_skipped(
                reason=skip_reason,
                content_len=content_len,
                has_tool_calls=has_tool_calls,
                source_tier=gate_tier,
                tool_call_count=tool_call_count,
                episodic_wired=episodic_wired,
                semantic_wired=semantic_wired,
                memory_agent_wired=memory_agent_wired,
            )
        except Exception:
            pass  # never break the act path on logging

    # Store significant responses in episodic memory (if wired).
    # Tool-heavy turns (short prose, many tool_calls) also go through the gate
    # and the episodic path — store `turn_signal[:500]` so the tool-activity
    # marker is preserved for later retrieval and causal edge building.
    if (loop.episodic_memory and (len(content) > 100 or bool(tool_names))
            and not loop._cb_episodic.should_skip() and not loop._skip_memory):
        episodic_payload = turn_signal[:500] if tool_names else content[:500]
        if _gate_allows(episodic_payload):
            try:
                await loop.episodic_memory.store(
                    key=f"step-{loop.step_count}", content=episodic_payload,
                    metadata={
                        "task": task,
                        "step": loop.step_count,
                        "tool_activity": bool(tool_names),
                    })
                loop._cb_episodic.record_success()
            except Exception as e:
                loop._cb_episodic.record_failure(e)

    # Semantic memory: extract entities from response
    if (loop.memory_agent and loop.semantic_memory and content and len(content) > 50
            and not loop._cb_entity.should_skip() and not loop._skip_memory):
        semantic_payload = content[:1000]
        if _gate_allows(semantic_payload):
            try:
                extraction = await loop.memory_agent.extract(semantic_payload)
                if extraction.entities:
                    loop.semantic_memory.add_extraction(extraction)
                    # Causal edges: consecutive entities form causal chains
                    # AMA-Bench (2602.22769): memory without causality fails
                    if (loop.causal_memory
                            and len(extraction.entities) >= 2
                            and not loop._cb_causal.should_skip()):
                        try:
                            for i in range(len(extraction.entities) - 1):
                                src = extraction.entities[i]
                                tgt = extraction.entities[i + 1]
                                loop.causal_memory.add_entity(src)
                                loop.causal_memory.add_entity(tgt)
                                loop.causal_memory.add_causal_edge(
                                    src, tgt, cause_type="enabled",
                                )
                            loop._cb_causal.record_success()
                        except Exception as exc:
                            loop._cb_causal.record_failure(exc)
                loop._cb_entity.record_success()
            except Exception as e:
                loop._cb_entity.record_failure(e)

    # No tool calls -> final answer
    if not response.tool_calls:
        # A8 Phase 2 (2026-04-24): propagate thinking even on the
        # final turn — downstream consumers (tests, bench telemetry)
        # may still inspect it. Cheap (empty string on non-thinking).
        messages.append(
            Message(
                role=Role.ASSISTANT,
                content=content,
                thinking=getattr(response, "thinking", "") or "",
            )
        )
        return _ActResult(content=content, result_text=content,
                          loop_action="break", has_tool_calls=False)

    # === Execute tools ===
    # A8 Phase 2 (2026-04-24): propagate reasoning_content /
    # ThinkingPart onto the assistant Message so the next turn's
    # outgoing ModelRequest carries it back to the provider.
    # Moonshot (kimi-k2.5/k2.6) and DeepSeek (v4-pro) require every
    # prior assistant-with-tool-calls message in history to include
    # reasoning_content, or they reject the 4th+ tool-call turn with
    # HTTP 400. The PydanticAI wrapper in providers/pydantic_ai_provider
    # translates this Message.thinking → ThinkingPart → reasoning_content
    # automatically once the field is populated here.
    messages.append(
        Message(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=response.tool_calls or None,
            thinking=getattr(response, "thinking", "") or "",
        )
    )
    # Telemetry — counters previously declared on loop but never incremented.
    # Telemetry-only: gates no decision; surfaces to PipelineContext and bench
    # manifests so "agent never called tools" vs "tools were called but step
    # budget ran out" can be told apart.
    loop.tool_turn_count += 1
    loop.tool_call_count += len(response.tool_calls)
    for tc in response.tool_calls:
        loop._emit(LoopPhase.ACT, tool=tc.name, args=tc.arguments)
        output = await loop._execute_tool_call(tc)
        # Record every tool call as "<tool_name>: <first_120_char_arg>".
        # 2026-04-21 audit (docs/audits/2026-04-21-exocortex-swebench-usage.md)
        # showed the prior bash-only gate meant `_executed_commands` only
        # counted bash, so `tool_call_count - len(executed_commands) > 0`
        # was invisible — we couldn't tell "agent called search_exocortex
        # but didn't bash" from "agent called nothing". Now every tool
        # gets a forensic entry; bash keeps its command payload, others
        # get a short arg summary.
        arg_summary = ""
        try:
            import json as _json
            args = tc.arguments if isinstance(tc.arguments, dict) else _json.loads(tc.arguments or "{}")
            if tc.name == "execute_bash":
                arg_summary = str(args.get("command", ""))[:120]
            elif isinstance(args, dict) and args:
                # Pick the first string-valued arg as the summary
                first = next(
                    (str(v) for v in args.values() if isinstance(v, str) and v),
                    "",
                )
                arg_summary = first[:120]
        except (ValueError, TypeError, AttributeError):
            pass
        entry = f"{tc.name}: {arg_summary}" if arg_summary else tc.name
        loop.executed_commands.append(entry)
        loop.working_memory.add_event("TOOL", f"{tc.name} -> {output}")
        messages.append(Message(role=Role.TOOL, content=output,
                                tool_call_id=tc.id, name=tc.name))
        # Causal edge: tool invocation "triggered" its output entity
        if (loop.causal_memory
                and not loop._cb_causal.should_skip()
                and not loop._skip_memory):
            try:
                tool_entity = f"tool:{tc.name}"
                output_entity = f"result:{tc.name}:{loop.step_count}"
                loop.causal_memory.add_entity(tool_entity)
                loop.causal_memory.add_entity(output_entity)
                loop.causal_memory.add_causal_edge(
                    tool_entity, output_entity, cause_type="triggered",
                )
                loop._cb_causal.record_success()
            except Exception as exc:
                loop._cb_causal.record_failure(exc)

    # Trim messages to prevent unbounded growth (orphan-tool aware).
    if len(messages) > MAX_MESSAGES:
        messages[:] = _truncate_messages_orphan_safe(messages, MAX_MESSAGES)

    return _ActResult(content=content, loop_action="proceed", has_tool_calls=True)


def _truncate_messages_orphan_safe(
    messages: list[Message], max_msgs: int,
) -> list[Message]:
    """Truncate keeping system+user head and recent tail, dropping orphan tools.

    The naive split (`messages[:2] + messages[-(N-2):]`) drops any
    assistant carrying tool_calls in the middle but keeps their
    tool_result in the tail — producing an orphan: a `role: "tool"`
    message whose `tool_call_id` matches NO surviving assistant.

    Lenient providers (OpenAI, Gemini) accept it. Strict providers
    (MiniMax-m2.7 since late 2025) reject the request:
        400 invalid params, tool result's tool id(call_X) not found (2013)
    Surfaced 2026-04-17 SWE-bench smoke after F7 floor started routing
    nodes onto MiniMax-tier models.

    Fix: after the head/tail split, drop any tool message whose
    tool_call_id has no surviving assistant counterpart in the kept
    window.
    """
    if len(messages) <= max_msgs:
        return list(messages)
    head = messages[:2]
    tail = messages[-(max_msgs - 2):]
    surviving_call_ids: set[str] = set()
    for m in head + tail:
        if m.role == Role.ASSISTANT and m.tool_calls:
            for tc in m.tool_calls:
                if tc.id:
                    surviving_call_ids.add(tc.id)
    cleaned_tail = [
        m for m in tail
        if not (
            m.role == Role.TOOL
            and m.tool_call_id
            and m.tool_call_id not in surviving_call_ids
        )
    ]
    return head + cleaned_tail
