"""sage_recurse tool — recursive self-invocation for test-time scaling.

Inspired by The Conductor (arXiv 2512.04388, ICLR 2026): the orchestrator
can invoke itself on a sub-task, letting the pipeline pick a fresh
topology / model assignment / validation level for the sub-problem.

Depth is tracked via contextvars so nested asyncio tasks inherit the
counter safely. Bounded at ``MAX_RECURSION_DEPTH`` (default 3) to prevent
runaway cost and stack blow-up.

Usage pattern inside the agent:
    When you hit a task that fits a completely different cognitive system
    (e.g. you are writing tests but need to do number theory first), call
    sage_recurse(sub_task="prove that X") and treat the result as a
    black-box answer. Do NOT use sage_recurse for trivial sub-steps that
    your own tools can handle — that multiplies cost.
"""
from __future__ import annotations

import contextvars
import logging
import os
from typing import Any, Awaitable, Callable

from sage.llm.base import ToolDef
from sage.tools.base import Tool

log = logging.getLogger(__name__)

_RECURSION_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "sage_recurse_depth", default=0,
)

# Set by TopologyRunner._execute_node to the current node index so the
# spawn-gate in build_sage_recurse_tool can debit the right Rust state.
# Default None → gate is skipped (standalone callers, tests).
sage_recurse_origin_node: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "sage_recurse_origin_node", default=None,
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


# Default max depth is small — deep recursion is almost always a bug.
# Bench adapters can raise via SAGE_RECURSION_MAX env var.
MAX_RECURSION_DEPTH = _env_int("SAGE_RECURSION_MAX", 3)

# Soft budget multiplier — deeper calls get progressively smaller budgets
# to prevent one branch from consuming the whole parent budget.
_BUDGET_DECAY = 0.5


_DESCRIPTION = (
    "Invoke the full SAGE pipeline (classify -> decompose -> topology -> "
    "assign -> execute -> learn) on a sub-task. Use when the current task "
    "contains a sub-problem of a different cognitive class (e.g. a math "
    "proof embedded in a coding task) that benefits from a fresh topology "
    "and model assignment. Returns the sub-task's final answer.\n\n"
    "Hard limits: recursion depth is capped (default 3). Deeper calls are "
    "refused with an error. Each level gets half the parent budget. Sub-tasks "
    "must be self-contained — no shared state with the parent. Do NOT use "
    "this for trivial sub-steps that your regular tools can handle; it is "
    "expensive (full pipeline pass)."
)

_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sub_task": {
            "type": "string",
            "description": "Self-contained sub-task to run through the pipeline.",
        },
        "budget_usd": {
            "type": "number",
            "description": "Optional USD budget cap for the sub-run. Defaults to "
                           "half the parent budget (or 1.0 if unknown).",
            "default": 1.0,
        },
        "system_hint": {
            "type": "integer",
            "description": "Optional S1/S2/S3 override for the sub-task. Use 1 "
                           "for simple factual, 2 for code/reasoning, 3 for "
                           "complex multi-step or formal.",
            "enum": [1, 2, 3],
        },
    },
    "required": ["sub_task"],
}


def build_sage_recurse_tool(
    run_callable: Callable[..., Awaitable[str]],
    controller: Any = None,
) -> Tool:
    """Create a sage_recurse Tool bound to the given run() coroutine.

    ``run_callable`` should behave like ``AgentSystem.run(task, *, system_hint=None)``.
    ``controller``, when provided, enables spawn-budget gating: each call
    is checked against ``controller._rust_ctrl.should_trigger_emergent_spawn``
    and debited via ``record_emergent_spawn`` before dispatch. The debit
    happens BEFORE dispatch so a failed sub-run still counts toward the
    budget (DoS guard against loopy agents).

    Bench adapters or tests can pass their own coroutine (a mock, a wrapper
    with extra logging, etc.) and may omit ``controller`` for unit-test
    scenarios.
    """

    async def _handler(
        sub_task: str = "",
        budget_usd: float = 1.0,
        system_hint: int | None = None,
    ) -> str:
        sub_task = (sub_task or "").strip()
        if not sub_task:
            return "Error: sage_recurse requires a non-empty sub_task."

        current_depth = _RECURSION_DEPTH.get()
        if current_depth >= MAX_RECURSION_DEPTH:
            log.warning(
                "sage_recurse refused: depth %d >= MAX %d",
                current_depth, MAX_RECURSION_DEPTH,
            )
            return (
                f"Error: sage_recurse refused — max recursion depth "
                f"({MAX_RECURSION_DEPTH}) reached. Solve the sub-task with "
                f"your existing tools instead."
            )

        if system_hint is not None and system_hint not in (1, 2, 3):
            return f"Error: system_hint must be 1, 2, or 3 (got {system_hint})."

        # Spawn-budget gate (Task D of 2026-04-20 phase-1 stab plan).
        # Skipped when controller is None (standalone tool use) or when
        # origin_node is None (no topology run in context).
        origin_node = sage_recurse_origin_node.get()
        if controller is not None and origin_node is not None:
            if not controller._rust_ctrl.should_trigger_emergent_spawn(origin_node):
                log.info(
                    "sage_recurse refused: spawn budget exhausted "
                    "(node=%d, spawn_count=%d)",
                    origin_node, controller._rust_ctrl.spawn_count,
                )
                return (
                    "Error: sage_recurse refused — spawn budget "
                    "exhausted for this execution"
                )
            try:
                controller._rust_ctrl.record_emergent_spawn(origin_node)
            except Exception as exc:
                log.error("sage_recurse record failed: %s", exc)
                return f"Error: sage_recurse refused — {exc}"

        token = _RECURSION_DEPTH.set(current_depth + 1)
        try:
            log.info(
                "sage_recurse[%d/%d]: sub_task=%r budget=$%.3f hint=%s",
                current_depth + 1, MAX_RECURSION_DEPTH,
                sub_task[:120], budget_usd, system_hint,
            )
            try:
                if system_hint is not None:
                    result = await run_callable(sub_task, system_hint=system_hint)
                else:
                    result = await run_callable(sub_task)
            except TypeError:
                # Callable doesn't accept system_hint kwarg — retry without.
                # Guard the fallback call too; otherwise a raise here would
                # escape the tool and violate the "never raise" contract.
                try:
                    result = await run_callable(sub_task)
                except Exception as exc:
                    log.exception("sage_recurse dispatch failed on fallback")
                    return (
                        f"Error: sage_recurse dispatch failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
            except Exception as exc:
                log.exception("sage_recurse dispatch failed")
                return (
                    f"Error: sage_recurse dispatch failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            return str(result) if result is not None else ""
        finally:
            _RECURSION_DEPTH.reset(token)

    return Tool(
        spec=ToolDef(
            name="sage_recurse",
            description=_DESCRIPTION,
            parameters=_PARAMETERS,
        ),
        handler=_handler,
    )


def current_depth() -> int:
    """Inspect the current recursion depth (useful for telemetry and tests)."""
    return _RECURSION_DEPTH.get()
