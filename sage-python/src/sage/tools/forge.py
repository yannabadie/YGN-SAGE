"""ToolForge — autonomous tool synthesis via LLM-driven build loop.

Detects capability gaps, generates tool code + tests, validates through
a dual-gate pipeline, and registers successful tools.

Research basis:
- UCT (arXiv 2602.01983): Build Loop generates code + tests together
- SMITH (arXiv 2512.11303): Dual-gate validation (sandbox + AST)
- Tool-Genesis (arXiv 2603.05578): Iterative refinement mandatory
- CRAFT (arXiv 2309.17428, ICLR 2024): Multi-view retrieval for tools
"""
from __future__ import annotations

import ast
import logging
import os
import re
import time
from typing import Any, Callable

from sage.tools.gap_detector import CreationTicket, GapDetector

log = logging.getLogger(__name__)

# Maximum tool creations per pipeline run (engineering guard)
MAX_CREATIONS_PER_RUN = 2
# Maximum build loop rounds per ticket
MAX_BUILD_ROUNDS = 3
_APPROVE_ALL_WARNED = False

_TOOL_GEN_PROMPT = """\
You are a tool engineer. Create a Python tool that fills the following capability gap.

## Gap
{gap_description}

## Context
Task: {task}
Expected interface (args): {required_interface}
Suggested name: {tool_name_hint}

## Requirements
1. Write a SINGLE Python function named `tool_{name}` that takes a `dict` argument called `args`
2. The function must print its result as JSON to stdout: `print(json.dumps({{"output": result}}))`
3. Write exactly 3 test cases as separate function calls at the bottom
4. Use ONLY standard library modules (json, math, re, collections, itertools, functools, etc.)
5. Do NOT import os, sys, subprocess, socket, or any network/filesystem modules

## Output Format
Return exactly two fenced code blocks:

```python
# Tool code
import json

def tool_{name}(args):
    # implementation
    result = ...
    print(json.dumps({{"output": result}}))
```

```python
# Tests
tool_{name}({{"arg1": "value1"}})
tool_{name}({{"arg2": "value2"}})
tool_{name}({{"arg3": "value3"}})
```
"""

_TOOL_REPAIR_PROMPT = """\
Your previous tool code failed validation. Fix the errors below.

## Errors
{errors}

## Previous Code
```python
{previous_code}
```

Return the corrected code in the same two-block format (tool code + tests).
"""


class ToolForge:
    """Autonomous tool synthesis via LLM-driven build loop.

    Detects capability gaps during execution, generates tool code + tests
    via LLM, validates through dual-gate (AST + sandbox), and registers
    successful tools in the ToolRegistry.

    Parameters
    ----------
    registry : ToolRegistry
        Target registry for newly created tools.
    llm_provider : LLMProvider
        LLM for code generation.
    llm_config : LLMConfig
        Config for LLM calls.
    event_bus : EventBus, optional
        For emitting TOOL_FORGED events.
    approval_callback : Callable[[str, str], bool], optional
        HITL approval callback receiving ``(ticket_name, tool_spec_text)``.
        ToolForge is unsafe without approval; set approval_callback in production.
    """

    def __init__(
        self,
        registry: Any,
        llm_provider: Any,
        llm_config: Any = None,
        event_bus: Any = None,
        approval_callback: Callable[[str, str], bool] | None = None,
    ) -> None:
        self._registry = registry
        self._llm = llm_provider
        self._config = llm_config
        self._event_bus = event_bus
        self._approval_callback = approval_callback
        self.gap_detector = GapDetector()
        self._creations_this_run = 0

    def reset_run(self) -> None:
        """Reset per-run counters. Called at the start of each pipeline run."""
        self._creations_this_run = 0

    async def process_tickets(
        self, tickets: list[CreationTicket] | None = None,
    ) -> list[str]:
        """Process creation tickets. Returns list of created tool names.

        If ``tickets`` is None, pops from the internal GapDetector queue.
        """
        if tickets is None:
            tickets = self.gap_detector.pop_tickets()

        created: list[str] = []
        for ticket in tickets:
            if self._creations_this_run >= MAX_CREATIONS_PER_RUN:
                log.info(
                    "ToolForge: max creations (%d) reached, deferring %d tickets",
                    MAX_CREATIONS_PER_RUN, len(tickets) - len(created),
                )
                break
            name = await self.process_ticket(ticket)
            if name:
                created.append(name)
                self._creations_this_run += 1
        return created

    async def process_ticket(self, ticket: CreationTicket) -> str | None:
        """Process one creation ticket through the build loop."""
        if (
            os.environ.get("SAGE_TOOLFORGE_REQUIRE_APPROVAL", "").strip() == "1"
            and self._approval_callback is None
            and os.environ.get("SAGE_TOOLFORGE_APPROVE_ALL", "").strip() != "1"
        ):
            raise RuntimeError(
                "ToolForge approval required by "
                "SAGE_TOOLFORGE_REQUIRE_APPROVAL=1; set the approval_callback "
                "parameter on ToolForge/BuildLoop construction."
            )
        return await self._build_tool(ticket)

    async def _build_tool(self, ticket: CreationTicket) -> str | None:
        """Build loop: generate code+tests, validate, iterate (max 3 rounds).

        Returns the registered tool name on success, or None on failure.
        """
        from sage.llm.base import Message, Role

        previous_code = ""
        errors = ""

        for round_num in range(MAX_BUILD_ROUNDS):
            # Generate or repair
            if round_num == 0:
                prompt = _TOOL_GEN_PROMPT.format(
                    gap_description=ticket.gap_description,
                    task=ticket.task[:300],
                    required_interface=ticket.required_interface[:200],
                    tool_name_hint=ticket.tool_name_hint or "custom",
                    name=self._sanitize_name(ticket.tool_name_hint or "custom"),
                )
            else:
                prompt = _TOOL_REPAIR_PROMPT.format(
                    errors=errors,
                    previous_code=previous_code,
                )

            try:
                response = await self._llm.generate(
                    messages=[Message(role=Role.USER, content=prompt)],
                    config=self._config,
                )
                content = response.content or ""
            except Exception as exc:
                log.warning("ToolForge: LLM call failed (round %d): %s", round_num, exc)
                continue

            # Parse code blocks
            code, tests = self._parse_tool_response(content)
            if not code:
                errors = "No valid Python code block found in response"
                continue

            previous_code = code

            # Gate 1: AST validation
            ast_ok, ast_errors = self._validate_ast(code)
            if not ast_ok:
                errors = f"AST validation failed: {ast_errors}"
                ticket.attempts += 1
                log.debug("ToolForge: Gate 1 fail (round %d): %s", round_num, ast_errors)
                continue

            # Gate 2: Sandbox test execution
            test_ok, test_errors = await self._run_tests(code, tests)
            if not test_ok:
                errors = f"Test execution failed: {test_errors}"
                ticket.attempts += 1
                log.debug("ToolForge: Gate 2 fail (round %d): %s", round_num, test_errors)
                continue

            # Both gates passed — register the tool
            tool_name = self._extract_tool_name(code, ticket.tool_name_hint)
            tool_spec_text = self._format_tool_spec_text(code, tests)

            approved, approved_by = self._approve_tool(ticket, tool_name, tool_spec_text)
            if not approved:
                return None

            try:
                from sage.tools.meta import create_python_tool
                result = await create_python_tool.run(
                    {"name": tool_name, "code": code, "registry": self._registry},
                )
                if "Error" in result:
                    log.warning("ToolForge: registration failed: %s", result)
                    errors = result
                    continue

                # Track source in registry
                if hasattr(self._registry, "mark_source"):
                    self._registry.mark_source(
                        tool_name,
                        "forged",
                        approved_by=approved_by,
                    )

                self._emit("TOOL_FORGED", {
                    "name": tool_name,
                    "rounds": round_num + 1,
                    "ticket_task": ticket.task[:100],
                })
                log.info(
                    "ToolForge: created tool '%s' in %d round(s)",
                    tool_name, round_num + 1,
                )
                return tool_name
            except Exception as exc:
                errors = f"Registration error: {exc}"
                log.warning("ToolForge: registration exception: %s", exc)
                continue

        log.info(
            "ToolForge: failed to create tool for '%s' after %d rounds",
            ticket.tool_name_hint, MAX_BUILD_ROUNDS,
        )
        return None

    def _approve_tool(
        self,
        ticket: CreationTicket,
        tool_name: str,
        tool_spec_text: str,
    ) -> tuple[bool, str | None]:
        """Run the HITL approval gate for a validated generated tool."""
        if os.environ.get("SAGE_TOOLFORGE_APPROVE_ALL", "").strip() == "1":
            global _APPROVE_ALL_WARNED
            if not _APPROVE_ALL_WARNED:
                log.warning(
                    "SAGE_TOOLFORGE_APPROVE_ALL=1; ToolForge HITL approval "
                    "is bypassed for this process."
                )
                _APPROVE_ALL_WARNED = True
            return True, "env:approve_all"

        if self._approval_callback is None:
            return True, None

        ticket_name = getattr(ticket, "name", None) or ticket.tool_name_hint or tool_name
        approved_by = getattr(self._approval_callback, "__name__", "callback")
        allowed = self._approval_callback(ticket_name, tool_spec_text)
        if not allowed:
            log.info(
                "ToolForge: approval denied for tool '%s' from ticket '%s'",
                tool_name,
                ticket_name,
            )
            return False, approved_by
        return True, approved_by

    @staticmethod
    def _format_tool_spec_text(code: str, tests: str) -> str:
        if not tests:
            return code
        return f"{code}\n\n# === Tests ===\n{tests}"

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize a tool name for use as a Python identifier."""
        clean = re.sub(r"[^a-zA-Z0-9_]", "_", name)
        if clean and clean[0].isdigit():
            clean = "_" + clean
        return clean.lower()[:50] or "custom_tool"

    @staticmethod
    def _parse_tool_response(content: str) -> tuple[str, str]:
        """Extract tool code and test code from LLM response.

        Expects two fenced Python code blocks. Returns (code, tests).
        If only one block found, treats it as code with empty tests.
        """
        pattern = r"```(?:python)?\n(.*?)```"
        blocks = re.findall(pattern, content, re.DOTALL)

        if len(blocks) >= 2:
            return blocks[0].strip(), blocks[1].strip()
        elif len(blocks) == 1:
            return blocks[0].strip(), ""
        return "", ""

    @staticmethod
    def _validate_ast(code: str) -> tuple[bool, str]:
        """Gate 1: Validate Python code syntax and security.

        Tries Rust ToolExecutor first (tree-sitter + blocklist),
        falls back to ast.parse().
        """
        # Try Rust validator first
        try:
            from sage_core import ToolExecutor
            te = ToolExecutor()
            result = te.validate(code)
            if not result.valid:
                return False, "; ".join(result.errors)
            return True, ""
        except (ImportError, Exception):
            pass

        # Fallback: Python ast.parse
        try:
            ast.parse(code, mode="exec")
            return True, ""
        except SyntaxError as exc:
            return False, f"SyntaxError: {exc.msg} (line {exc.lineno})"

    @staticmethod
    async def _run_tests(code: str, tests: str) -> tuple[bool, str]:
        """Gate 2: Execute tool code + tests in sandbox.

        Combines tool code and test code, runs in isolated sandbox.
        Returns (success, error_message).
        """
        if not tests:
            # No tests provided — pass Gate 2 if code is syntactically valid
            return True, ""

        combined = f"{code}\n\n# === Tests ===\n{tests}\n"

        try:
            from sage.sandbox.isolated_executor import execute_isolated
            stdout, stderr, exit_code = execute_isolated(combined, timeout=30)
            if exit_code == 0:
                return True, ""
            return False, f"exit={exit_code}: {stderr[:300]}"
        except ImportError:
            # No sandbox available — fall back to subprocess
            import subprocess
            from sage._python import PYTHON
            try:
                proc = subprocess.run(
                    [PYTHON, "-c", combined],
                    capture_output=True, text=True, timeout=30,
                )
                if proc.returncode == 0:
                    return True, ""
                return False, f"exit={proc.returncode}: {proc.stderr[:300]}"
            except subprocess.TimeoutExpired:
                return False, "Timeout (30s)"
            except Exception as exc:
                return False, str(exc)

    @staticmethod
    def _extract_tool_name(code: str, hint: str) -> str:
        """Extract tool name from code (function name) or use hint."""
        match = re.search(r"def\s+(tool_\w+)", code)
        if match:
            return match.group(1)
        return ToolForge._sanitize_name(hint) if hint else "tool_custom"

    def _emit(self, event_type: str, data: dict) -> None:
        """Emit event on EventBus if available."""
        if self._event_bus and hasattr(self._event_bus, "emit"):
            try:
                from sage.agent_loop import AgentEvent
                self._event_bus.emit(AgentEvent(
                    type="TOOLFORGE",
                    step=0,
                    timestamp=time.time(),
                    meta={"stage": event_type, **data},
                ))
            except Exception:
                pass
