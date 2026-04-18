"""TopologyRunner: execute TopologyGraph as real multi-agent system.

Bridges the gap between topology IR (Rust petgraph) and agent execution.
Uses TopologyExecutor for readiness-based scheduling and spawns per-node LLM calls.

Architecture follows MASFactory (2603.06007):
- Node lifecycle: aggregate predecessor outputs → build prompt → LLM call → store output
- Readiness: node executes when TopologyExecutor marks it ready
- Context: predecessor outputs injected via TopologyGraph.get_predecessors()
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Callable

import os

from sage._python import PYTHON
from sage.llm.base import LLMConfig, LLMProvider, Message, Role

log = logging.getLogger(__name__)

# Edge type constants (matching sage-core/src/topology/topology_graph.rs)
EDGE_CONTROL = 0
EDGE_MESSAGE = 1
EDGE_STATE = 2

# Prefix of phases/learn.py EMPTY_STEP_SENTINEL. An output starting with this
# means the upstream node exited its step budget without producing content —
# forwarding it to downstream nodes only teaches them that predecessors
# failed, then they cascade the same sentinel. We drop such outputs from
# predecessor context entirely (without fabricating a replacement prompt).
_SENTINEL_PREFIX = "[sage: agent exited after"

# Roles recognized as "planner" for the optional planner-output injection
# experiment. Kept in sync with topology/role_prompts.py _PLANNER aliases.
_PLANNER_ROLE_KEYWORDS = ("planner", "input_processor", "decomposer")

# Max characters of planner output injected into downstream system prompt.
# Keeps the prompt bounded regardless of how verbose the planner is.
_PLANNER_INJECTION_BUDGET = 2000


def _is_sentinel(output: str) -> bool:
    """True if output is the EMPTY_STEP_SENTINEL string from phases/learn.py."""
    return isinstance(output, str) and output.strip().startswith(_SENTINEL_PREFIX)


def _is_planner_role(role: str) -> bool:
    """True if a node role is a planner/decomposer variant."""
    if not isinstance(role, str):
        return False
    rl = role.lower()
    return any(kw in rl for kw in _PLANNER_ROLE_KEYWORDS)


class TopologyRunner:
    """Execute a TopologyGraph as a real multi-agent system.

    Single-use: each instance runs one topology execution. Do not call
    ``run()`` more than once (``_node_outputs`` is not reset between runs).

    Parameters
    ----------
    graph : TopologyGraph or compatible stub
        Must have ``node_count()``, ``get_node(idx)``.
    executor : TopologyExecutor or compatible stub
        Must have ``next_ready(graph)``, ``mark_completed(idx)``, ``is_done()``.
    llm_provider : LLMProvider
        The LLM provider for generating responses per node.
    llm_config : LLMConfig, optional
        Optional LLMConfig override.
    controller : TopologyController, optional
        Runtime adaptation controller (Phase C). If None, behaves as Phase B
        (no adaptation). When provided, ``evaluate_and_decide()`` is called
        after each node to trigger upgrade_model, spawn_subagent, reroute or
        prune actions.
    """

    def __init__(
        self,
        graph: Any,
        executor: Any,
        llm_provider: LLMProvider,
        llm_config: LLMConfig | None = None,
        *,
        provider_pool: Any | None = None,
        controller: Any | None = None,
        axis_hint: str = "",
        approval_callback: Callable | None = None,
        harness_config: Any | None = None,
        agent_loop_factory: Any | None = None,
    ) -> None:
        self.graph = graph
        self.executor = executor
        self._llm = llm_provider
        self._config = llm_config
        self._provider_pool = provider_pool
        self._controller = controller
        self._axis_hint = axis_hint
        self._approval_callback = approval_callback  # HITL: async fn(decision) -> bool
        self._node_outputs: dict[int, str] = {}
        self._node_exec_count: dict[int, int] = {}  # track per-node execution count
        # Aggregate tool-use telemetry summed across all per-node AgentLoops
        # executed by this runner. Surfaced to pipeline ctx so bench manifests
        # reflect real multi-agent behavior, not per-node zeros.
        self.tool_call_count: int = 0
        self.tool_turn_count: int = 0
        self.executed_commands: list[str] = []
        # Per-run cost aggregate. Before Apr 18 2026 per-node loops each had
        # their own total_cost_usd but nothing rolled them up, so benches read
        # system.agent_loop.total_cost_usd (the single top-level loop) and
        # saw 0 whenever topology ran in multi-node mode. Aggregating here
        # matches the tool_call_count pattern above and feeds the pipeline ctx.
        self.total_cost_usd: float = 0.0

        # Meta-Harness (arXiv 2603.28052): optional harness config overlay.
        # Loaded from config/harness.json at boot. Overrides context budget,
        # predecessor format, similarity threshold, system prompt templates,
        # and debate rounds — WITHOUT replacing any methods.
        self._harness = harness_config
        self._agent_loop_factory = agent_loop_factory
        self._max_rounds = (
            harness_config.execution.max_debate_rounds
            if harness_config else 3
        )

    def _context_budget_per_predecessor(self, n_predecessors: int, node_idx: int = 0) -> int:
        """Compute per-predecessor character budget based on model context window.

        Uses the receiving node's model context_window (from ModelCard, e.g.
        128K for GPT-5.4, 1M for Gemini), NOT config.max_tokens which is the
        output token limit. Reserves 30% for system prompt + task text.

        Based on TalkHier (arXiv 2502.11098): structured communication carrying
        full intermediate outputs improves accuracy over truncated handoffs.
        """
        context_window = 131072  # safe default (128K tokens)
        # Try to read real context_window from the node's assigned model
        try:
            node = self.graph.get_node(node_idx)
            model_id = getattr(node, "model_id", "")
            if model_id and self._provider_pool:
                _, resolved_config = self._provider_pool.resolve(model_id)
                cw = getattr(resolved_config, "context_window", 0)
                if cw and cw > 0:
                    context_window = cw
        except (AttributeError, RuntimeError):
            pass  # graph or pool unavailable — use default
        # 70% of context for predecessor outputs, ~4 chars per token
        _budget_ratio = self._harness.context.budget_ratio if self._harness else 0.7
        _chars_per_token = self._harness.context.chars_per_token if self._harness else 4
        _floor = self._harness.context.budget_floor_chars if self._harness else 1000
        available_chars = int(context_window * _budget_ratio * _chars_per_token)
        budget = available_chars // max(n_predecessors, 1)
        return max(budget, _floor)  # floor at 1000 chars

    def _truncate_output(self, output: str, budget: int) -> str:
        """Truncate output to budget, appending '...' if cut."""
        if len(output) <= budget:
            return output
        return output[:budget] + "..."

    def _gather_predecessor_context(self, node_idx: int) -> str:
        """Collect outputs from direct predecessors of node_idx only.

        Uses Rust TopologyGraph.get_predecessors() for correct DAG traversal.
        Falls back to all completed nodes if get_predecessors unavailable.
        Deduplicates near-identical outputs (S2-MAD arXiv 2502.04790).
        """
        predecessor_indices: list[int] = []
        try:
            predecessor_indices = self.graph.get_predecessors(node_idx)
        except (AttributeError, Exception):
            return self._gather_all_context()

        budget = self._context_budget_per_predecessor(len(predecessor_indices), node_idx)

        parts_with_roles: list[tuple[str, str]] = []
        for idx in predecessor_indices:
            output = self._node_outputs.get(idx)
            if not output:
                continue
            # Drop EMPTY_STEP_SENTINEL — forwarding "agent exited after N steps
            # with no content" to a downstream synthesizer teaches it that its
            # inputs failed, and it replies with the same sentinel (cascade
            # observed on smoke v4: 5/10 tasks ended with SENTINEL patches).
            # We strip at the source; no fabricated replacement — if all
            # predecessors are sentinels, the downstream node sees empty
            # context and falls back to the task prompt alone, which is
            # strictly better than being told "everyone failed".
            if _is_sentinel(output):
                log.info(
                    "topology.runner: dropped sentinel output from predecessor %d (role=%s)",
                    idx,
                    getattr(self.graph.get_node(idx), "role", f"node-{idx}"),
                )
                continue
            node = self.graph.get_node(idx)
            role = getattr(node, "role", f"node-{idx}")
            truncated = self._truncate_output(output, budget)
            parts_with_roles.append((truncated, role))

        # Similarity gate: deduplicate near-identical predecessor outputs
        # Saves tokens when parallel workers produce similar answers (S2-MAD)
        _sim_threshold = self._harness.context.similarity_threshold if self._harness else 0.90
        deduplicated = self._deduplicate_context(parts_with_roles, _sim_threshold)

        # Format with harness template if available
        _fmt = self._harness.context.predecessor_format if self._harness else "[{role}]: {text}"
        _sep = self._harness.context.predecessor_separator if self._harness else "\n\n"
        formatted = _sep.join(
            _fmt.format(role=role, text=text, node_idx=0, model_id="")
            for text, role in deduplicated
        )

        # Wrap with injection template if harness provides one
        if self._harness and formatted:
            return self._harness.context.injection_template.format(
                context=formatted,
                n_predecessors=len(predecessor_indices),
                task_preview="",
            )
        return formatted

    def _maybe_planner_injection(self, node_idx: int, system_prompt: str) -> str:
        """Optionally prepend upstream planner output to this node's system_prompt.

        Gated by SAGE_PLANNER_INJECTION=1 (default: off). The experiment is
        backed by MASS (arXiv 2502.02533): the structured decomposition plan
        is higher-signal for downstream nodes than the raw predecessor
        context mixed with other outputs. Still emitted via predecessor
        context too — this only adds explicit section at the top of the
        system prompt for nodes downstream of a planner.

        No-op if:
        - Flag is off
        - Current node IS a planner (skip self-injection)
        - No planner found among predecessors
        - Planner output is a sentinel (already stripped elsewhere, guard anyway)
        """
        if os.environ.get("SAGE_PLANNER_INJECTION") != "1":
            return system_prompt

        current = self.graph.get_node(node_idx)
        current_role = getattr(current, "role", "")
        if _is_planner_role(current_role):
            return system_prompt

        try:
            predecessors = self.graph.get_predecessors(node_idx)
        except (AttributeError, Exception):
            return system_prompt

        for pred_idx in predecessors:
            pred_node = self.graph.get_node(pred_idx)
            pred_role = getattr(pred_node, "role", "")
            if not _is_planner_role(pred_role):
                continue
            pred_output = self._node_outputs.get(pred_idx, "")
            if not pred_output or _is_sentinel(pred_output):
                continue
            # Bound the injection; planner output can be long
            truncated = pred_output[:_PLANNER_INJECTION_BUDGET]
            if len(pred_output) > _PLANNER_INJECTION_BUDGET:
                truncated += "\n... [truncated]"
            log.info(
                "topology.runner: injecting planner output (%d chars, role=%s) into node %d system prompt",
                len(truncated), pred_role, node_idx,
            )
            return (
                f"## Upstream plan (from {pred_role}):\n{truncated}\n\n"
                f"## Your role\n{system_prompt}"
            )
        return system_prompt

    def _gather_all_context(self) -> str:
        """Fallback: all completed nodes (legacy behavior)."""
        n_completed = len(self._node_outputs)
        budget = self._context_budget_per_predecessor(n_completed)

        parts_with_roles: list[tuple[str, str]] = []
        for idx in sorted(self._node_outputs.keys()):
            output = self._node_outputs[idx]
            if not output:
                continue
            # Same sentinel-strip logic as _gather_predecessor_context above.
            if _is_sentinel(output):
                continue
            node = self.graph.get_node(idx)
            role = getattr(node, "role", f"node-{idx}")
            truncated = self._truncate_output(output, budget)
            parts_with_roles.append((truncated, role))

        deduplicated = self._deduplicate_context(parts_with_roles)
        return "\n\n".join(f"[{role}]: {text}" for text, role in deduplicated)

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two embedding vectors (pure Python)."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a < 1e-15 or norm_b < 1e-15:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _deduplicate_jaccard(
        parts: list[tuple[str, str]],
        threshold: float = 0.85,
    ) -> list[tuple[str, str]]:
        """Fallback: Jaccard word dedup when embeddings unavailable."""
        if len(parts) <= 1:
            return parts
        deduplicated: list[tuple[str, str]] = []
        for text_i, role_i in parts:
            words_i = set(text_i.lower().split())
            is_duplicate = False
            for j, (text_j, _) in enumerate(deduplicated):
                words_j = set(text_j.lower().split())
                if words_i and words_j:
                    jaccard = len(words_i & words_j) / len(words_i | words_j)
                    if jaccard > threshold:
                        if len(text_i) < len(text_j):
                            deduplicated[j] = (text_i, role_i)
                        is_duplicate = True
                        break
            if not is_duplicate:
                deduplicated.append((text_i, role_i))
        return deduplicated

    @classmethod
    def _deduplicate_context(
        cls,
        parts: list[tuple[str, str]],
        threshold: float = 0.90,
    ) -> list[tuple[str, str]]:
        """Remove near-duplicate predecessor outputs via semantic similarity.

        Uses cosine similarity on arctic-embed-m embeddings (768-dim) when
        available. Falls back to Jaccard word similarity otherwise.

        Tie-breaker: keep **shortest** (penalize verbosity, not reward it).

        Based on S2-MAD (arXiv 2502.04790): -94% tokens, <2% perf loss.
        """
        if len(parts) <= 1:
            return parts

        # Try semantic similarity (cosine on embeddings)
        try:
            from sage.memory.embedder import Embedder
            emb = Embedder()
            if not emb.is_semantic:
                return cls._deduplicate_jaccard(parts)
            texts = [t for t, _ in parts]
            vectors = emb.embed_batch(texts)
        except (ImportError, RuntimeError, AttributeError):
            return cls._deduplicate_jaccard(parts)

        deduplicated: list[tuple[str, str, list[float]]] = []
        for (text_i, role_i), vec_i in zip(parts, vectors):
            is_duplicate = False
            for j, (text_j, _, vec_j) in enumerate(deduplicated):
                sim = cls._cosine_sim(vec_i, vec_j)
                if sim > threshold:
                    # Keep the SHORTER one (penalize verbosity)
                    if len(text_i) < len(text_j):
                        deduplicated[j] = (text_i, role_i, vec_i)
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduplicated.append((text_i, role_i, vec_i))

        return [(t, r) for t, r, _ in deduplicated]

    async def _execute_node_via_agent_loop(
        self, node_idx: int, task: str, context_override: str | None = None,
    ) -> str:
        """Execute an LLM node via per-node AgentLoop (Phase 2).

        Creates an independent AgentLoop instance for this node with:
        - Role-filtered tools (H6)
        - Validation level from system classification (H6)
        - Skip routing (H1) and topology (H4) flags
        - Predecessor context in user message (H7)
        """
        node = self.graph.get_node(node_idx)
        role = getattr(node, "role", f"node-{node_idx}")
        caps = getattr(node, "required_capabilities", [])

        # Build system prompt (same logic as _execute_node)
        custom_prompt = getattr(node, "prompt", "")
        if custom_prompt:
            system_prompt = custom_prompt
        else:
            # New (2026-04-17): try the per-role prompt registry first. The
            # Rust template factories (sequential / parallel / robust) build
            # nodes with empty `prompt` fields for planner/coder/worker, so
            # the previous fallback — "You are acting as: {role}." — was
            # what every non-synthesizer agent actually saw. Smoke evidence:
            # docs/benchmarks/2026-04-17-swebench-smoke-debug.md.
            from sage.topology.role_prompts import get_role_prompt
            _role_prompt = get_role_prompt(role)
            if _role_prompt:
                system_prompt = _role_prompt
            else:
                _default_tmpl = (
                    self._harness.prompts.default_template if self._harness
                    else "You are acting as: {role}."
                )
                system_prompt = _default_tmpl.format(
                    role=role, capabilities=", ".join(caps) if caps else "",
                    task_preview=task[:200], n_predecessors=0,
                )
            if caps:
                _cap_tmpl = (
                    self._harness.prompts.capability_template if self._harness
                    else " Your capabilities: {capabilities}."
                )
                system_prompt += _cap_tmpl.format(capabilities=", ".join(caps))

        if self._harness:
            if self._harness.prompts.global_prefix:
                system_prompt = self._harness.prompts.global_prefix + "\n" + system_prompt
            if self._harness.prompts.global_suffix:
                system_prompt = system_prompt + "\n" + self._harness.prompts.global_suffix

        # Optional: prepend upstream planner output (MASS arXiv 2502.02533).
        # Gated by SAGE_PLANNER_INJECTION=1 — off by default so the base
        # behavior is unchanged.
        system_prompt = self._maybe_planner_injection(node_idx, system_prompt)

        # Resolve per-node model
        node_model_id = getattr(node, "model_id", "")
        if node_model_id and self._provider_pool:
            provider, config = self._provider_pool.resolve(node_model_id)
        else:
            provider, config = self._llm, self._config

        # Create per-node AgentLoop (H8: independent instance)
        loop = self._agent_loop_factory(
            node_role=role,
            node_name=f"node-{node_idx}-{role}",
            llm_provider=provider,
            llm_config=config,
            system_prompt=system_prompt,
        )

        # Build task with predecessor context (H7)
        context = (
            context_override
            if context_override is not None
            else self._gather_predecessor_context(node_idx)
        )
        if context:
            full_task = (
                f"## Previous agent output:\n{context}\n\n"
                f"## Task:\n{task}"
            )
        else:
            full_task = task

        # Execute
        result = await loop.run(full_task)
        self._node_outputs[node_idx] = result
        # Aggregate tool-use telemetry — per-node counters are local to each
        # AgentLoop; without this rollup the pipeline ctx sees zero even
        # when nodes did call tools.
        self.tool_call_count += int(getattr(loop, "tool_call_count", 0) or 0)
        self.tool_turn_count += int(getattr(loop, "tool_turn_count", 0) or 0)
        self.total_cost_usd += float(getattr(loop, "total_cost_usd", 0.0) or 0.0)
        node_commands = list(getattr(loop, "executed_commands", []) or [])
        if node_commands:
            self.executed_commands.extend(f"[{role}] {c}" for c in node_commands)
        log.info(
            "[TopologyRunner] node %d (%s) completed via agent_loop, output %d chars, tool_calls=%d",
            node_idx, role, len(result), int(getattr(loop, "tool_call_count", 0) or 0),
        )

        # Inter-node quality signal (VPRMs pattern, arXiv 2601.17223):
        # evaluate output quality and let controller decide next action.
        if self._controller and result:
            try:
                node_ctx = {
                    "node_idx": node_idx,
                    "latency_ms": 0.0,
                    "model_id": getattr(node, "model_id", ""),
                    "output_length": len(result),
                    "axis_hint": self._axis_hint,
                }
                decision = self._controller.evaluate_and_decide(
                    node_idx=node_idx,
                    result=result,
                    task=task,
                    topology=self.graph,
                    ctx=node_ctx,
                )
                if decision and hasattr(decision, 'action'):
                    log.debug(
                        "[TopologyRunner] controller decision for node %d: %s",
                        node_idx, decision.action,
                    )
            except Exception as exc:
                log.debug("[TopologyRunner] controller evaluation failed: %s", exc)

        return result

    async def _execute_code_node(
        self, node_idx: int, task: str, context_override: str | None = None,
    ) -> str:
        """Execute a code node in sandbox (HyEvo v^Code deterministic execution).

        Code nodes run synthesized Python in a restricted sandbox instead of
        calling an LLM. This offloads deterministic work (validation, parsing,
        computation) from expensive LLM inference.
        """
        import json
        import time

        node = self.graph.get_node(node_idx)
        role = getattr(node, "role", f"node-{node_idx}")
        code_spec = getattr(node, "code_spec", "") or getattr(node, "prompt", "")

        if not code_spec:
            log.error("Code node %d (%s) has no code_spec", node_idx, role)
            return f"ERROR: code node {node_idx} has no code_spec"

        context = (
            context_override
            if context_override is not None
            else self._gather_predecessor_context(node_idx)
        )

        t0 = time.monotonic()

        # Build a self-contained script that receives task+context via globals
        wrapped_code = (
            f"_TASK = {json.dumps(task[:2000])}\n"
            f"_CONTEXT = {json.dumps(context[:5000])}\n"
            f"{code_spec}\n"
        )

        try:
            from sage.sandbox.isolated_executor import execute_isolated
            stdout, stderr, exit_code = execute_isolated(wrapped_code, timeout=30)
            output = stdout
        except ImportError:
            # Fallback: subprocess execution (no bwrap)
            import subprocess
            try:
                proc = subprocess.run(
                    [PYTHON, "-c", wrapped_code],
                    capture_output=True, text=True, timeout=30,
                )
                output = proc.stdout
                stderr = proc.stderr
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                output = ""
                stderr = "TIMEOUT"
                exit_code = -1
            except (OSError, ValueError) as exc:
                output = ""
                stderr = str(exc)
                exit_code = -1

        latency_ms = (time.monotonic() - t0) * 1000

        if stderr and exit_code != 0:
            log.warning(
                "Code node %d (%s) failed (exit=%d, %.0fms): %s",
                node_idx, role, exit_code, latency_ms, stderr[:200],
            )
        else:
            log.info(
                "Code node %d (%s) completed (%.0fms, %d chars output)",
                node_idx, role, latency_ms, len(output),
            )

        self._node_outputs[node_idx] = output
        return output

    async def _execute_solver_node(
        self, node_idx: int, task: str, context_override: str | None = None,
    ) -> str:
        """Execute a formal solver node — try Rust solver, fall back to LLM.

        Hybrid approach (MALT, arXiv 2412.01928):
        1. Parse equations from formalizer output
        2. Solve via Rust (exact, sub-ms, deterministic)
        3. If solver fails → fall back to LLM chain-of-thought on the
           original task (the LLM can reason through what it can't formalize)

        This gives us the best of both: exact answers when formalization
        works, LLM reasoning when it doesn't.
        """
        import time as _time

        node = self.graph.get_node(node_idx)
        role = getattr(node, "role", f"node-{node_idx}")

        context = (
            context_override
            if context_override is not None
            else self._gather_predecessor_context(node_idx)
        )

        t0 = _time.monotonic()

        # ── Phase 1: Parse equations from formalizer output ────────────
        equations = []
        answer_var = None
        source = context if context else task

        for line in source.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("```") or line.startswith("-") or line.startswith("*"):
                continue
            if "=" in line and not line.startswith("="):
                parts = line.split("=", 1)
                var = parts[0].strip().lower().replace(" ", "_")
                expr = parts[1].strip()
                if var in ("answer", "the_answer", "final_answer", "result"):
                    answer_var = expr.strip().lower().replace(" ", "_")
                else:
                    equations.append((var, expr))

        # ── Phase 2: Try Rust solver ───────────────────────────────────
        solver_answer = None
        n_resolved = 0
        if equations:
            try:
                from sage_core import SmtVerifier
                solved = SmtVerifier.solve_equations(equations)
                n_resolved = len(solved)
                if solved:
                    # Find the answer variable
                    if answer_var and answer_var in solved:
                        solver_answer = str(solved[answer_var])
                    else:
                        target = self._infer_answer_variable(task, solved)
                        if target:
                            solver_answer = str(solved[target])
                        elif equations:
                            last_var = equations[-1][0]
                            if last_var in solved:
                                solver_answer = str(solved[last_var])
            except (ImportError, RuntimeError) as exc:
                log.warning("Solver node %d: Rust solver error: %s", node_idx, exc)

        solve_ms = (_time.monotonic() - t0) * 1000

        # ── Phase 3: Decide — use solver or fall back to LLM ──────────
        # Solver succeeded if: we got equations, resolved most of them,
        # and found a numeric answer.
        solver_ok = (
            solver_answer is not None
            and n_resolved >= len(equations) * 0.7  # resolved ≥70% of vars
        )

        if solver_ok:
            output = solver_answer
            log.info(
                "Solver node %d (%s): Rust solved %d/%d vars, answer=%s (%.1fms)",
                node_idx, role, n_resolved, len(equations), output, solve_ms,
            )
        else:
            # Fall back to LLM chain-of-thought on the original task.
            # The formalizer tried but the solver couldn't handle it —
            # let a strong LLM reason through the problem directly.
            log.info(
                "Solver node %d (%s): Rust solved %d/%d vars (%.1fms) — "
                "falling back to LLM chain-of-thought",
                node_idx, role, n_resolved, len(equations), solve_ms,
            )
            try:
                from sage.llm.base import Message, Role
                messages = [
                    Message(
                        role=Role.SYSTEM,
                        content=(
                            "Solve this math problem step by step. "
                            "Verify each intermediate result. "
                            "Give your final answer as a single number only."
                        ),
                    ),
                    Message(role=Role.USER, content=task),
                ]
                response = await self._llm.generate(
                    messages=messages, config=self._config,
                )
                output = response.content or ""
            except Exception as exc:
                log.warning("Solver node %d: LLM fallback failed: %s", node_idx, exc)
                output = solver_answer or ""

        self._node_outputs[node_idx] = output
        return output

    @staticmethod
    def _infer_answer_variable(
        question: str, solved: dict[str, int],
    ) -> str | None:
        """Infer which solved variable the question asks about.

        Uses word-overlap scoring between the question and each solved
        variable name. The variable with the highest overlap wins.
        No regex — pure set intersection on normalized words.
        """
        # Normalize question into a set of lowercase word tokens
        q_lower = question.lower().replace("'s", " ").replace("'", " ")
        # Strip punctuation
        q_clean = "".join(c if c.isalnum() or c == " " else " " for c in q_lower)
        q_words = set(q_clean.split())

        # Remove common stop words that add noise
        stop = {
            "the", "a", "an", "is", "are", "was", "were", "how", "many",
            "much", "does", "do", "did", "have", "has", "had", "what",
            "each", "of", "in", "to", "for", "and", "or", "that", "this",
            "number", "total", "give", "your", "final", "answer", "only",
            "as", "step", "by", "think", "verify", "result", "intermediate",
        }
        q_content = q_words - stop

        if not q_content:
            return None

        # Score each solved variable by word overlap with the question
        best_var: str | None = None
        best_score = 0
        for var_name in solved:
            var_words = set(var_name.lower().split("_"))
            overlap = len(q_content & var_words)
            if overlap > best_score:
                best_score = overlap
                best_var = var_name

        return best_var if best_score >= 2 else None

    async def _execute_node(
        self, node_idx: int, task: str, context_override: str | None = None,
    ) -> str:
        """Execute a single topology node — LLM call or code sandbox.

        HyEvo hybrid dispatch (arXiv 2603.19639):
          - node_type="llm" → LLM inference via ProviderPool
          - node_type="code" → deterministic sandbox execution

        Parameters
        ----------
        context_override : str, optional
            Pre-captured context snapshot. Used by parallel batches to avoid
            race conditions on ``_node_outputs`` during ``asyncio.gather``.
        """
        node = self.graph.get_node(node_idx)

        # HyEvo code node dispatch: deterministic sandbox execution
        node_type = getattr(node, "node_type", "llm")
        if node_type == "code":
            return await self._execute_code_node(node_idx, task, context_override)

        # Formal solver node: parse equations from predecessor, solve via Rust
        if node_type == "solver" or getattr(node, "role", "") == "solver":
            return await self._execute_solver_node(node_idx, task, context_override)

        # Phase 2: LLM nodes use agent_loop when factory available
        if self._agent_loop_factory:
            return await self._execute_node_via_agent_loop(node_idx, task, context_override)

        role = getattr(node, "role", f"node-{node_idx}")
        caps = getattr(node, "required_capabilities", [])

        # Use custom prompt if available, otherwise generate from role
        custom_prompt = getattr(node, "prompt", "")
        if custom_prompt:
            system_prompt = custom_prompt
        else:
            # Meta-Harness: configurable default template
            _default_tmpl = (
                self._harness.prompts.default_template if self._harness
                else "You are acting as: {role}."
            )
            system_prompt = _default_tmpl.format(
                role=role, capabilities=", ".join(caps) if caps else "",
                task_preview=task[:200], n_predecessors=0,
            )
            if caps:
                _cap_tmpl = (
                    self._harness.prompts.capability_template if self._harness
                    else " Your capabilities: {capabilities}."
                )
                system_prompt += _cap_tmpl.format(capabilities=", ".join(caps))

        # Meta-Harness: global prefix/suffix applied to ALL system prompts
        if self._harness:
            if self._harness.prompts.global_prefix:
                system_prompt = self._harness.prompts.global_prefix + "\n" + system_prompt
            if self._harness.prompts.global_suffix:
                system_prompt = system_prompt + "\n" + self._harness.prompts.global_suffix

        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=system_prompt),
        ]

        context = context_override if context_override is not None else self._gather_predecessor_context(node_idx)
        if context:
            messages.append(Message(
                role=Role.SYSTEM,
                content=f"Context from previous agents:\n{context}",
            ))

        messages.append(Message(role=Role.USER, content=task))

        # Resolve per-node model if ProviderPool available
        node_model_id = getattr(node, "model_id", "")
        if node_model_id and self._provider_pool:
            provider, config = self._provider_pool.resolve(node_model_id)
        else:
            provider, config = self._llm, self._config

        # Context gate: if payload exceeds model window, compress via summarization
        # Uses context_window (input limit, e.g. 128K) NOT max_tokens (output limit, e.g. 8K)
        total_chars = sum(len(m.content) for m in messages)
        context_window = getattr(config, 'context_window', 0) or 128000
        estimated_tokens = total_chars // 4
        if estimated_tokens > context_window * 0.85:
            # Compress the context message to fit
            context_msg = next(
                (m for m in messages if m.content.startswith("Context from previous")),
                None,
            )
            if context_msg:
                log.warning(
                    "Context overflow for node %d (%s): %d tokens > %d * 0.85, compressing",
                    node_idx, role, estimated_tokens, context_window,
                )
                try:
                    summary_msgs = [
                        Message(role=Role.SYSTEM, content="Summarize concisely. Preserve all key facts, numbers, code, and conclusions."),
                        Message(role=Role.USER, content=context_msg.content[:context_window * 2]),
                    ]
                    summary_resp = await asyncio.wait_for(
                        provider.generate(messages=summary_msgs, config=config),
                        timeout=30.0,
                    )
                    context_msg.content = f"Context (summarized):\n{summary_resp.content or ''}"
                except (RuntimeError, TimeoutError, asyncio.TimeoutError) as exc:
                    # Compression failed — hard truncate as last resort
                    max_chars = int(context_window * 0.6 * 4)
                    context_msg.content = context_msg.content[:max_chars] + "\n[truncated]"
                    log.error("Context compression failed for node %d: %s", node_idx, exc)

        # Per-node resilience: timeout + retry with fallback provider
        output = ""
        try:
            response = await asyncio.wait_for(
                provider.generate(messages=messages, config=config),
                timeout=60.0,  # 60s per node, not per topology
            )
            output = response.content or ""
            # Record success in circuit breaker
            provider_name = getattr(config, "provider", "unknown")
            if self._provider_pool and hasattr(self._provider_pool, "record_success"):
                self._provider_pool.record_success(provider_name)
        except (RuntimeError, TimeoutError, asyncio.TimeoutError, ConnectionError) as exc:
            provider_name = getattr(config, "provider", "unknown")
            # Record failure in circuit breaker
            if self._provider_pool and hasattr(self._provider_pool, "record_failure"):
                self._provider_pool.record_failure(provider_name, exc)
            log.warning(
                "[TopologyRunner] node %d (%s) failed with %s provider: %s — retrying with default",
                node_idx, role, provider_name, str(exc)[:150],
            )
            # Fallback to first available provider (connector.py = source of truth)
            if provider is not self._llm:
                try:
                    import os
                    from sage.providers.connector import get_available_providers
                    from sage.providers.openai_compat import OpenAICompatProvider
                    fallback_cfgs = get_available_providers()
                    fallback_cfg = fallback_cfgs[0] if fallback_cfgs else None
                    if fallback_cfg and fallback_cfg.get("sdk") != "google-genai":
                        fallback_provider = OpenAICompatProvider(
                            api_key=os.environ.get(fallback_cfg["api_key_env"], ""),
                            base_url=fallback_cfg["base_url"],
                            provider_name=fallback_cfg["provider"],
                        )
                        fallback_model = fallback_cfg.get("default_model", "")
                        fallback_config = LLMConfig(provider=fallback_cfg["provider"], model=fallback_model)
                        response = await fallback_provider.generate(
                            messages=messages,
                            config=fallback_config,
                        )
                    else:
                        response = await self._llm.generate(
                            messages=messages,
                            config=self._config or LLMConfig(provider="default", model="default"),
                        )
                    output = response.content or ""
                    log.info(
                        "[TopologyRunner] node %d (%s) succeeded with fallback (%s)",
                        node_idx, role, fallback_cfg["provider"] if fallback_cfg else "default",
                    )
                except (RuntimeError, TimeoutError, asyncio.TimeoutError, ConnectionError) as fallback_exc:
                    log.error(
                        "[TopologyRunner] node %d (%s) fallback also failed: %s",
                        node_idx, role, str(fallback_exc)[:150],
                    )
            else:
                log.error(
                    "[TopologyRunner] node %d (%s) default provider failed, no fallback: %s",
                    node_idx, role, str(exc)[:150],
                )
        self._node_outputs[node_idx] = output
        log.info(
            "[TopologyRunner] node %d (%s) completed, output %d chars",
            node_idx,
            role,
            len(output),
        )
        return output

    async def _retry_with_upgrade(self, node_idx: int, decision: Any, task: str) -> str:
        """Model upgrade: re-resolve provider via ProviderPool and retry node.

        The controller already called assign_single_node on the topology to
        update the node's model_id. Re-executing the node picks up the new
        model automatically via ProviderPool.resolve().
        """
        if decision.new_model_id:
            try:
                self.graph.set_node_model_id(node_idx, decision.new_model_id)
                log.info("Node %d model upgraded to %s", node_idx, decision.new_model_id)
            except (AttributeError, Exception) as exc:
                log.warning("Could not set model_id on node %d: %s", node_idx, exc)
        return await self._execute_node(node_idx, task)

    async def _spawn_sub(self, node_idx: int, decision: Any, task: str) -> None:
        """Sub-agent spawn: run emergent sub-task and inject result into node output."""
        sub_task = decision.reason
        if not sub_task:
            return
        try:
            from sage.llm.base import Message, Role  # local re-import for clarity
            provider = self._llm
            config = self._config
            if self._provider_pool:
                node = self.graph.get_node(node_idx) if hasattr(self.graph, "get_node") else None
                model_id = getattr(node, "model_id", "") if node else ""
                if model_id:
                    provider, config = self._provider_pool.resolve(model_id)
            response = await provider.generate(
                messages=[Message(role=Role.USER, content=sub_task)],
                config=config,
            )
            sub_result = response.content or ""
            # Inject into node outputs
            existing = self._node_outputs.get(node_idx, "")
            self._node_outputs[node_idx] = f"{existing}\n[Sub-agent]: {sub_result}"
        except (RuntimeError, TimeoutError, ValueError) as exc:
            log.warning("Sub-agent spawn failed: %s", exc)

    async def run(self, task: str) -> str:
        """Execute the full topology, returning the final node's output.

        For parallel batches, ``last_output`` is the last node in executor
        order. Topologies that need aggregation should include an explicit
        aggregator node in a subsequent batch.

        If a controller is attached and decides ``reroute_topology``, this
        method returns the special sentinel ``"__REROUTE__"`` so the caller
        (Pipeline Stage 4) can handle the reroute.
        """
        last_output = ""

        while not self.executor.is_done():
            ready = self.executor.next_ready(self.graph)
            if not ready:
                break

            if len(ready) == 1:
                node_idx = ready[0]
                import time as _time
                _t0 = _time.monotonic()
                result = await self._execute_node(node_idx, task)
                _latency_ms = (_time.monotonic() - _t0) * 1000

                # Phase C: runtime adaptation (single-node path)
                if self._controller:
                    node_ctx = {
                        "node_idx": node_idx,
                        "latency_ms": _latency_ms,
                        "model_id": getattr(self.graph.get_node(node_idx), "model_id", ""),
                        "output_length": len(result),
                        "axis_hint": self._axis_hint,
                    }
                    decision = self._controller.evaluate_and_decide(
                        node_idx, result, task, self.graph, node_ctx,
                        parallel_outputs=None,
                    )
                    # HITL: pause for human approval on disruptive actions
                    # Based on LangGraph interrupt() + A2A input-required
                    if (self._approval_callback
                            and decision.action in ("upgrade_model", "reroute_topology", "open_gate")):
                        try:
                            approved = await self._approval_callback(decision)
                            if not approved:
                                log.info("HITL rejected %s for node %d", decision.action, node_idx)
                                decision = type(decision)(action="continue", target_node=node_idx)
                        except (RuntimeError, TimeoutError, asyncio.TimeoutError) as exc:
                            log.warning("HITL callback failed: %s, proceeding", exc)
                    if decision.action == "upgrade_model":
                        result = await self._retry_with_upgrade(node_idx, decision, task)
                        self._node_outputs[node_idx] = result
                    elif decision.action == "spawn_subagent":
                        await self._spawn_sub(node_idx, decision, task)
                    elif decision.action == "reroute_topology":
                        return "__REROUTE__"
                    elif decision.action == "prune_node":
                        try:
                            self.executor.mark_skipped(decision.target_node)
                        except (AttributeError, Exception):
                            pass  # Executor may not support skip
                        log.info("Node %d pruned by controller", decision.target_node)
                    elif decision.action == "open_gate":
                        # Multi-turn: re-enable a node for another round
                        # MALT (arXiv 2412.01928): Generation→Verification→Refinement
                        target = decision.gate_target
                        source = decision.gate_source
                        if target is not None and source is not None:
                            count = self._node_exec_count.get(target, 1)
                            if count < self._max_rounds:
                                self.executor.open_gate(self.graph, source, target)
                                self.executor.reset_node(target)
                                self._node_exec_count[target] = count + 1
                                log.info(
                                    "Multi-turn: reopened gate %d→%d (round %d/%d)",
                                    source, target, count + 1, self._max_rounds,
                                )
                            else:
                                log.info(
                                    "Multi-turn: max rounds reached for node %d (%d/%d)",
                                    target, count, self._max_rounds,
                                )

                self.executor.mark_completed(node_idx)
                self._node_exec_count[node_idx] = self._node_exec_count.get(node_idx, 0) + 1
                last_output = self._node_outputs.get(node_idx, result)
            else:
                # Snapshot context before gather to prevent race:
                # concurrent coroutines must not see each other's outputs.
                ctx_snapshot = self._gather_all_context()
                coros = [
                    self._execute_node(idx, task, context_override=ctx_snapshot)
                    for idx in ready
                ]
                import time as _time
                _t0_par = _time.monotonic()
                results = await asyncio.gather(*coros)
                _par_latency_ms = (_time.monotonic() - _t0_par) * 1000

                # Phase C: runtime adaptation (parallel path)
                if self._controller:
                    parallel_outputs = list(results)
                    for idx, result in zip(ready, results):
                        node_ctx = {
                            "node_idx": idx,
                            "latency_ms": _par_latency_ms,  # Total parallel batch latency
                            "model_id": getattr(self.graph.get_node(idx), "model_id", ""),
                            "output_length": len(result),
                        }
                        decision = self._controller.evaluate_and_decide(
                            idx, result, task, self.graph, node_ctx,
                            parallel_outputs=parallel_outputs,
                        )
                        if decision.action == "upgrade_model":
                            upgraded = await self._retry_with_upgrade(idx, decision, task)
                            self._node_outputs[idx] = upgraded
                        elif decision.action == "spawn_subagent":
                            await self._spawn_sub(idx, decision, task)
                        elif decision.action == "reroute_topology":
                            return "__REROUTE__"
                        elif decision.action == "prune_node":
                            try:
                                self.executor.mark_skipped(decision.target_node)
                            except (AttributeError, Exception):
                                pass  # Executor may not support skip
                            log.info("Node %d pruned by controller", decision.target_node)

                for idx, output in zip(ready, results):
                    self.executor.mark_completed(idx)
                    last_output = self._node_outputs.get(idx, output)

        return last_output

    async def run_traced(self, task: str) -> list[dict]:
        """Execute topology and return per-node traces for GiGPO step rewards.

        Returns a list of dicts, one per executed node:
            [{"node_idx": 0, "role": "coder", "output": "...", "latency": 1.2}, ...]

        Uses the same execution logic as run() (ProviderPool, controller, fallback)
        but captures per-node metadata instead of just the final output.
        """
        import time
        traces: list[dict] = []

        while not self.executor.is_done():
            ready = self.executor.next_ready(self.graph)
            if not ready:
                break

            for node_idx in ready:
                t0 = time.time()
                node = self.graph.get_node(node_idx)
                role = getattr(node, "role", f"node-{node_idx}")

                result = await self._execute_node(node_idx, task)

                _node_latency_ms = (time.time() - t0) * 1000

                # Phase C adaptation (same as run())
                if self._controller:
                    node_ctx = {
                        "node_idx": node_idx,
                        "latency_ms": _node_latency_ms,
                        "model_id": getattr(node, "model_id", ""),
                        "output_length": len(result),
                    }
                    decision = self._controller.evaluate_and_decide(
                        node_idx, result, task, self.graph, node_ctx,
                        parallel_outputs=None,
                    )
                    if decision.action == "upgrade_model":
                        result = await self._retry_with_upgrade(node_idx, decision, task)
                        self._node_outputs[node_idx] = result

                self.executor.mark_completed(node_idx)

                traces.append({
                    "node_idx": node_idx,
                    "role": role,
                    "output": self._node_outputs.get(node_idx, result),
                    "latency": time.time() - t0,
                    "model_id": getattr(node, "model_id", ""),
                })

        return traces

    async def run_stream(self, task: str):
        """Execute topology, yielding per-node events as an async generator.

        Yields dicts with event types:
        - {"type": "node_start", "node_idx": int, "role": str}
        - {"type": "node_done", "node_idx": int, "role": str, "output": str, "latency_ms": float}
        - {"type": "topology_done", "final_output": str, "node_count": int}

        Enables real-time UI updates (LangGraph-style streaming) and is the
        foundation for HITL interrupt/resume (Patch 6).
        """
        import time as _time

        last_output = ""
        nodes_executed = 0

        while not self.executor.is_done():
            ready = self.executor.next_ready(self.graph)
            if not ready:
                break

            if len(ready) == 1:
                node_idx = ready[0]
                node = self.graph.get_node(node_idx)
                role = getattr(node, "role", f"node-{node_idx}")

                yield {"type": "node_start", "node_idx": node_idx, "role": role}

                _t0 = _time.monotonic()
                result = await self._execute_node(node_idx, task)
                _latency_ms = (_time.monotonic() - _t0) * 1000

                # Controller adaptation (same logic as run())
                if self._controller:
                    node_ctx = {
                        "node_idx": node_idx,
                        "latency_ms": _latency_ms,
                        "model_id": getattr(node, "model_id", ""),
                        "output_length": len(result),
                        "axis_hint": self._axis_hint,
                    }
                    decision = self._controller.evaluate_and_decide(
                        node_idx, result, task, self.graph, node_ctx,
                        parallel_outputs=None,
                    )
                    if decision.action == "upgrade_model":
                        result = await self._retry_with_upgrade(node_idx, decision, task)
                        self._node_outputs[node_idx] = result
                    elif decision.action == "reroute_topology":
                        yield {"type": "topology_reroute", "reason": decision.reason}
                        return
                    elif decision.action == "open_gate" and decision.gate_target is not None:
                        target = decision.gate_target
                        source = decision.gate_source
                        count = self._node_exec_count.get(target, 1)
                        if source is not None and count < self._max_rounds:
                            self.executor.open_gate(self.graph, source, target)
                            self.executor.reset_node(target)
                            self._node_exec_count[target] = count + 1

                self.executor.mark_completed(node_idx)
                self._node_exec_count[node_idx] = self._node_exec_count.get(node_idx, 0) + 1
                last_output = self._node_outputs.get(node_idx, result)
                nodes_executed += 1

                yield {
                    "type": "node_done",
                    "node_idx": node_idx,
                    "role": role,
                    "output": last_output,
                    "latency_ms": _latency_ms,
                }
            else:
                # Parallel batch — emit start for all, then done for all
                for idx in ready:
                    node = self.graph.get_node(idx)
                    yield {"type": "node_start", "node_idx": idx,
                           "role": getattr(node, "role", f"node-{idx}")}

                ctx_snapshot = self._gather_all_context()
                coros = [
                    self._execute_node(idx, task, context_override=ctx_snapshot)
                    for idx in ready
                ]
                _t0_par = _time.monotonic()
                results = await asyncio.gather(*coros)
                _par_latency_ms = (_time.monotonic() - _t0_par) * 1000

                for idx, output in zip(ready, results):
                    self.executor.mark_completed(idx)
                    last_output = self._node_outputs.get(idx, output)
                    nodes_executed += 1
                    yield {
                        "type": "node_done",
                        "node_idx": idx,
                        "role": getattr(self.graph.get_node(idx), "role", f"node-{idx}"),
                        "output": last_output,
                        "latency_ms": _par_latency_ms,
                    }

        yield {
            "type": "topology_done",
            "final_output": last_output,
            "node_count": nodes_executed,
        }
