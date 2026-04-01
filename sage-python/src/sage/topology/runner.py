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

from sage._python import PYTHON
from sage.llm.base import LLMConfig, LLMProvider, Message, Role

log = logging.getLogger(__name__)

# Edge type constants (matching sage-core/src/topology/topology_graph.rs)
EDGE_CONTROL = 0
EDGE_MESSAGE = 1
EDGE_STATE = 2


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
        self._max_rounds = 3  # max re-executions per node (multi-turn debate)
        self._node_exec_count: dict[int, int] = {}  # track per-node execution count

    def _context_budget_per_predecessor(self, n_predecessors: int) -> int:
        """Compute per-predecessor character budget based on model context window.

        Adapts to the receiving model's capacity instead of a fixed 1000-char
        truncation. Reserves 30% of context for system prompt + task text,
        divides the rest among predecessors. Floors at 1000 chars.

        Based on TalkHier (arXiv 2502.11098): structured communication carrying
        full intermediate outputs improves accuracy over truncated handoffs.
        """
        max_tokens = 4096  # conservative default
        if self._config and hasattr(self._config, "max_tokens") and self._config.max_tokens:
            max_tokens = self._config.max_tokens
        # Estimate chars from tokens (~4 chars/token), reserve 30% for prompt+task
        available_chars = int(max_tokens * 0.7 * 4)
        budget = available_chars // max(n_predecessors, 1)
        return max(budget, 1000)  # never go below 1000

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

        budget = self._context_budget_per_predecessor(len(predecessor_indices))

        parts_with_roles: list[tuple[str, str]] = []
        for idx in predecessor_indices:
            output = self._node_outputs.get(idx)
            if output:
                node = self.graph.get_node(idx)
                role = getattr(node, "role", f"node-{idx}")
                truncated = self._truncate_output(output, budget)
                parts_with_roles.append((truncated, role))

        # Similarity gate: deduplicate near-identical predecessor outputs
        # Saves tokens when parallel workers produce similar answers (S2-MAD)
        deduplicated = self._deduplicate_context(parts_with_roles)

        return "\n\n".join(f"[{role}]: {text}" for text, role in deduplicated)

    def _gather_all_context(self) -> str:
        """Fallback: all completed nodes (legacy behavior)."""
        n_completed = len(self._node_outputs)
        budget = self._context_budget_per_predecessor(n_completed)

        parts_with_roles: list[tuple[str, str]] = []
        for idx in sorted(self._node_outputs.keys()):
            output = self._node_outputs[idx]
            if output:
                node = self.graph.get_node(idx)
                role = getattr(node, "role", f"node-{idx}")
                truncated = self._truncate_output(output, budget)
                parts_with_roles.append((truncated, role))

        deduplicated = self._deduplicate_context(parts_with_roles)
        return "\n\n".join(f"[{role}]: {text}" for text, role in deduplicated)

    @staticmethod
    def _deduplicate_context(
        parts: list[tuple[str, str]],
        threshold: float = 0.85,
    ) -> list[tuple[str, str]]:
        """Remove near-duplicate predecessor outputs via Jaccard word similarity.

        When multiple parallel workers produce similar answers (e.g., robust
        template with 3 workers), passing all of them wastes context. Keep only
        the longest unique output per similarity cluster.

        Based on S2-MAD (arXiv 2502.04790): -94% tokens, <2% perf loss.
        """
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
                        # Keep the longer one
                        if len(text_i) > len(text_j):
                            deduplicated[j] = (text_i, role_i)
                        is_duplicate = True
                        break
            if not is_duplicate:
                deduplicated.append((text_i, role_i))
        return deduplicated

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
            except Exception as exc:
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

        role = getattr(node, "role", f"node-{node_idx}")
        caps = getattr(node, "required_capabilities", [])

        # Use custom prompt if available, otherwise generate from role
        custom_prompt = getattr(node, "prompt", "")
        if custom_prompt:
            system_prompt = custom_prompt
        else:
            system_prompt = f"You are acting as: {role}."
            if caps:
                system_prompt += f" Your capabilities: {', '.join(caps)}."

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
        except Exception as exc:
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
                except Exception as fallback_exc:
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
        except Exception as exc:
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
                        except Exception as exc:
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
