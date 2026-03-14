"""TopologyRunner: execute TopologyGraph as real multi-agent system.

Bridges the gap between topology IR (Rust petgraph) and agent execution.
Uses TopologyExecutor for readiness-based scheduling and spawns per-node LLM calls.

Architecture follows MASFactory (2603.06007):
- Node lifecycle: aggregate predecessor outputs → build prompt → LLM call → store output
- Readiness: node executes when TopologyExecutor marks it ready
- Context: all completed predecessor outputs are injected (execution-order tracking,
  since TopologyGraph does not expose get_edges() to Python yet)
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

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
    ) -> None:
        self.graph = graph
        self.executor = executor
        self._llm = llm_provider
        self._config = llm_config
        self._provider_pool = provider_pool
        self._controller = controller
        self._node_outputs: dict[int, str] = {}

    def _gather_completed_context(self) -> str:
        """Collect outputs from ALL previously completed nodes.

        Uses execution-order tracking instead of edge-level queries
        (TopologyGraph does not expose get_edges() to Python).
        """
        context_parts: list[str] = []
        for idx in sorted(self._node_outputs.keys()):
            output = self._node_outputs[idx]
            if output:
                node = self.graph.get_node(idx)
                role = getattr(node, "role", f"node-{idx}")
                context_parts.append(f"[{role}]: {output}")
        return "\n\n".join(context_parts)

    async def _execute_node(
        self, node_idx: int, task: str, context_override: str | None = None,
    ) -> str:
        """Execute a single topology node as an LLM call.

        Parameters
        ----------
        context_override : str, optional
            Pre-captured context snapshot. Used by parallel batches to avoid
            race conditions on ``_node_outputs`` during ``asyncio.gather``.
        """
        node = self.graph.get_node(node_idx)
        role = getattr(node, "role", f"node-{node_idx}")
        caps = getattr(node, "required_capabilities", [])

        system_prompt = f"You are acting as: {role}."
        if caps:
            system_prompt += f" Your capabilities: {', '.join(caps)}."

        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=system_prompt),
        ]

        context = context_override if context_override is not None else self._gather_completed_context()
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

        response = await provider.generate(
            messages=messages,
            config=config,
        )
        output = response.content or ""
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
        if decision.new_model_id and self._provider_pool:
            # Controller already updated the node's model_id; ProviderPool will
            # resolve it on the next _execute_node call.
            pass
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
                result = await self._execute_node(node_idx, task)

                # Phase C: runtime adaptation (single-node path)
                if self._controller:
                    decision = self._controller.evaluate_and_decide(
                        node_idx, result, task, self.graph, None,
                        parallel_outputs=None,
                    )
                    if decision.action == "upgrade_model":
                        result = await self._retry_with_upgrade(node_idx, decision, task)
                        self._node_outputs[node_idx] = result
                    elif decision.action == "spawn_subagent":
                        await self._spawn_sub(node_idx, decision, task)
                    elif decision.action == "reroute_topology":
                        return "__REROUTE__"
                    # prune_node and continue: no special handling needed

                self.executor.mark_completed(node_idx)
                last_output = self._node_outputs.get(node_idx, result)
            else:
                # Snapshot context before gather to prevent race:
                # concurrent coroutines must not see each other's outputs.
                ctx_snapshot = self._gather_completed_context()
                coros = [
                    self._execute_node(idx, task, context_override=ctx_snapshot)
                    for idx in ready
                ]
                results = await asyncio.gather(*coros)

                # Phase C: runtime adaptation (parallel path)
                if self._controller:
                    parallel_outputs = list(results)
                    for idx, result in zip(ready, results):
                        decision = self._controller.evaluate_and_decide(
                            idx, result, task, self.graph, None,
                            parallel_outputs=parallel_outputs,
                        )
                        if decision.action == "upgrade_model":
                            upgraded = await self._retry_with_upgrade(idx, decision, task)
                            self._node_outputs[idx] = upgraded
                        elif decision.action == "spawn_subagent":
                            await self._spawn_sub(idx, decision, task)
                        elif decision.action == "reroute_topology":
                            return "__REROUTE__"
                        # prune_node and continue: no special handling needed

                for idx, output in zip(ready, results):
                    self.executor.mark_completed(idx)
                    last_output = self._node_outputs.get(idx, output)

        return last_output
