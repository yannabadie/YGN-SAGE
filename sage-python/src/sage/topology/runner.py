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
    """

    def __init__(
        self,
        graph: Any,
        executor: Any,
        llm_provider: LLMProvider,
        llm_config: LLMConfig | None = None,
    ) -> None:
        self.graph = graph
        self.executor = executor
        self._llm = llm_provider
        self._config = llm_config
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

    async def _execute_node(self, node_idx: int, task: str) -> str:
        """Execute a single topology node as an LLM call."""
        node = self.graph.get_node(node_idx)
        role = getattr(node, "role", f"node-{node_idx}")
        caps = getattr(node, "required_capabilities", [])

        system_prompt = f"You are acting as: {role}."
        if caps:
            system_prompt += f" Your capabilities: {', '.join(caps)}."

        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=system_prompt),
        ]

        context = self._gather_completed_context()
        if context:
            messages.append(Message(
                role=Role.SYSTEM,
                content=f"Context from previous agents:\n{context}",
            ))

        messages.append(Message(role=Role.USER, content=task))

        response = await self._llm.generate(
            messages=messages,
            config=self._config,
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

    async def run(self, task: str) -> str:
        """Execute the full topology, returning the final node's output."""
        last_output = ""

        while not self.executor.is_done():
            ready = self.executor.next_ready(self.graph)
            if not ready:
                break

            if len(ready) == 1:
                last_output = await self._execute_node(ready[0], task)
                self.executor.mark_completed(ready[0])
            else:
                coros = [self._execute_node(idx, task) for idx in ready]
                results = await asyncio.gather(*coros)
                for idx, output in zip(ready, results):
                    self.executor.mark_completed(idx)
                    last_output = output

        return last_output
