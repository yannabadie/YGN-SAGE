"""SageTopologyEnv — Multi-step environment for GiGPO topology training.

Implements the verl-agent gym-style interface (reset/step) where:
  Step 0: Model generates YAML topology → structural reward + anchor(prompt)
  Steps 1..N: Env executes each topology node → per-node reward + anchor(role, difficulty, context)
  Step N+1: Terminal — code extracted, tested in sandbox → execution reward

This enables GiGPO step-level advantage: different trajectories sharing the same
anchor state (same role + same context) are grouped, and rewards are normalized
within each group. This provides temporal credit assignment that flat GRPO cannot.

Reference: GiGPO (arXiv 2505.10978), verl-agent env_manager interface.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import yaml

log = logging.getLogger("topology_env")


@dataclass
class StepResult:
    """Output of one step in the episode."""
    step_idx: int
    node_idx: int
    role: str
    output: str
    reward: float
    latency: float
    anchor_key: str


@dataclass
class EpisodeTrace:
    """Complete trace of one episode (one trajectory)."""
    prompt: str
    task_id: str
    topology_yaml: str = ""
    steps: list[StepResult] = field(default_factory=list)
    final_code: str | None = None
    total_reward: float = 0.0
    status: str = ""


def _make_anchor(role: str, difficulty: str, context_hash: str) -> str:
    """Build anchor state key for GiGPO step-level grouping.

    Two steps share an anchor if they have the same:
    - role (coder, reviewer, synthesizer)
    - difficulty (simple, moderate, complex)
    - context hash (hash of predecessor outputs)

    This enables GiGPO to compare: "when a reviewer sees good coder output,
    does it produce better feedback?" across trajectories.
    """
    return f"{role}:{difficulty}:{context_hash}"


class SageTopologyEnv:
    """Gym-style environment for topology execution.

    Interface:
        reset(prompt, task_id) -> observation dict
        step(model_response) -> (observation, reward, done, info)

    Observations include 'anchor' field for GiGPO grouping.
    """

    def __init__(self, config: dict | None = None):
        self._config = config or {}
        self._trace: EpisodeTrace | None = None
        self._topo_dict: dict | None = None
        self._graph = None
        self._executor = None
        self._node_outputs: dict[int, str] = {}
        self._node_order: list[int] = []
        self._current_step = 0
        self._difficulty = "moderate"
        self._provider = None
        self._model = ""

    def reset(self, prompt: str, task_id: str = "") -> dict:
        """Start a new episode. Returns initial observation."""
        self._trace = EpisodeTrace(prompt=prompt, task_id=task_id)
        self._topo_dict = None
        self._graph = None
        self._executor = None
        self._node_outputs = {}
        self._node_order = []
        self._current_step = 0
        self._difficulty = "moderate"

        return {
            "text": prompt,
            "image": None,
            "anchor": _make_anchor("topology_generator", "unknown", hashlib.md5(prompt.encode()).hexdigest()[:8]),
        }

    def step(self, model_response: str) -> tuple[dict, float, bool, dict]:
        """Execute one step.

        Step 0: model_response = YAML topology (the main model action)
        Steps 1..N: model_response = model's observation/reaction (may be "continue")
                    The env executes the next node automatically.

        Returns: (observation, reward, done, info)
        """
        if self._current_step == 0:
            return self._step_parse_topology(model_response)
        else:
            return self._step_execute_node(model_response)

    def _step_parse_topology(self, yaml_text: str) -> tuple[dict, float, bool, dict]:
        """Step 0: parse YAML, build graph, prepare execution."""
        self._trace.topology_yaml = yaml_text

        # Parse YAML
        try:
            topo = yaml.safe_load(yaml_text)
            if not isinstance(topo, dict) or "nodes" not in topo:
                reward = -1.0
                self._trace.total_reward = reward
                self._trace.status = "INVALID_YAML"
                return {
                    "text": "Invalid topology: missing 'nodes' key.",
                    "image": None,
                    "anchor": _make_anchor("topology_generator", "unknown", "invalid"),
                }, reward, True, {"status": "INVALID_YAML"}

            nodes = topo.get("nodes", [])
            if not isinstance(nodes, list) or len(nodes) == 0:
                reward = -0.5
                self._trace.total_reward = reward
                self._trace.status = "EMPTY_TOPOLOGY"
                return {
                    "text": "Empty topology: no nodes defined.",
                    "image": None,
                    "anchor": _make_anchor("topology_generator", "unknown", "empty"),
                }, reward, True, {"status": "EMPTY_TOPOLOGY"}

        except yaml.YAMLError:
            reward = -2.0
            self._trace.total_reward = reward
            self._trace.status = "YAML_ERROR"
            return {
                "text": "YAML parse error.",
                "image": None,
                "anchor": _make_anchor("topology_generator", "unknown", "parse_error"),
            }, reward, True, {"status": "YAML_ERROR"}

        self._topo_dict = topo
        self._difficulty = topo.get("difficulty", "moderate")

        # Structural reward for step 0
        struct_score = 0.0
        if 1 <= len(nodes) <= 10:
            struct_score += 0.3
        if topo.get("edges"):
            struct_score += 0.2
        if all(isinstance(n, dict) and "role" in n for n in nodes):
            struct_score += 0.3
        if topo.get("reasoning"):
            struct_score += 0.2
        reward = struct_score

        # Record step 0
        anchor = _make_anchor("topology_generator", self._difficulty,
                              hashlib.md5(yaml_text.encode()).hexdigest()[:8])
        self._trace.steps.append(StepResult(
            step_idx=0, node_idx=-1, role="topology_generator",
            output=yaml_text, reward=reward, latency=0.0, anchor_key=anchor,
        ))

        # Build graph for execution (Rust if available, fallback to sequential)
        try:
            from sage.grpo.execution_reward import _build_topology_graph, _RUST_AVAILABLE
            if _RUST_AVAILABLE:
                topo["_task_id"] = self._trace.task_id
                self._graph = _build_topology_graph(topo)
            else:
                self._graph = None  # Will use sequential fallback
        except Exception:
            self._graph = None

        # Prepare node execution order
        if self._graph is not None:
            try:
                from sage_core import TopologyExecutor
                self._executor = TopologyExecutor(self._graph)
            except ImportError:
                self._executor = None
        else:
            # Fallback: sequential node execution without Rust graph
            self._executor = None

        self._current_step = 1

        # Next observation: show what node 0 will be
        first_node = nodes[0] if nodes else {}
        first_role = first_node.get("role", "agent") if isinstance(first_node, dict) else "agent"

        obs = {
            "text": f"Topology parsed: {len(nodes)} nodes, {self._difficulty} difficulty. "
                    f"Executing node 0 ({first_role})...",
            "image": None,
            "anchor": _make_anchor(first_role, self._difficulty, ""),
        }
        return obs, reward, False, {"status": "TOPOLOGY_PARSED", "n_nodes": len(nodes)}

    def _step_execute_node(self, model_response: str) -> tuple[dict, float, bool, dict]:
        """Steps 1..N: execute the next topology node."""
        nodes = self._topo_dict.get("nodes", [])
        node_idx = self._current_step - 1  # step 1 = node 0, step 2 = node 1, etc.

        if node_idx >= len(nodes):
            return self._finalize_episode()

        node = nodes[node_idx]
        role = node.get("role", f"node-{node_idx}") if isinstance(node, dict) else f"node-{node_idx}"
        prompt = node.get("prompt", f"You are: {role}") if isinstance(node, dict) else f"You are: {role}"

        # Build context from predecessor outputs
        context_parts = []
        for idx in sorted(self._node_outputs.keys()):
            prev_node = nodes[idx] if idx < len(nodes) else {}
            prev_role = prev_node.get("role", f"node-{idx}") if isinstance(prev_node, dict) else f"node-{idx}"
            context_parts.append(f"[{prev_role}]: {self._node_outputs[idx][:500]}")
        context = "\n\n".join(context_parts)
        context_hash = hashlib.md5(context.encode()).hexdigest()[:8] if context else ""

        # Execute node via LLM API
        t0 = time.time()
        output = ""
        try:
            output = asyncio.get_event_loop().run_until_complete(
                self._execute_node_llm(role, prompt, context, self._trace.prompt)
            )
        except RuntimeError:
            # No event loop — create one
            try:
                output = asyncio.run(
                    self._execute_node_llm(role, prompt, context, self._trace.prompt)
                )
            except Exception as exc:
                output = f"ERROR: {str(exc)[:200]}"
        except Exception as exc:
            output = f"ERROR: {str(exc)[:200]}"

        latency = time.time() - t0
        self._node_outputs[node_idx] = output

        # Per-node reward
        reward = self._compute_node_reward(node_idx, role, output)

        # Anchor state for this step
        anchor = _make_anchor(role, self._difficulty, context_hash)

        # Record step
        self._trace.steps.append(StepResult(
            step_idx=self._current_step, node_idx=node_idx, role=role,
            output=output, reward=reward, latency=latency, anchor_key=anchor,
        ))

        self._current_step += 1

        # Check if this was the last node
        if node_idx >= len(nodes) - 1:
            return self._finalize_episode()

        # Next observation
        next_node = nodes[node_idx + 1] if node_idx + 1 < len(nodes) else {}
        next_role = next_node.get("role", "agent") if isinstance(next_node, dict) else "agent"

        obs = {
            "text": f"Node {node_idx} ({role}) completed ({latency:.1f}s). Output: {output[:300]}...\n"
                    f"Executing node {node_idx + 1} ({next_role})...",
            "image": None,
            "anchor": _make_anchor(next_role, self._difficulty, context_hash),
        }
        return obs, reward, False, {"status": "NODE_COMPLETED", "node_idx": node_idx, "role": role}

    def _finalize_episode(self) -> tuple[dict, float, bool, dict]:
        """Terminal step: extract code, test, return final reward."""
        from sage.grpo.execution_reward import extract_python_code, compute_execution_score

        if not self._node_outputs:
            self._trace.total_reward = 0.0
            self._trace.status = "NO_OUTPUT"
            return {"text": "No output.", "image": None, "anchor": "terminal:none"}, 0.0, True, {"status": "NO_OUTPUT"}

        last_idx = max(self._node_outputs.keys())
        final_output = self._node_outputs[last_idx]
        code = extract_python_code(final_output)

        if code is None:
            self._trace.total_reward = sum(s.reward for s in self._trace.steps)
            self._trace.status = "NO_CODE"
            return {
                "text": "No code extracted from final node.",
                "image": None,
                "anchor": "terminal:no_code",
            }, 0.0, True, {"status": "NO_CODE"}

        self._trace.final_code = code

        # Test the code
        try:
            exec_score, status = asyncio.get_event_loop().run_until_complete(
                compute_execution_score(code, self._trace.task_id, timeout=30)
            )
        except RuntimeError:
            try:
                exec_score, status = asyncio.run(
                    compute_execution_score(code, self._trace.task_id, timeout=30)
                )
            except Exception:
                exec_score, status = 0.0, "EXEC_ERROR"
        except Exception:
            exec_score, status = 0.0, "EXEC_ERROR"

        self._trace.total_reward = sum(s.reward for s in self._trace.steps) + exec_score
        self._trace.status = status

        return {
            "text": f"Execution: {status} (score: {exec_score})",
            "image": None,
            "anchor": f"terminal:{status}",
        }, exec_score, True, {"status": status, "exec_score": exec_score}

    async def _execute_node_llm(self, role: str, prompt: str, context: str, task: str) -> str:
        """Execute a single node via LLM API using the provider pool."""
        from sage.grpo.execution_reward import _get_agent_provider
        from sage.llm.base import Message, Role as LLMRole, LLMConfig

        provider, model = _get_agent_provider()
        if provider is None:
            return f"[{role}]: No LLM provider available."

        messages = [Message(role=LLMRole.SYSTEM, content=prompt)]
        if context:
            messages.append(Message(role=LLMRole.SYSTEM, content=f"Context from previous agents:\n{context}"))
        messages.append(Message(role=LLMRole.USER, content=task[:2000]))

        try:
            response = await asyncio.wait_for(
                provider.generate(
                    messages=messages,
                    config=LLMConfig(provider="agent", model=model),
                ),
                timeout=60.0,
            )
            return response.content or ""
        except Exception as exc:
            return f"ERROR: {str(exc)[:200]}"

    def _compute_node_reward(self, node_idx: int, role: str, output: str) -> float:
        """Per-node quality reward (intermediate signal for GiGPO)."""
        from sage.grpo.execution_reward import extract_python_code

        reward = 0.0

        # Non-empty output
        if output and len(output.strip()) > 10 and not output.startswith("ERROR"):
            reward += 0.1

        # Code present (good for coder/synthesizer)
        if role in ("coder", "synthesizer", "programmer"):
            if extract_python_code(output):
                reward += 0.2
            else:
                reward -= 0.1

        # Substantive review
        if role == "reviewer" and len(output) > 50:
            reward += 0.1

        # Planner with structure
        if role == "planner" and len(output) > 100:
            reward += 0.1

        return reward

    def get_trace(self) -> EpisodeTrace:
        return self._trace
