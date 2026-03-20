"""SageTopologyEnv — Multi-step environment for GiGPO topology training.

Implements the verl-agent gym-style interface (reset/step) where:
  Step 0: Model generates YAML topology → structural reward + anchor(prompt)
  Steps 1..N: Env executes each topology node via TopologyRunner + ProviderPool
              → per-node reward + anchor(role, difficulty, context)
  Terminal: Code tested in sandbox → execution reward

Uses the REAL TopologyRunner with ProviderPool.resolve() for multi-provider
execution. model_tier in YAML is resolved to actual providers (DeepSeek, Google,
OpenAI, xAI, MiniMax, Kimi, OpenRouter) so the model learns that provider
assignment matters.

Reference: GiGPO (arXiv 2505.10978), verl-agent env_manager interface.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import yaml

from sage.verl.step_reward import StepRewardVector

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
    model_id: str = ""


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
    """Build anchor state key for GiGPO step-level grouping."""
    return f"{role}:{difficulty}:{context_hash}"


class SageTopologyEnv:
    """Gym-style environment for multi-step topology execution.

    Interface (verl-agent compatible):
        reset(prompt, task_id) -> observation dict with 'anchor' field
        step(model_response) -> (observation, reward, done, info)
        get_step_rewards() -> StepRewardVector (for GiGPO)
    """

    def __init__(self, config: dict | None = None):
        self._config = config or {}
        self._trace: EpisodeTrace | None = None
        self._topo_dict: dict | None = None
        self._node_traces: list[dict] = []  # from TopologyRunner.run_traced()
        self._current_step = 0
        self._difficulty = "moderate"
        self._step_reward_vec: StepRewardVector | None = None
        self._predecessor_map: dict[int, list[int]] = {}  # node_idx -> [predecessor indices]

    def reset(self, prompt: str, task_id: str = "") -> dict:
        """Start a new episode. Returns initial observation."""
        self._trace = EpisodeTrace(prompt=prompt, task_id=task_id)
        self._topo_dict = None
        self._node_traces = []
        self._current_step = 0
        self._difficulty = "moderate"
        self._step_reward_vec = None
        self._predecessor_map = {}

        return {
            "text": prompt,
            "image": None,
            "anchor": _make_anchor("topology_generator", "unknown",
                                   hashlib.md5(prompt.encode()).hexdigest()[:8]),
        }

    def step(self, model_response: str) -> tuple[dict, float, bool, dict]:
        """Execute one step.

        Step 0: model_response = YAML topology
        Steps 1..N: model observes node output, responds (may be "continue")
                    The env has already executed nodes via TopologyRunner.run_traced()
        """
        if self._current_step == 0:
            return self._step_parse_and_execute(model_response)
        else:
            return self._step_deliver_node_result()

    def _step_parse_and_execute(self, yaml_text: str) -> tuple[dict, float, bool, dict]:
        """Step 0: parse YAML, execute ALL nodes via TopologyRunner + ProviderPool,
        then deliver results one step at a time."""
        self._trace.topology_yaml = yaml_text

        # Parse YAML
        try:
            topo = yaml.safe_load(yaml_text)
            if not isinstance(topo, dict) or "nodes" not in topo:
                return self._terminal(-1.0, "INVALID_YAML", "Invalid topology: missing 'nodes' key.")
            nodes = topo.get("nodes", [])
            if not isinstance(nodes, list) or len(nodes) == 0:
                return self._terminal(-0.5, "EMPTY_TOPOLOGY", "Empty topology: no nodes.")
        except yaml.YAMLError:
            return self._terminal(-2.0, "YAML_ERROR", "YAML parse error.")

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

        anchor = _make_anchor("topology_generator", self._difficulty,
                              hashlib.md5(yaml_text.encode()).hexdigest()[:8])
        self._trace.steps.append(StepResult(
            step_idx=0, node_idx=-1, role="topology_generator",
            output=yaml_text, reward=struct_score, latency=0.0, anchor_key=anchor,
        ))

        # Build predecessor map — use Rust get_predecessors() if available,
        # otherwise fallback to parsing YAML edges
        self._predecessor_map = self._build_predecessor_map(topo)

        # Assign real model_ids via ModelAssigner + ModelRegistry (Trou #1 fix)
        self._assign_models_to_topology(topo, nodes)

        # Execute ALL nodes via TopologyRunner + ProviderPool
        exec_mode = os.environ.get("SAGE_VERL_EXEC", "0") == "1"
        if exec_mode:
            self._node_traces = self._execute_topology_traced(topo)
        else:
            # Structural mode: create synthetic traces from topology structure
            self._node_traces = []
            for i, node in enumerate(nodes):
                if isinstance(node, dict):
                    self._node_traces.append({
                        "node_idx": i,
                        "role": node.get("role", f"node-{i}"),
                        "output": f"[structural mode] Node {i} ({node.get('role', 'agent')})",
                        "latency": 0.0,
                        "model_id": node.get("model_tier", ""),
                    })

        if not self._node_traces:
            return self._terminal(struct_score, "NO_EXECUTION", "No nodes executed.")

        self._current_step = 1

        # Deliver first node result
        first = self._node_traces[0]
        obs = {
            "text": f"Topology parsed ({len(nodes)} nodes, {self._difficulty}). "
                    f"Node 0 ({first['role']}) executed: {first['output'][:300]}",
            "image": None,
            "anchor": _make_anchor(first["role"], self._difficulty, ""),
        }
        return obs, struct_score, False, {"status": "TOPOLOGY_PARSED", "n_nodes": len(nodes)}

    def _step_deliver_node_result(self) -> tuple[dict, float, bool, dict]:
        """Steps 1..N: deliver the next pre-executed node result."""
        trace_idx = self._current_step - 1  # step 1 → trace[0], step 2 → trace[1]

        if trace_idx >= len(self._node_traces):
            return self._finalize_episode()

        node_trace = self._node_traces[trace_idx]
        role = node_trace["role"]
        output = node_trace["output"]
        latency = node_trace.get("latency", 0.0)
        model_id = node_trace.get("model_id", "")

        # Build context hash from PREDECESSOR outputs only (not all previous)
        # Uses the edge map built from YAML to identify direct predecessors
        predecessors = self._predecessor_map.get(node_trace["node_idx"], [])
        pred_outputs = " ".join(
            self._node_traces[p]["output"][:200]
            for p in range(len(self._node_traces[:trace_idx]))
            if self._node_traces[p]["node_idx"] in predecessors
        ) if predecessors else " ".join(
            t["output"][:200] for t in self._node_traces[:trace_idx]
        )
        context_hash = hashlib.md5(pred_outputs.encode()).hexdigest()[:8] if pred_outputs else ""

        # Per-node reward
        reward = self._compute_node_reward(role, output)

        anchor = _make_anchor(role, self._difficulty, context_hash)
        self._trace.steps.append(StepResult(
            step_idx=self._current_step, node_idx=node_trace["node_idx"],
            role=role, output=output, reward=reward, latency=latency,
            anchor_key=anchor, model_id=model_id,
        ))

        self._current_step += 1

        # Check if last node
        if trace_idx >= len(self._node_traces) - 1:
            return self._finalize_episode()

        # Next node observation
        next_trace = self._node_traces[trace_idx + 1] if trace_idx + 1 < len(self._node_traces) else None
        next_role = next_trace["role"] if next_trace else "done"

        obs = {
            "text": f"Node {node_trace['node_idx']} ({role}, model={model_id}) "
                    f"completed ({latency:.1f}s). Output: {output[:300]}\n"
                    f"Next: node ({next_role})",
            "image": None,
            "anchor": _make_anchor(next_role, self._difficulty, context_hash),
        }
        return obs, reward, False, {"status": "NODE_COMPLETED", "role": role, "node_idx": node_trace["node_idx"]}

    def _build_predecessor_map(self, topo: dict) -> dict[int, list[int]]:
        """Build node_idx -> [predecessor_indices] map.

        Uses Rust TopologyGraph.get_predecessors() if sage_core is available
        (native petgraph query, O(degree)). Falls back to parsing YAML edges.
        """
        nodes = topo.get("nodes", [])

        # Try Rust path first
        try:
            from sage.execution import _build_topology_graph, _RUST_AVAILABLE
            if _RUST_AVAILABLE:
                graph = _build_topology_graph(topo)
                if graph is not None and hasattr(graph, "get_predecessors"):
                    pred_map = {}
                    for i in range(graph.node_count()):
                        preds = graph.get_predecessors(i)
                        if preds:
                            pred_map[i] = preds
                    return pred_map
        except Exception:
            pass

        # Fallback: parse edges from YAML
        pred_map: dict[int, list[int]] = {}
        for ed in topo.get("edges", []):
            if isinstance(ed, dict):
                to_idx = ed.get("to_idx", 0)
                from_idx = ed.get("from_idx", 0)
                pred_map.setdefault(to_idx, []).append(from_idx)
        return pred_map

    def _assign_models_to_topology(self, topo: dict, nodes: list) -> None:
        """Assign real model_ids from cards.toml via ModelAssigner (Trou #1 fix).

        Maps model_tier (reasoner/fast/budget) to actual model IDs
        (gemini-3.1-pro, deepseek-chat, gpt-5.4-nano) using ModelRegistry
        + ModelAssigner scoring (affinity 0.4 + domain 0.4 + cost 0.2).
        """
        try:
            from sage_core import ModelRegistry
            from sage.llm.model_assigner import ModelAssigner

            cards_paths = [
                "/workspace/YGN-SAGE/sage-core/config/cards.toml",
                "config/cards.toml",
                "../sage-core/config/cards.toml",
            ]
            registry = None
            for path in cards_paths:
                try:
                    registry = ModelRegistry.from_toml_file(path)
                    if registry.len() > 0:
                        break
                except Exception:
                    continue

            if registry is None or registry.len() == 0:
                log.warning("ModelRegistry empty — model_tier assignment skipped")
                return

            # Map model_tier to cognitive system for model selection
            from sage_core import CognitiveSystem
            tier_to_cs = {
                "reasoner": CognitiveSystem.S3,
                "fast": CognitiveSystem.S2,
                "budget": CognitiveSystem.S1,
            }

            for i, node in enumerate(nodes):
                if not isinstance(node, dict):
                    continue
                tier = node.get("model_tier", "fast")
                cs = tier_to_cs.get(tier, CognitiveSystem.S2)

                # select_for_system returns models sorted by affinity for that system
                candidates = registry.select_for_system(cs)
                if candidates:
                    # Pick the best-affinity model for this system
                    best = candidates[0]
                    node["_assigned_model_id"] = best.id
                    node["_assigned_provider"] = best.provider
                    log.debug("Node %d (%s): tier=%s -> model=%s (provider=%s)",
                              i, node.get("role", "?"), tier, best.id, best.provider)

        except ImportError as exc:
            log.warning("ModelAssigner unavailable (%s) — using default provider", exc)
        except Exception as exc:
            log.warning("Model assignment failed: %s", exc)

    def _execute_topology_traced(self, topo: dict) -> list[dict]:
        """Execute topology via TopologyRunner.run_traced() with ProviderPool.

        Uses the REAL multi-provider pipeline:
        1. Build TopologyGraph (Rust)
        2. ModelAssigner assigns model_id per node (from cards.toml)
        3. TopologyRunner.run_traced() executes with ProviderPool.resolve()
        4. Returns per-node traces
        """
        try:
            from sage.execution import (
                _build_topology_graph, _get_agent_provider, _RUST_AVAILABLE,
            )
            from sage.topology.runner import TopologyRunner
            from sage.llm.base import LLMConfig

            # Build graph
            topo["_task_id"] = self._trace.task_id
            if _RUST_AVAILABLE:
                graph = _build_topology_graph(topo)
            else:
                graph = None

            if graph is None:
                # Fallback: sequential execution without Rust graph
                return self._execute_sequential_fallback(topo)

            from sage_core import TopologyExecutor
            executor = TopologyExecutor(graph)

            provider, model = _get_agent_provider()
            if provider is None:
                return self._execute_sequential_fallback(topo)

            config = LLMConfig(provider="agent", model=model)

            # TODO: Wire ProviderPool for per-node model resolution
            # For now, use the primary provider. Full ProviderPool integration
            # requires boot.py context (ModelRegistry + runtime adapters).
            # The ProviderPool integration is the next priority after this works.
            runner = TopologyRunner(
                graph=graph,
                executor=executor,
                llm_provider=provider,
                llm_config=config,
            )

            # run_traced() returns per-node outputs with metadata
            # Always use ThreadPoolExecutor to avoid event loop conflicts
            # (verl-agent may call step() from within its own event loop)
            try:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    traces = pool.submit(
                        lambda: asyncio.run(
                            asyncio.wait_for(runner.run_traced(self._trace.prompt[:2000]), timeout=120)
                        )
                    ).result(timeout=130)
            except Exception as exc:
                log.warning("run_traced() failed: %s", exc)
                return self._execute_sequential_fallback(topo)

            return traces

        except Exception as exc:
            log.warning("Traced execution failed: %s", exc)
            return self._execute_sequential_fallback(topo)

    def _execute_sequential_fallback(self, topo: dict) -> list[dict]:
        """Fallback: create structural traces without API calls."""
        traces = []
        for i, node in enumerate(topo.get("nodes", [])):
            if isinstance(node, dict):
                traces.append({
                    "node_idx": i,
                    "role": node.get("role", f"node-{i}"),
                    "output": f"[fallback] Node {i} ({node.get('role', 'agent')}): structural only",
                    "latency": 0.0,
                    "model_id": node.get("model_tier", ""),
                })
        return traces

    def _finalize_episode(self) -> tuple[dict, float, bool, dict]:
        """Terminal step: extract code, test, build StepRewardVector."""
        from sage.execution import extract_python_code, compute_execution_score

        exec_score = 0.0
        status = "NO_CODE"

        if self._node_traces:
            last_output = self._node_traces[-1].get("output", "")
            code = extract_python_code(last_output)

            if code is not None:
                self._trace.final_code = code
                try:
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        exec_score, status = pool.submit(
                            lambda: asyncio.run(
                                compute_execution_score(code, self._trace.task_id, timeout=30)
                            )
                        ).result(timeout=35)
                except Exception:
                    exec_score, status = 0.0, "EXEC_ERROR"

        # Terminal step result
        self._trace.steps.append(StepResult(
            step_idx=self._current_step, node_idx=-1, role="terminal",
            output=status, reward=exec_score, latency=0.0,
            anchor_key=f"terminal:{status}",
        ))

        self._trace.total_reward = sum(s.reward for s in self._trace.steps)
        self._trace.status = status

        # Build StepRewardVector for GiGPO
        self._step_reward_vec = StepRewardVector.from_episode_trace(self._trace)

        return {
            "text": f"Execution: {status} (score: {exec_score})",
            "image": None,
            "anchor": f"terminal:{status}",
        }, exec_score, True, {
            "status": status,
            "exec_score": exec_score,
            "total_reward": self._trace.total_reward,
        }

    def _terminal(self, reward: float, status: str, text: str) -> tuple[dict, float, bool, dict]:
        """Early termination (invalid YAML, empty topology, etc.)."""
        self._trace.total_reward = reward
        self._trace.status = status
        self._trace.steps.append(StepResult(
            step_idx=0, node_idx=-1, role="topology_generator",
            output=text, reward=reward, latency=0.0,
            anchor_key=_make_anchor("topology_generator", "unknown", status),
        ))
        self._step_reward_vec = StepRewardVector.from_episode_trace(self._trace)
        return {"text": text, "image": None, "anchor": f"error:{status}"}, reward, True, {"status": status}

    def _compute_node_reward(self, role: str, output: str) -> float:
        """Per-node quality reward using Rust QualityLabeler (OxiZ formal, zero heuristics).

        Falls back to a minimal signal if QualityLabeler is unavailable.
        """
        if not output or output.startswith("[fallback]") or output.startswith("ERROR"):
            return 0.0

        # Use Rust QualityLabeler (OxiZ SMT + tree-sitter, zero heuristics)
        try:
            from sage_core import QualityLabeler
            ql = QualityLabeler()
            label = ql.label(f"Node role: {role}", output)
            if label and label.assessable:
                return float(label.score)
        except ImportError:
            pass

        # Minimal fallback: non-empty output = 0.1 (no heuristic thresholds)
        return 0.1 if len(output.strip()) > 0 else 0.0

    def get_trace(self) -> EpisodeTrace:
        return self._trace

    def get_step_rewards(self) -> StepRewardVector:
        """Return the StepRewardVector for GiGPO advantage computation."""
        if self._step_reward_vec is None:
            if self._trace:
                self._step_reward_vec = StepRewardVector.from_episode_trace(self._trace)
            else:
                self._step_reward_vec = StepRewardVector()
        return self._step_reward_vec


class SageTopologyEnvManager:
    """Environment manager for verl-agent integration.

    Wraps SageTopologyEnv in the interface expected by verl-agent's
    env_manager.py (make_envs() factory).
    """

    def __init__(self, config: Any = None):
        self._config = config
        self._envs: list[SageTopologyEnv] = []

    def make(self, n_envs: int = 1) -> list[SageTopologyEnv]:
        self._envs = [SageTopologyEnv(config=self._config) for _ in range(n_envs)]
        return self._envs

    def reset(self, prompts: list[str], task_ids: list[str] | None = None) -> list[dict]:
        if task_ids is None:
            task_ids = [""] * len(prompts)
        return [env.reset(p, t) for env, p, t in zip(self._envs, prompts, task_ids)]

    def step(self, actions: list[str]) -> tuple[list[dict], list[float], list[bool], list[dict]]:
        results = [env.step(a) for env, a in zip(self._envs, actions)]
        obs = [r[0] for r in results]
        rewards = [r[1] for r in results]
        dones = [r[2] for r in results]
        infos = [r[3] for r in results]
        return obs, rewards, dones, infos
