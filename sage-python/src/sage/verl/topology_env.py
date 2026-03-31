"""SageTopologyEnv — Multi-step environment for topology training.

Implements the verl-agent gym-style interface (reset/step) with a 4-state machine:
  AWAITING_YAML → EXECUTING → AWAITING_DECISION → EXECUTING → ... → TERMINAL

The model makes REAL decisions at checkpoint nodes:
  Step 0: Model generates YAML topology → structural reward + anchor(prompt)
  Checkpoint steps: Model decides continue/upgrade/reroute → step-level advantage
  Terminal: Code tested in sandbox → execution reward

IMPORTANT: This environment requires verl-agent (not vanilla verl 0.7.1).
Current training scripts (V3/V5) use vanilla verl with GRPO, NOT this env.
This env is only used when verl-agent is installed and GiGPO multi-step
training is active (Phase C / train_topology.sh, NOT train_topology_v5.sh).

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

# ---------------------------------------------------------------------------
# Reward constants for micro-decisions
# ---------------------------------------------------------------------------
_REWARD_UPGRADE_COST = -0.05
_REWARD_REROUTE_PENALTY = -0.3
_REWARD_UPGRADE_SUCCESS = 0.15


def _get_embedding(text: str) -> "np.ndarray":
    """Compute embedding via Embedder, fallback to zeros."""
    import numpy as np
    try:
        from sage.memory.embedder import Embedder
        embedder = Embedder()
        if embedder.is_semantic:
            emb = embedder.embed(text[:500])
            if emb is not None and len(emb) == 768:
                return np.array(emb, dtype=np.float32)
    except Exception:
        pass
    return np.zeros(768, dtype=np.float32)


def _quality_bucket(quality: float, threshold: float) -> str:
    if quality < threshold * 0.6:
        return "very_low"
    elif quality < threshold:
        return "low"
    elif quality < threshold * 1.4:
        return "adequate"
    else:
        return "high"


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
    action: str = ""            # text of the model's action
    was_upgraded: bool = False
    quality_before: float = 0.0
    quality_after: float = 0.0


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
    node_traces_for_rewardflow: list[dict] = field(default_factory=list)


def _make_anchor(role: str, difficulty: str, context_hash: str) -> str:
    """Build anchor state key for GiGPO step-level grouping."""
    return f"{role}:{difficulty}:{context_hash}"


class SageTopologyEnv:
    """Gym-style environment for multi-step topology execution.

    4-state machine: awaiting_yaml → executing → awaiting_decision → terminal

    Interface (verl-agent compatible):
        reset(prompt, task_id) -> observation dict with 'anchor' field
        step(model_response) -> (observation, reward, done, info)
        get_step_rewards() -> StepRewardVector (for GiGPO)
    """

    _VERL_AGENT_AVAILABLE: bool | None = None

    @classmethod
    def _check_verl_agent(cls) -> bool:
        """Check if verl-agent is installed. Cached after first call."""
        if cls._VERL_AGENT_AVAILABLE is None:
            try:
                import agent_system.environments.env_manager  # noqa: F401
                cls._VERL_AGENT_AVAILABLE = True
            except ImportError:
                cls._VERL_AGENT_AVAILABLE = False
        return cls._VERL_AGENT_AVAILABLE

    def __init__(self, config: dict | None = None):
        # Info: log which mode is active. SageTopologyEnv works in two modes:
        # 1. Direct use (train_phase_c_custom.py) — no verl-agent needed
        # 2. Via env_register.py + verl-agent — requires agent_system package
        # The guard in env_register.py handles case 2. Direct use is always allowed.
        if not self._check_verl_agent():
            log.info(
                "SageTopologyEnv: verl-agent not installed. "
                "Direct use (train_phase_c_custom.py) works. "
                "verl-agent integration (train_topology_phase_c.sh) requires: "
                "pip install -e /workspace/verl-agent"
            )
        self._config = config or {}
        self._trace: EpisodeTrace | None = None
        self._topo_dict: dict | None = None
        self._node_traces: list[dict] = []  # from incremental execution
        self._difficulty = "moderate"
        self._step_reward_vec: StepRewardVector | None = None
        self._predecessor_map: dict[int, list[int]] = {}
        # V2: Adaptive topology state
        self._memory: Any = None
        self._checkpoints: set = set()
        self._max_upgrades = 0
        self._quality_threshold = 0.5
        # V3: Micro-decision state machine
        self._state = "awaiting_yaml"
        self._exec_cursor = 0
        self._pending_checkpoint = None
        self._upgrades_used = 0
        self._node_outputs: dict[int, str] = {}
        # Legacy compat field (kept for test_reset_clears_v2_state)
        self._awaiting_decision = False
        # Initialize TrainingMemory if configured
        db = self._config.get("memory_db", "")
        if db:
            try:
                from sage.verl.training_memory import TrainingMemory
                self._memory = TrainingMemory(db_path=db)
            except Exception:
                pass

    def reset(self, prompt: str, task_id: str = "") -> dict:
        """Start a new episode. Returns initial observation."""
        self._trace = EpisodeTrace(prompt=prompt, task_id=task_id)
        self._topo_dict = None
        self._node_traces = []
        self._difficulty = "moderate"
        self._step_reward_vec = None
        self._predecessor_map = {}
        self._checkpoints = set()
        self._max_upgrades = 0
        self._quality_threshold = 0.5
        # V3: Reset state machine
        self._state = "awaiting_yaml"
        self._exec_cursor = 0
        self._pending_checkpoint = None
        self._upgrades_used = 0
        self._node_outputs = {}
        # Legacy compat
        self._awaiting_decision = False

        # V2: Query episodic memory for similar past episodes
        memory_ctx = ""
        if self._memory:
            try:
                query_emb = _get_embedding(prompt)
                episodes = self._memory.query_similar(query_emb, k=3)
                memory_ctx = self._memory.format_context(episodes)
            except Exception:
                pass

        obs_text = prompt
        if memory_ctx:
            obs_text = prompt + "\n\n" + memory_ctx

        return {
            "text": obs_text,
            "image": None,
            "anchor": _make_anchor("topology_generator", "unknown",
                                   hashlib.md5(prompt.encode()).hexdigest()[:8]),
        }

    # ------------------------------------------------------------------
    # step() dispatcher — 4-state machine
    # ------------------------------------------------------------------

    def step(self, model_response: str) -> tuple[dict, float, bool, dict]:
        """Execute one step in the 4-state machine.

        awaiting_yaml: model_response = YAML topology
        awaiting_decision: model_response = continue/upgrade/reroute
        terminal: finalize episode
        executing: should not happen externally, drive execution internally
        """
        if self._state == "awaiting_yaml":
            return self._handle_yaml(model_response)
        elif self._state == "awaiting_decision":
            return self._handle_decision(model_response)
        elif self._state == "terminal":
            return self._finalize_episode()
        else:
            # _state == "executing": should not happen from outside
            return self._execute_until_checkpoint_or_end()

    # ------------------------------------------------------------------
    # _handle_yaml — replaces _step_parse_and_execute
    # ------------------------------------------------------------------

    def _handle_yaml(self, yaml_text: str) -> tuple[dict, float, bool, dict]:
        """Step 0: parse YAML/JSON, setup incremental execution, run until
        first checkpoint or end."""
        self._trace.topology_yaml = yaml_text

        # Parse topology from model output
        # Supports: <tool_call>{...}</tool_call>, raw JSON, YAML
        import json
        import re
        topo = None

        # Try <tool_call> format first (Qwen3-4B local model)
        tc_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', yaml_text, re.DOTALL)
        if not tc_match:
            tc_match = re.search(r'<tool_call>\s*(\{.*)', yaml_text, re.DOTALL)
        if tc_match:
            try:
                call = json.loads(tc_match.group(1))
                if call.get("name") == "create_topology" and "arguments" in call:
                    topo = call["arguments"]
                elif "nodes" in call:
                    topo = call
            except (json.JSONDecodeError, ValueError):
                pass

        # Try raw JSON
        if topo is None:
            try:
                topo = json.loads(yaml_text.strip())
                if not isinstance(topo, dict) or "nodes" not in topo:
                    topo = None
            except (json.JSONDecodeError, ValueError):
                pass

        # Fall back to YAML
        if topo is None:
            try:
                topo = yaml.safe_load(yaml_text)
                if not isinstance(topo, dict) or "nodes" not in topo:
                    return self._terminal(-1.0, "INVALID_YAML", "Invalid topology: missing 'nodes' key.")
            except yaml.YAMLError:
                return self._terminal(-2.0, "YAML_ERROR", "YAML parse error.")

        nodes = topo.get("nodes", [])
        if not isinstance(nodes, list) or len(nodes) == 0:
            return self._terminal(-0.5, "EMPTY_TOPOLOGY", "Empty topology: no nodes.")

        self._topo_dict = topo
        self._difficulty = topo.get("difficulty", "moderate")

        # Parse adaptation metadata for checkpoints
        adaptation = topo.get("adaptation", {})
        if isinstance(adaptation, dict):
            self._checkpoints = set(adaptation.get("checkpoints", []))
            self._max_upgrades = adaptation.get("max_upgrades", 0)
            self._quality_threshold = adaptation.get("quality_threshold", 0.5)

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

        # Build predecessor map
        self._predecessor_map = self._build_predecessor_map(topo)

        # Assign real model_ids via ModelAssigner + ModelRegistry
        self._assign_models_to_topology(topo, nodes)

        # DO NOT execute all nodes at once. Start incremental execution.
        self._state = "executing"
        self._exec_cursor = 0
        return self._execute_until_checkpoint_or_end()

    # ------------------------------------------------------------------
    # _execute_until_checkpoint_or_end — incremental node execution
    # ------------------------------------------------------------------

    def _execute_until_checkpoint_or_end(self) -> tuple[dict, float, bool, dict]:
        """Execute nodes one by one from _exec_cursor.

        Pauses at checkpoint nodes to ask the model for a decision.
        """
        nodes = self._topo_dict.get("nodes", [])

        while self._exec_cursor < len(nodes):
            node_idx = self._exec_cursor
            node = nodes[node_idx]

            # Execute THIS node (one at a time)
            trace = self._execute_single_node(node_idx, node)
            self._node_traces.append(trace)
            self._node_outputs[node_idx] = trace["output"]

            # Per-node reward
            role = trace["role"]
            reward = self._compute_node_reward(role, trace["output"])

            # Build anchor from predecessor context
            predecessors = self._predecessor_map.get(node_idx, [])
            pred_text = " ".join(self._node_outputs.get(p, "")[:200] for p in predecessors)
            context_hash = hashlib.md5(pred_text.encode()).hexdigest()[:8] if pred_text else ""
            anchor = _make_anchor(role, self._difficulty, context_hash)

            # Record the step
            self._trace.steps.append(StepResult(
                step_idx=len(self._trace.steps),
                node_idx=node_idx,
                role=role,
                output=trace["output"],
                reward=reward,
                latency=trace.get("latency", 0.0),
                anchor_key=anchor,
                model_id=trace.get("model_id", ""),
            ))

            self._exec_cursor += 1

            # If this node is a checkpoint
            if node_idx in self._checkpoints:
                quality = self._estimate_quality(trace["output"], role)

                # Store pending checkpoint
                self._pending_checkpoint = {
                    "node_idx": node_idx,
                    "role": role,
                    "quality": quality,
                    "output": trace["output"][:300],
                    "model_tier": node.get("model_tier", ""),
                    "fallback_tier": node.get("fallback_tier", ""),
                }
                self._state = "awaiting_decision"

                # Build observation with quality bucket in anchor
                q_bucket = _quality_bucket(quality, self._quality_threshold)
                decision_anchor = _make_anchor(
                    f"decision:{role}", self._difficulty,
                    f"{q_bucket}:{context_hash}"
                )

                remaining_upgrades = self._max_upgrades - self._upgrades_used
                has_fallback = bool(node.get("fallback_tier", ""))

                obs_text = (
                    f"[CHECKPOINT] Node {node_idx} ({role}, {node.get('model_tier', '?')}) completed.\n"
                    f"Output quality: {quality:.2f} (threshold: {self._quality_threshold})\n"
                    f"Output preview: {trace['output'][:200]}\n"
                )
                if has_fallback and remaining_upgrades > 0:
                    obs_text += (
                        f"Fallback available: {node['fallback_tier']}\n"
                        f"Upgrades remaining: {remaining_upgrades}/{self._max_upgrades}\n"
                        f"Actions: [continue] [upgrade] [reroute]\n"
                    )
                else:
                    obs_text += "No fallback available or no upgrades remaining.\nActions: [continue] [reroute]\n"

                return (
                    {"text": obs_text, "image": None, "anchor": decision_anchor},
                    reward,
                    False,
                    {"status": "CHECKPOINT", "node_idx": node_idx, "quality": quality},
                )

        # All nodes executed → terminal
        self._state = "terminal"
        return self._finalize_episode()

    # ------------------------------------------------------------------
    # _handle_decision — parse continue/upgrade/reroute
    # ------------------------------------------------------------------

    def _handle_decision(self, model_response: str) -> tuple[dict, float, bool, dict]:
        """Parse the model's decision at a checkpoint and act on it."""
        decision = self._parse_decision(model_response)
        cp = self._pending_checkpoint
        self._pending_checkpoint = None

        node_idx = cp["node_idx"]
        role = cp["role"]
        reward = 0.0

        if decision == "upgrade" and cp["fallback_tier"] and self._upgrades_used < self._max_upgrades:
            # Re-execute the node with the fallback_tier
            self._upgrades_used += 1

            # Modify the model_tier of the node in topo_dict
            node = self._topo_dict["nodes"][node_idx]
            original_tier = node.get("model_tier", "")
            node["model_tier"] = cp["fallback_tier"]

            # Re-assign the real model
            self._assign_models_to_topology(self._topo_dict, self._topo_dict["nodes"])

            # Re-execute
            new_trace = self._execute_single_node(node_idx, node)
            self._node_outputs[node_idx] = new_trace["output"]

            # Update trace
            new_quality = self._estimate_quality(new_trace["output"], role)
            quality_improved = new_quality > cp["quality"]

            reward = _REWARD_UPGRADE_COST  # cost of the upgrade
            if quality_improved:
                reward += _REWARD_UPGRADE_SUCCESS

            # Record the upgrade step
            self._trace.steps.append(StepResult(
                step_idx=len(self._trace.steps),
                node_idx=node_idx,
                role=f"upgrade:{role}",
                output=new_trace["output"],
                reward=reward,
                latency=new_trace.get("latency", 0.0),
                anchor_key=_make_anchor(f"upgrade:{role}", self._difficulty, ""),
                model_id=new_trace.get("model_id", ""),
                action="upgrade",
                was_upgraded=True,
                quality_before=cp["quality"],
                quality_after=new_quality,
            ))

            obs_text = (
                f"Node {node_idx} upgraded {original_tier}\u2192{cp['fallback_tier']}. "
                f"Quality: {cp['quality']:.2f}\u2192{new_quality:.2f}. "
                f"Continuing execution."
            )

        elif decision == "reroute":
            reward = _REWARD_REROUTE_PENALTY
            self._trace.steps.append(StepResult(
                step_idx=len(self._trace.steps),
                node_idx=node_idx,
                role="reroute",
                output="REROUTE",
                reward=reward,
                latency=0.0,
                anchor_key=_make_anchor("reroute", self._difficulty, ""),
                action="reroute",
            ))
            self._state = "terminal"
            self._trace.status = "REROUTED"
            return self._finalize_episode()

        else:  # "continue"
            reward = 0.0
            self._trace.steps.append(StepResult(
                step_idx=len(self._trace.steps),
                node_idx=node_idx,
                role=f"continue:{role}",
                output="continue",
                reward=reward,
                latency=0.0,
                anchor_key=_make_anchor(f"decision:{role}", self._difficulty, "continue"),
                action="continue",
            ))
            obs_text = f"Continuing with node {node_idx} output as-is."

        # Resume execution
        self._state = "executing"
        return self._execute_until_checkpoint_or_end()

    # ------------------------------------------------------------------
    # _execute_single_node — one node at a time
    # ------------------------------------------------------------------

    # ── Multi-provider tier→provider mapping ──
    # Maps model_tier from training data to real provider configs.
    # Each tier maps to a different provider for true multi-provider execution.
    _TIER_PROVIDER_MAP = {
        "budget": "deepseek",      # $0.28/M — cheapest
        "fast": "google",          # Gemini Flash — lowest latency
        "balanced": "xai",         # Grok — 2M context, mid-tier
        "reasoner": "openai",      # GPT-5.4 — strong reasoning
        "codex": "openai",         # GPT-5.4 — best coding
    }
    # Fallback order if preferred provider unavailable
    _PROVIDER_FALLBACK = ["deepseek", "google", "openai", "xai", "kimi", "minimax", "openrouter"]

    def _resolve_provider_for_tier(self, model_tier: str):
        """Resolve model_tier to (provider, model_id, base_url, api_key)."""
        from sage.providers.connector import PROVIDER_CONFIGS
        import os

        # Preferred provider for this tier
        preferred = self._TIER_PROVIDER_MAP.get(model_tier, "deepseek")
        order = [preferred] + [p for p in self._PROVIDER_FALLBACK if p != preferred]

        for prov_name in order:
            cfg = next((c for c in PROVIDER_CONFIGS if c["provider"] == prov_name), None)
            if cfg is None:
                continue
            api_key = os.environ.get(cfg["api_key_env"], "")
            if not api_key:
                continue
            model_id = cfg.get("default_model", "")
            return cfg["provider"], model_id, cfg["base_url"], api_key

        return None, None, None, None

    def _execute_single_node(self, node_idx: int, node_dict: dict) -> dict:
        """Execute a single node — MULTI-PROVIDER based on model_tier."""
        role = node_dict.get("role", f"node-{node_idx}")
        exec_mode = os.environ.get("SAGE_VERL_EXEC", "0") == "1"

        if not exec_mode:
            return {
                "node_idx": node_idx,
                "role": role,
                "output": f"[structural mode] Node {node_idx} ({role})",
                "latency": 0.0,
                "model_id": node_dict.get("model_tier", ""),
            }

        # Real execution: resolve tier → provider
        model_tier = node_dict.get("model_tier", "budget")
        try:
            prov_name, model_id, base_url, api_key = self._resolve_provider_for_tier(model_tier)
            if prov_name is None:
                log.warning("No provider for tier=%s, structural fallback", model_tier)
                return self._structural_stub(node_idx, role, node_dict)

            from sage.providers.openai_compat import OpenAICompatProvider
            from sage.llm.base import LLMConfig, Message, Role

            provider = OpenAICompatProvider(
                api_key=api_key, base_url=base_url, provider_name=prov_name,
            )

            # Build prompt from role + predecessor context
            predecessors = self._predecessor_map.get(node_idx, [])
            context = "\n\n".join(
                f"[{self._topo_dict['nodes'][p].get('role', f'node-{p}')}]: "
                f"{self._node_outputs.get(p, '')[:500]}"
                for p in predecessors if p in self._node_outputs
            )

            custom_prompt = node_dict.get("prompt", f"You are acting as: {role}")
            messages = [
                Message(role=Role.SYSTEM, content=custom_prompt),
            ]
            if context:
                messages.append(Message(role=Role.SYSTEM, content=f"Context from previous agents:\n{context}"))
            messages.append(Message(role=Role.USER, content=self._trace.prompt[:2000]))

            config = LLMConfig(provider=prov_name, model=model_id)

            t0 = time.time()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                response = pool.submit(
                    lambda: asyncio.run(provider.generate(messages=messages, config=config))
                ).result(timeout=60)
            output = response.content or ""
            latency = time.time() - t0

            log.debug("Node %d (%s) tier=%s → %s/%s (%.1fs)",
                      node_idx, role, model_tier, prov_name, model_id, latency)

            return {
                "node_idx": node_idx,
                "role": role,
                "output": output,
                "latency": latency,
                "model_id": f"{prov_name}/{model_id}",
                "provider": prov_name,
                "cost": latency * 0.00002 if prov_name == "openai" else latency * 0.000005,
            }
        except Exception as exc:
            log.warning("Node %d (%s) tier=%s exec failed: %s", node_idx, role, model_tier, exc)
            return self._structural_stub(node_idx, role, node_dict)

    def _structural_stub(self, node_idx: int, role: str, node_dict: dict) -> dict:
        return {
            "node_idx": node_idx,
            "role": role,
            "output": f"[fallback] Node {node_idx} ({role}): structural only",
            "latency": 0.0,
            "model_id": node_dict.get("model_tier", ""),
        }

    # ------------------------------------------------------------------
    # _estimate_quality / _parse_decision
    # ------------------------------------------------------------------

    def _estimate_quality(self, output: str, role: str) -> float:
        """Estimate output quality for checkpoint decision."""
        # Try Rust QualityLabeler
        try:
            from sage_core import QualityLabeler
            ql = QualityLabeler()
            label = ql.label(f"Node role: {role}", output)
            if label and label.assessable:
                return float(label.score)
        except ImportError:
            pass

        # Minimal heuristic for structural mode
        if not output or output.startswith("[structural") or output.startswith("[fallback"):
            return 0.5  # neutral in structural mode
        if output.startswith("ERROR"):
            return 0.1
        # Presence of code = better quality
        if "```" in output or "def " in output:
            return 0.7
        return 0.4

    def _parse_decision(self, text: str) -> str:
        text = text.strip()
        import json
        import re

        # <tool_call> format: {"name": "adapt_topology", "arguments": {"action": "upgrade"}}
        tc_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
        if not tc_match:
            tc_match = re.search(r'<tool_call>\s*(\{.*)', text, re.DOTALL)
        if tc_match:
            try:
                call = json.loads(tc_match.group(1))
                args = call.get("arguments", call)
                return args.get("action", "continue")
            except (json.JSONDecodeError, ValueError):
                pass

        # Raw JSON format: {"action": "continue"}
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data.get("action", data.get("arguments", {}).get("action", "continue"))
        except Exception:
            pass

        # Text fallback
        t = text.lower()
        if "upgrade" in t:
            return "upgrade"
        elif "reroute" in t:
            return "reroute"
        return "continue"

    # ------------------------------------------------------------------
    # Methods kept intact from original
    # ------------------------------------------------------------------

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
                gate = ed.get("gate", "open")
                if gate == "conditional":
                    gate = "open"
                pred_map.setdefault(to_idx, []).append(from_idx)
        return pred_map

    def _assign_models_to_topology(self, topo: dict, nodes: list) -> None:
        """Assign real model_ids from cards.toml via ModelAssigner.

        Maps model_tier (reasoner/fast/budget) to actual model IDs
        using ModelRegistry + ModelAssigner scoring.
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

                candidates = registry.select_for_system(cs)
                if candidates:
                    best = candidates[0]
                    node["_assigned_model_id"] = best.id
                    node["_assigned_provider"] = best.provider
                    log.debug("Node %d (%s): tier=%s -> model=%s (provider=%s)",
                              i, node.get("role", "?"), tier, best.id, best.provider)

        except ImportError as exc:
            log.warning("ModelAssigner unavailable (%s) — using default provider", exc)
        except Exception as exc:
            log.warning("Model assignment failed: %s", exc)

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

        # Use REROUTED status if set by _handle_decision
        if self._trace.status == "REROUTED":
            status = "REROUTED"

        # Resilience bonus for successful upgrades
        n_upgrades = sum(1 for s in self._trace.steps if s.was_upgraded)
        if n_upgrades > 0:
            any_succeeded = any(
                s.was_upgraded and s.quality_after > s.quality_before
                for s in self._trace.steps
            )
            if any_succeeded and status == "PASSED":
                resilience_bonus = 0.5
            elif any_succeeded:
                resilience_bonus = 0.3
            else:
                resilience_bonus = 0.0
            self._trace.steps.append(StepResult(
                step_idx=len(self._trace.steps), node_idx=-1, role="resilience_bonus",
                output=f"upgrades={n_upgrades}, bonus={resilience_bonus}",
                reward=resilience_bonus, latency=0.0,
                anchor_key="resilience:bonus",
            ))

        # Terminal step result
        self._trace.steps.append(StepResult(
            step_idx=len(self._trace.steps), node_idx=-1, role="terminal",
            output=status, reward=exec_score, latency=0.0,
            anchor_key=f"terminal:{status}",
        ))

        self._trace.total_reward = sum(s.reward for s in self._trace.steps)
        if self._trace.status != "REROUTED":
            self._trace.status = status

        # V2: Store episode in episodic memory for future reference
        if self._memory:
            try:
                self._memory.store_episode(
                    task_id=self._trace.task_id,
                    prompt_hash=hashlib.md5(self._trace.prompt.encode()).hexdigest()[:8],
                    domain="code",
                    topology_yaml=self._trace.topology_yaml[:2000],
                    n_nodes=len(self._node_traces),
                    difficulty=self._difficulty,
                    outcome=status,
                    total_reward=self._trace.total_reward,
                    per_node_results=[
                        {"role": t.get("role", ""), "reward": t.get("reward", 0)}
                        for t in self._node_traces
                    ],
                    adaptations_triggered=sum(
                        1 for t in self._node_traces if t.get("was_upgraded", False)
                    ),
                    embedding=_get_embedding(self._trace.prompt),
                )
            except Exception:
                pass

        # RewardFlow per-node credit (if multiple rollouts available)
        # Applied at batch level in the training loop, not per-episode
        # Store the node traces for later batch-level processing
        self._trace.node_traces_for_rewardflow = [
            {
                "node_idx": t["node_idx"],
                "role": t.get("role", "agent"),
                "quality": self._compute_node_reward(t.get("role", "agent"), t.get("output", "")),
            }
            for t in self._node_traces
        ]

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
