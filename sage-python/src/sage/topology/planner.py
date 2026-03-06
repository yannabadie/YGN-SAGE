"""Topology Planner: Self-Generating Agent Topology Engine (OpenSage SOTA).

Implements Stochastic Differentiable Tree Search (S-DTS) to dynamically 
explore, construct, and evaluate agent DAG configurations.
"""
from __future__ import annotations

import json
import math
import logging
from typing import Dict

from sage.topology.engine import TopologyEngine, TopologyType
from sage.llm.base import LLMProvider, Message, Role

class StochasticDTS:
    """Stochastic Differentiable Tree Search for topological exploration.
    
    Dynamically balances between exploring new agent DAG structures and 
    exploiting known high-reward configurations using UCT-like formulas.
    """
    def __init__(self, exploration_constant: float = 1.414):
        self.exploration_constant = exploration_constant
        # State key -> {"visits": int, "reward": float, "topologies": List[dict]}
        self.tree: Dict[str, dict] = {}
        self.logger = logging.getLogger(__name__)

    def _get_state_key(self, task: str) -> str:
        # Simplified state key for the ADK
        return task.lower()[:50]

    def select_node(self, task: str) -> str | None:
        key = self._get_state_key(task)
        if key not in self.tree or not self.tree[key]["topologies"]:
            return None
            
        node_data = self.tree[key]
        topologies = node_data["topologies"]
        
        # UCT selection
        best_score = -float('inf')
        best_topo_idx = 0
        total_visits = sum(t["visits"] for t in topologies) + 1
        
        for idx, topo in enumerate(topologies):
            if topo["visits"] == 0:
                score = float('inf')
            else:
                exploitation = topo["reward"] / topo["visits"]
                exploration = self.exploration_constant * math.sqrt(math.log(total_visits) / topo["visits"])
                score = exploitation + exploration
                
            if score > best_score:
                best_score = score
                best_topo_idx = idx
                
        # Return the JSON string definition of the selected topology
        return topologies[best_topo_idx]["definition"]

    def backpropagate(self, task: str, topo_def: str, reward: float):
        key = self._get_state_key(task)
        if key not in self.tree:
            self.tree[key] = {"visits": 0, "reward": 0.0, "topologies": []}
            
        node_data = self.tree[key]
        node_data["visits"] += 1
        node_data["reward"] += reward
        
        # Find or add the topology
        for topo in node_data["topologies"]:
            if topo["definition"] == topo_def:
                topo["visits"] += 1
                topo["reward"] += reward
                return
                
        node_data["topologies"].append({
            "definition": topo_def,
            "visits": 1,
            "reward": reward
        })


class TopologyPlanner:
    """Generates agent topologies dynamically using LLMs and S-DTS (OpenSage)."""

    def __init__(self, engine: TopologyEngine, llm: LLMProvider):
        self.engine = engine
        self.llm = llm
        self.sdts = StochasticDTS()
        self.logger = logging.getLogger(__name__)

    async def generate_topology(self, task: str, max_nodes: int = 5) -> str:
        """Dynamically create a topology for a given task.
        
        Returns the created topology_id.
        """
        # 1. SDTS Selection (Prior knowledge)
        best_known_def = self.sdts.select_node(task)
        
        prompt = f"""You are the OpenSage Topology Planner.
Your job is to decompose the following complex task into a Multi-Agent DAG (Directed Acyclic Graph).
Task: {task}
Max Nodes: {max_nodes}

You must output ONLY a valid JSON object defining the topology. Do not include markdown blocks.
Format:
{{
    "type": "vertical" | "horizontal" | "mesh",
    "nodes": [
        {{"id": "agent1", "name": "Planner", "role": "coordinator"}},
        {{"id": "agent2", "name": "Coder", "role": "worker"}}
    ],
    "edges": [
        {{"source": "agent1", "target": "agent2", "type": "delegates_to"}}
    ]
}}"""
        if best_known_def:
             prompt += f"\nConsider varying or improving this baseline architecture:\n{best_known_def}"

        messages = [Message(role=Role.USER, content=prompt)]
        response = await self.llm.generate(messages)
        
        try:
            # Clean possible markdown ticks
            content = response.content.replace("```json", "").replace("```", "").strip()
            topo_def = json.loads(content)
            
            topo_type_str = topo_def.get("type", "vertical").upper()
            try:
                topo_type = TopologyType[topo_type_str]
            except KeyError:
                topo_type = TopologyType.VERTICAL
                
            topo_id = self.engine.create_topology(topo_type)
            
            # Add nodes
            node_map = {}
            for n in topo_def.get("nodes", []):
                # The engine generates a new uuid, but we map it from the LLM's temporary id
                real_id = self.engine.add_node(topo_id, name=n["name"], role=n["role"])
                node_map[n["id"]] = real_id
                
            # Add edges
            for e in topo_def.get("edges", []):
                src = node_map.get(e["source"])
                tgt = node_map.get(e["target"])
                edge_type = e.get("type", "delegates_to")
                if src and tgt:
                    self.engine.connect(topo_id, src, tgt, edge_type)
            
            # Register in SDTS tree with a default reward (will be updated via backpropagate)
            # In a full run, backpropagate is called AFTER the topology executes.
            self.sdts.backpropagate(task, json.dumps(topo_def), reward=0.0)
            
            return topo_id
            
        except Exception as e:
            self.logger.error(f"Failed to parse or create topology from LLM: {e}")
            # Fallback to a simple static vertical pattern
            topo_id = self.engine.create_topology(TopologyType.VERTICAL)
            parent_id = self.engine.add_node(topo_id, name="coordinator", role="coordinator")
            child_id = self.engine.add_node(topo_id, name="worker", role="worker")
            self.engine.connect(topo_id, parent_id, child_id, "delegates_to")
            return topo_id
