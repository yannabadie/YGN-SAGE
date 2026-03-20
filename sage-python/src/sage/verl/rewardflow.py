"""RewardFlow — per-node credit via state-graph PageRank propagation.

Inspired by RewardFlow (arXiv 2603.18859, AAMAS 2026).
Builds a state graph from K rollouts, propagates terminal rewards
backward via Personalized PageRank to assign per-node credit.

Usage:
    prop = RewardFlowPropagator(damping=0.85, max_iters=20)
    per_node_rewards = prop.compute(rollouts)
    # per_node_rewards[i] = {node_idx: reward} for rollout i
"""
from __future__ import annotations

import logging
from collections import defaultdict

log = logging.getLogger("rewardflow")


def _quality_bucket(quality: float) -> str:
    """Bin quality score into low/med/high."""
    if quality < 0.3:
        return "low"
    if quality < 0.7:
        return "med"
    return "high"


class RewardFlowPropagator:
    """Per-node credit assignment via state-graph PageRank."""

    def __init__(self, damping: float = 0.85, max_iters: int = 20):
        self._damping = damping
        self._max_iters = max_iters

    def compute(self, rollouts: list[dict]) -> list[dict[int, float]]:
        """Build state-graph from K rollouts, propagate terminal rewards.

        Args:
            rollouts: list of {"node_traces": [...], "terminal_reward": float}
                Each node_trace: {"node_idx": int, "role": str, "quality": float}

        Returns:
            list of {node_idx: reward} dicts, one per rollout.
        """
        if not rollouts:
            return []

        # 1. Build state graph: state = (role, quality_bucket)
        # Edges: transition counts between consecutive states
        transitions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        terminal_rewards: dict[str, list[float]] = defaultdict(list)

        for rollout in rollouts:
            nodes = rollout.get("node_traces", [])
            term_reward = rollout.get("terminal_reward", 0.0)

            prev_state = None
            for node in nodes:
                role = node.get("role", "agent")
                quality = node.get("quality", 0.5)
                state = f"{role}:{_quality_bucket(quality)}"

                if prev_state is not None:
                    transitions[prev_state][state] += 1
                prev_state = state

            # Terminal state gets the execution reward
            if prev_state is not None:
                terminal_rewards[prev_state].append(term_reward)

        # 2. Personalized PageRank backward propagation
        all_states = set(transitions.keys())
        for targets in transitions.values():
            all_states.update(targets.keys())
        all_states.update(terminal_rewards.keys())

        if not all_states:
            return [{} for _ in rollouts]

        # Initialize: terminal states get their mean reward, others get 0
        state_reward: dict[str, float] = {}
        for s in all_states:
            if s in terminal_rewards:
                state_reward[s] = sum(terminal_rewards[s]) / len(terminal_rewards[s])
            else:
                state_reward[s] = 0.0

        # Build reverse transition graph (for backward propagation)
        reverse_trans: dict[str, dict[str, float]] = defaultdict(dict)
        for src, targets in transitions.items():
            total = sum(targets.values())
            for tgt, count in targets.items():
                reverse_trans[tgt][src] = count / total

        # PageRank iterations
        for _ in range(self._max_iters):
            new_rewards = {}
            for state in all_states:
                # Seed from terminal rewards
                seed = 0.0
                if state in terminal_rewards:
                    seed = sum(terminal_rewards[state]) / len(terminal_rewards[state])

                # Propagation from successors
                prop = 0.0
                if state in transitions:
                    total = sum(transitions[state].values())
                    for tgt, count in transitions[state].items():
                        prop += (count / total) * state_reward.get(tgt, 0.0)

                new_rewards[state] = (1 - self._damping) * seed + self._damping * prop

            state_reward = new_rewards

        # 3. Map back to per-rollout per-node rewards
        results = []
        for rollout in rollouts:
            node_rewards = {}
            for node in rollout.get("node_traces", []):
                role = node.get("role", "agent")
                quality = node.get("quality", 0.5)
                state = f"{role}:{_quality_bucket(quality)}"
                node_rewards[node["node_idx"]] = state_reward.get(state, 0.0)
            results.append(node_rewards)

        return results
