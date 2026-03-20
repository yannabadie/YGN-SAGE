"""StepRewardVector — Per-node reward decomposition for GiGPO.

Transforms the flat reward (one float per topology) into a vector of
rewards per step, compatible with verl-agent's GiGPO advantage estimator.

Each step has:
- reward: float (the step's contribution)
- anchor_key: str (for GiGPO step-level grouping)

The total episode reward = sum of all step rewards.
GiGPO normalizes within anchor groups to provide credit assignment.

Reference: GiGPO (arXiv 2505.10978), Section 3.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StepRewardVector:
    """Reward decomposed by step for GiGPO."""

    step_rewards: list[float] = field(default_factory=list)
    anchor_keys: list[str] = field(default_factory=list)
    episode_reward: float = 0.0
    status: str = ""

    @property
    def n_steps(self) -> int:
        return len(self.step_rewards)

    @property
    def flat_reward(self) -> float:
        """Backward-compat: single float reward."""
        return self.episode_reward

    def to_verl_format(self) -> dict:
        """Format for verl-agent GiGPO.

        verl-agent expects per-step rewards and anchor keys in the batch data.
        The rollout loop collects these per trajectory.
        """
        return {
            "rewards": self.step_rewards,
            "anchor_keys": self.anchor_keys,
            "total_return": self.episode_reward,
            "n_steps": self.n_steps,
        }

    @classmethod
    def from_episode_trace(cls, trace) -> StepRewardVector:
        """Build from a SageTopologyEnv EpisodeTrace."""
        vec = cls()
        for step in trace.steps:
            vec.step_rewards.append(step.reward)
            vec.anchor_keys.append(step.anchor_key)
        vec.episode_reward = trace.total_reward
        vec.status = trace.status
        return vec
