"""Vectorized environment wrapper for verl-agent integration.

Wraps SageTopologyEnv instances in the interface expected by verl-agent's
EnvironmentManagerBase: reset(), step(), build_text_obs(), success_evaluator().

Each environment runs independently with its own 4-state machine
(awaiting_yaml -> executing -> awaiting_decision -> terminal).

CRITICAL: Observations include 'anchor' keys for GiGPO step-level grouping.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from sage.verl.step_reward import StepRewardVector
from sage.verl.topology_env import SageTopologyEnv

log = logging.getLogger("sage_env_package")


class SageTopologyVerlEnv:
    """verl-agent compatible vectorized wrapper around SageTopologyEnv.

    Provides the interface expected by verl-agent's environment manager:
    - reset(prompts, **kwargs) -> list of observation dicts
    - step(actions) -> (observations, rewards, dones, infos)
    - build_text_obs(obs) -> list of text strings for the model
    - success_evaluator(trajectories) -> list of bools
    - get_step_rewards() -> list of StepRewardVectors

    Each observation dict contains:
    - "text": str -- the text observation for the model
    - "image": None -- placeholder for multimodal (not used)
    - "anchor": str -- GiGPO anchor key for step-level grouping
    """

    def __init__(self, config: dict | None = None):
        self._config: dict = config if config is not None else {}
        self._envs: list[SageTopologyEnv] = []
        self._n_envs: int = self._config.get("n_envs", 1)
        self._dones: list[bool] = []

    @property
    def n_envs(self) -> int:
        """Number of active environments."""
        return len(self._envs)

    @property
    def env_name(self) -> str:
        return "sage_topology"

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, prompts: list[str], **kwargs: Any) -> list[dict]:
        """Reset all environments with given prompts.

        Args:
            prompts: List of task prompts, one per environment.
            **kwargs:
                task_ids (list[str]): Optional task IDs for each environment.

        Returns:
            List of observation dicts with "text", "image", "anchor" keys.
        """
        task_ids = kwargs.get("task_ids", [""] * len(prompts))

        # Create one SageTopologyEnv per prompt
        self._envs = [
            SageTopologyEnv(config=self._config) for _ in range(len(prompts))
        ]
        self._dones = [False] * len(prompts)

        observations = []
        for env, prompt, tid in zip(self._envs, prompts, task_ids):
            obs = env.reset(prompt, tid)
            observations.append(obs)

        return observations

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self, actions: list[str]
    ) -> tuple[list[dict], list[float], list[bool], list[dict]]:
        """Step all environments with model actions.

        For environments that are already done, returns a no-op observation.

        Args:
            actions: List of action strings, one per environment.

        Returns:
            Tuple of (observations, rewards, dones, infos).
        """
        obs_list: list[dict] = []
        reward_list: list[float] = []
        done_list: list[bool] = []
        info_list: list[dict] = []

        for i, (env, action) in enumerate(zip(self._envs, actions)):
            if self._dones[i]:
                # Already terminated -- return no-op
                obs_list.append({
                    "text": "[TERMINATED]",
                    "image": None,
                    "anchor": "terminal:done",
                })
                reward_list.append(0.0)
                done_list.append(True)
                info_list.append({"status": "ALREADY_DONE"})
                continue

            obs, reward, done, info = env.step(action)
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
            self._dones[i] = done

        return obs_list, reward_list, done_list, info_list

    # ------------------------------------------------------------------
    # build_text_obs -- verl-agent interface
    # ------------------------------------------------------------------

    def build_text_obs(self, observations: list[dict]) -> list[str]:
        """Convert observation dicts to text strings for the model.

        verl-agent calls this to build the prompt/observation text
        that gets tokenized and fed to the policy model.

        Args:
            observations: List of observation dicts from reset()/step().

        Returns:
            List of text strings.
        """
        return [obs.get("text", "") for obs in observations]

    # ------------------------------------------------------------------
    # success_evaluator -- verl-agent interface
    # ------------------------------------------------------------------

    def success_evaluator(self, trajectories: list[dict]) -> list[bool]:
        """Evaluate whether each trajectory was successful.

        verl-agent calls this at the end of an episode for logging/metrics.

        Args:
            trajectories: List of trajectory dicts. Each should have
                'infos' (list of info dicts from step()).

        Returns:
            List of bools indicating success.
        """
        results = []
        for traj in trajectories:
            infos = traj.get("infos", [])
            if infos:
                last_info = infos[-1]
                status = last_info.get("status", "")
                results.append(status == "PASSED")
            else:
                results.append(False)
        return results

    # ------------------------------------------------------------------
    # get_step_rewards -- GiGPO interface
    # ------------------------------------------------------------------

    def get_step_rewards(self) -> list[StepRewardVector]:
        """Get StepRewardVectors for all environments.

        Returns one StepRewardVector per environment, containing per-step
        rewards and anchor keys for GiGPO advantage computation.
        """
        return [env.get_step_rewards() for env in self._envs]

    # ------------------------------------------------------------------
    # get_traces -- debugging/logging
    # ------------------------------------------------------------------

    def get_traces(self) -> list:
        """Get EpisodeTraces for all environments (for logging/debugging)."""
        return [env.get_trace() for env in self._envs]

    def close(self) -> None:
        """Clean up resources."""
        self._envs.clear()
        self._dones.clear()
