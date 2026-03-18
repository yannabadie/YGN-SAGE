"""Execution reward for GRPO v3 — DeepSeek Reasoner + TopologyReward Rust.

Single provider, no pool, no fallback. Designed for overnight uninterrupted runs.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import yaml

log = logging.getLogger("grpo_v2")

CONCURRENCY = int(os.environ.get("SAGE_DEEPSEEK_CONCURRENCY", "8"))

# Rust reward infrastructure (optional)
try:
    from sage_core import TopologyReward, TopologyDensity, TopologyGraph, TopologyNode
    from sage_core import PyHybridVerifier

    _reward_scorer = TopologyReward()
    _density_scorer = TopologyDensity()
    _verifier = PyHybridVerifier()
    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False

# Lazy-loaded provider
_PROVIDER = None


class ExecutionStats:
    def __init__(self):
        self.total = 0
        self.success = 0
        self.failed = 0
        self.timeout = 0
        self.total_tokens = 0
        self.total_latency = 0.0

    def record(self, *, success: bool, timeout: bool = False,
               tokens: int = 0, latency: float = 0.0):
        self.total += 1
        if success:
            self.success += 1
        elif timeout:
            self.timeout += 1
        else:
            self.failed += 1
        self.total_tokens += tokens
        self.total_latency += latency
        if self.total % 10 == 0:
            log.info(
                "Exec: %d total | %d ok (%.0f%%) | %d fail | %d timeout | "
                "~%d tok | avg %.1fs | ~$%.4f",
                self.total, self.success,
                100 * self.success / max(self.total, 1),
                self.failed, self.timeout,
                self.total_tokens,
                self.total_latency / max(self.total, 1),
                self.total_tokens * 2.19 / 1_000_000,
            )


_stats = ExecutionStats()


def _get_provider():
    global _PROVIDER
    if _PROVIDER is not None:
        return _PROVIDER

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        log.warning("DEEPSEEK_API_KEY not set")
        return None

    try:
        from sage.providers.openai_compat import OpenAICompatProvider
        _PROVIDER = OpenAICompatProvider(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            provider_name="deepseek",
        )
        log.info("DeepSeek Reasoner initialized (concurrency=%d)", CONCURRENCY)
        return _PROVIDER
    except Exception as exc:
        log.warning("DeepSeek init failed: %s", str(exc)[:100])
        return None


def _build_topology_graph(topology_dict: dict) -> Any:
    if not _RUST_AVAILABLE:
        return None
    nodes = topology_dict.get("nodes", [])
    if not nodes:
        return None
    graph = TopologyGraph("sequential")
    for node_data in nodes:
        if not isinstance(node_data, dict):
            continue
        node = TopologyNode(
            role=node_data.get("role", "agent"),
            model_id="",
            system=2,
            prompt=node_data.get("prompt", ""),
        )
        graph.add_node(node)
    return graph


def _compute_rust_reward(graph: Any, execution_passed: bool) -> float:
    if not _RUST_AVAILABLE or graph is None:
        return 1.0 if execution_passed else 0.0
    try:
        density = _density_scorer.compute(graph, 2)
        verification = _verifier.verify(graph)
        structural_score = 1.0 if verification.valid else 0.5
        reward = _reward_scorer.compute(
            execution_passed=execution_passed,
            structural_score=structural_score,
            density_score=density.s_complex,
            temporal_score=None,
        )
        score = reward.total
        if density.over_budget:
            score *= 0.5
        return float(score)
    except Exception:
        return 1.0 if execution_passed else 0.0


async def evaluate_topology(
    task: str, topology_dict: dict, semaphore: asyncio.Semaphore,
) -> float:
    async with semaphore:
        t0 = time.time()
        graph = _build_topology_graph(topology_dict)
        provider = _get_provider()
        if provider is None:
            return 0.0

        from sage.llm.base import Message, Role, LLMConfig
        nodes = topology_dict.get("nodes", [])
        system_prompt = nodes[0].get("prompt", "You are a helpful assistant.") if nodes else ""
        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=task[:1000]),
        ]
        config = LLMConfig(provider="deepseek", model="deepseek-reasoner")

        try:
            response = await asyncio.wait_for(
                provider.generate(messages=messages, config=config),
                timeout=120.0,
            )
            result = response.content or ""
            execution_passed = len(result.strip()) > 20
            tokens = len(result) // 4
            _stats.record(success=execution_passed, tokens=tokens, latency=time.time() - t0)
            return _compute_rust_reward(graph, execution_passed)
        except asyncio.TimeoutError:
            _stats.record(success=False, timeout=True, latency=time.time() - t0)
            return 0.0
        except Exception:
            _stats.record(success=False, latency=time.time() - t0)
            return 0.0


def execution_reward_batch(completions: list, **kwargs) -> list[float]:
    """Sync wrapper for batch execution reward. Called by GRPOTrainer."""
    prompts = kwargs.get("prompts", [])

    tasks_and_topos = []
    indices = []
    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        prompt = prompts[i] if i < len(prompts) else ""
        if isinstance(prompt, list):
            prompt = prompt[-1]["content"] if prompt else ""

        topo = None
        try:
            topo = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            try:
                topo = yaml.safe_load(text)
            except Exception:
                pass

        if isinstance(topo, dict) and "nodes" in topo and len(topo.get("nodes", [])) > 0:
            tasks_and_topos.append((prompt, topo))
            indices.append(i)

    n_valid = len(tasks_and_topos)
    n_invalid = len(completions) - n_valid
    if (n_valid + n_invalid) > 0 and (n_valid + n_invalid) % 8 == 0:
        log.info(
            "Parse: %d valid (%.0f%%) | %d invalid",
            n_valid, 100 * n_valid / max(n_valid + n_invalid, 1), n_invalid,
        )

    rewards = [0.0] * len(completions)
    if not tasks_and_topos:
        return rewards

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def _run_all():
        coros = [
            evaluate_topology(task, topo, semaphore)
            for task, topo in tasks_and_topos
        ]
        return await asyncio.gather(*coros, return_exceptions=True)

    try:
        results = asyncio.run(_run_all())
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            results = pool.submit(lambda: asyncio.run(_run_all())).result(timeout=300)

    for idx, result in zip(indices, results):
        if isinstance(result, (int, float)):
            rewards[idx] = float(result)

    return rewards
