"""Execution reward for GRPO v2 — runs topologies via DeepSeek Reasoner + TopologyReward Rust.

Executes each generated topology on the original task using real LLM providers,
then scores with TopologyReward (structural + density + execution).

Provider: DeepSeek V3.2 Reasoner (thinking mode, $2.19/M output, best for algorithmic/reasoning)
Fallback: Gemini 3.1 Flash Lite Preview
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

# Concurrency — DeepSeek has "no official constraint" but be conservative
DEEPSEEK_CONCURRENCY = int(os.environ.get("SAGE_DEEPSEEK_CONCURRENCY", "8"))

# Lazy-loaded providers
_DEEPSEEK_PROVIDER = None
_FALLBACK_PROVIDER = None

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


class ExecutionStats:
    """Track execution reward statistics for logging."""

    def __init__(self):
        self.total = 0
        self.success = 0
        self.failed = 0
        self.timeout = 0
        self.api_errors = 0
        self.total_tokens = 0
        self.total_latency = 0.0

    def record(self, *, success: bool, timeout: bool = False,
               tokens: int = 0, latency: float = 0.0, api_error: bool = False):
        self.total += 1
        if success:
            self.success += 1
        elif timeout:
            self.timeout += 1
        elif api_error:
            self.api_errors += 1
        else:
            self.failed += 1
        self.total_tokens += tokens
        self.total_latency += latency
        if self.total % 10 == 0:
            log.info(
                "Exec: %d total | %d ok (%.0f%%) | %d fail | %d timeout | %d api_err | "
                "~%d tok | avg %.1fs | ~$%.4f",
                self.total, self.success,
                100 * self.success / max(self.total, 1),
                self.failed, self.timeout, self.api_errors,
                self.total_tokens,
                self.total_latency / max(self.total, 1),
                self.total_tokens * 2.19 / 1_000_000,  # reasoner output rate
            )


_stats = ExecutionStats()


def _get_deepseek_provider():
    """Lazy-load DeepSeek Reasoner provider."""
    global _DEEPSEEK_PROVIDER
    if _DEEPSEEK_PROVIDER is not None:
        return _DEEPSEEK_PROVIDER

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        log.warning("DEEPSEEK_API_KEY not set — execution reward disabled")
        return None

    try:
        from sage.providers.openai_compat import OpenAICompatProvider
        _DEEPSEEK_PROVIDER = OpenAICompatProvider(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            provider_name="deepseek",
        )
        log.info("DeepSeek Reasoner provider initialized")
        return _DEEPSEEK_PROVIDER
    except Exception as exc:
        log.warning("DeepSeek init failed: %s", str(exc)[:100])
        return None


def _get_fallback_provider():
    """Lazy-load Gemini 3.1 Flash Lite fallback."""
    global _FALLBACK_PROVIDER
    if _FALLBACK_PROVIDER is not None:
        return _FALLBACK_PROVIDER

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        return None

    try:
        from sage.llm.google import GoogleProvider
        _FALLBACK_PROVIDER = GoogleProvider(api_key=api_key)
        log.info("Gemini fallback provider initialized")
        return _FALLBACK_PROVIDER
    except Exception as exc:
        log.warning("Gemini fallback init failed: %s", str(exc)[:100])
        return None


def _build_topology_graph(topology_dict: dict) -> Any:
    """Build a TopologyGraph from a dict (same as pipeline.py Stage 2)."""
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
    """Compute TopologyReward from Rust infrastructure."""
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
    """Execute a topology on a task and return reward score (0.0-1.0)."""
    async with semaphore:
        t0 = time.time()

        # Build graph
        graph = _build_topology_graph(topology_dict)

        # Get provider
        provider = _get_deepseek_provider()
        model = "deepseek-reasoner"

        if provider is None:
            provider = _get_fallback_provider()
            model = "gemini-3.1-flash-lite-preview"

        if provider is None:
            _stats.record(success=False, api_error=True, latency=time.time() - t0)
            return 0.0

        # Execute: single-node simplified execution (not full TopologyRunner)
        # For GRPO speed, we just test if the first node's prompt + task produces useful output
        from sage.llm.base import Message, Role, LLMConfig

        nodes = topology_dict.get("nodes", [])
        system_prompt = nodes[0].get("prompt", "You are a helpful assistant.") if nodes else ""

        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=task[:1000]),
        ]
        config = LLMConfig(provider="deepseek", model=model)

        try:
            response = await asyncio.wait_for(
                provider.generate(messages=messages, config=config),
                timeout=60.0,
            )
            result = response.content or ""
            execution_passed = len(result.strip()) > 20

            tokens = len(result) // 4  # rough estimate
            _stats.record(
                success=execution_passed, tokens=tokens, latency=time.time() - t0,
            )

            return _compute_rust_reward(graph, execution_passed)

        except asyncio.TimeoutError:
            _stats.record(success=False, timeout=True, latency=time.time() - t0)
            # Retry with fallback
            fallback = _get_fallback_provider()
            if fallback and fallback is not provider:
                try:
                    fb_config = LLMConfig(provider="google", model="gemini-3.1-flash-lite-preview")
                    response = await asyncio.wait_for(
                        fallback.generate(messages=messages, config=fb_config),
                        timeout=60.0,
                    )
                    result = response.content or ""
                    execution_passed = len(result.strip()) > 20
                    return _compute_rust_reward(graph, execution_passed)
                except Exception:
                    pass
            return 0.0

        except Exception:
            _stats.record(success=False, api_error=True, latency=time.time() - t0)
            return 0.0


def execution_reward_batch(completions: list, **kwargs) -> list[float]:
    """Sync wrapper for batch execution reward. Called by GRPOTrainer."""
    prompts = kwargs.get("prompts", [])

    # Parse completions into topology dicts
    tasks_and_topos = []
    indices = []
    for i, completion in enumerate(completions):
        # Handle conversational format
        text = completion[0]["content"] if isinstance(completion, list) else completion
        prompt = prompts[i] if i < len(prompts) else ""
        if isinstance(prompt, list):
            prompt = prompt[-1]["content"] if prompt else ""

        # Parse YAML/JSON
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

    # All unparseable → 0.0
    rewards = [0.0] * len(completions)

    if not tasks_and_topos:
        return rewards

    # Run async evaluations
    semaphore = asyncio.Semaphore(DEEPSEEK_CONCURRENCY)

    async def _run_all():
        coros = [
            evaluate_topology(task, topo, semaphore)
            for task, topo in tasks_and_topos
        ]
        return await asyncio.gather(*coros, return_exceptions=True)

    try:
        results = asyncio.run(_run_all())
    except RuntimeError:
        # Already in an event loop — use nest_asyncio or thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            results = pool.submit(lambda: asyncio.run(_run_all())).result(timeout=300)

    for idx, result in zip(indices, results):
        if isinstance(result, (int, float)):
            rewards[idx] = float(result)

    # Log if all failed
    valid_count = sum(1 for r in rewards if r > 0)
    if valid_count == 0 and len(tasks_and_topos) > 0:
        log.warning("All execution rewards failed — degraded mode (format+structure only)")

    return rewards
