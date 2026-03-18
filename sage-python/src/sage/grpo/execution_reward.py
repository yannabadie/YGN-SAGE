"""Execution reward for GRPO v3 — multi-provider reasoning pool + TopologyReward Rust.

Executes each generated topology on the original task using MULTIPLE reasoning providers
in parallel (round-robin), then scores with TopologyReward (structural + density + execution).

Provider pool (all reasoning/thinking models):
  1. DeepSeek Reasoner  — unlimited RPM, ~40s latency, 64 concurrent
  2. Gemini 3.1 Pro     — thinking=HIGH, ~10s latency, 10 concurrent
  3. Qwen QwQ-Plus      — native reasoning, ~20s latency, 10 concurrent
  4. Kimi K2.5          — thinking mode, ~10s latency, 5 concurrent
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import yaml

log = logging.getLogger("grpo_v2")

# Per-provider concurrency limits
DEEPSEEK_CONCURRENCY = int(os.environ.get("SAGE_DEEPSEEK_CONCURRENCY", "64"))
GEMINI_CONCURRENCY = int(os.environ.get("SAGE_GEMINI_CONCURRENCY", "10"))
MINIMAX_CONCURRENCY = int(os.environ.get("SAGE_MINIMAX_CONCURRENCY", "20"))
KIMI_CONCURRENCY = int(os.environ.get("SAGE_KIMI_CONCURRENCY", "50"))  # Tier 3: 200 max, 5000 RPM

@dataclass
class ProviderSlot:
    """A reasoning provider with its own semaphore and config."""
    name: str
    provider: Any  # OpenAICompatProvider or GoogleProvider
    model: str
    semaphore: asyncio.Semaphore | None = None  # created per event loop
    concurrency: int = 10
    timeout: float = 120.0
    provider_type: str = "openai"  # "openai" or "google"
    calls: int = 0
    errors: int = 0

# Provider pool — populated lazily
_PROVIDER_POOL: list[ProviderSlot] = []
_DEEPSEEK_FALLBACK: ProviderSlot | None = None  # slow but reliable
_POOL_INITIALIZED = False
_POOL_COUNTER = 0  # round-robin counter

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


def _init_provider_pool():
    """Initialize all available reasoning providers."""
    global _PROVIDER_POOL, _POOL_INITIALIZED
    if _POOL_INITIALIZED:
        return
    _POOL_INITIALIZED = True

    from sage.providers.openai_compat import OpenAICompatProvider

    # 1. DeepSeek Reasoner — FALLBACK ONLY (40s/req too slow for primary pool)
    #    Kept as fallback in evaluate_topology() if fast providers fail
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if key:
        try:
            global _DEEPSEEK_FALLBACK
            _DEEPSEEK_FALLBACK = ProviderSlot(
                name="deepseek-reasoner",
                provider=OpenAICompatProvider(api_key=key, base_url="https://api.deepseek.com/v1", provider_name="deepseek"),
                model="deepseek-reasoner",
                concurrency=DEEPSEEK_CONCURRENCY,
                timeout=120.0,
                provider_type="openai",
            )
            log.info("Fallback: DeepSeek Reasoner (40s/req, concurrency=%d)", DEEPSEEK_CONCURRENCY)
        except Exception as e:
            log.warning("DeepSeek init failed: %s", str(e)[:80])

    # 2. Gemini 3.1 Pro Preview — via OpenAI-compat (bypasses aiohttp SSL issues)
    key = os.environ.get("GOOGLE_API_KEY", "")
    if key:
        try:
            _PROVIDER_POOL.append(ProviderSlot(
                name="gemini-pro",
                provider=OpenAICompatProvider(
                    api_key=key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                    provider_name="google",
                ),
                model="gemini-3.1-pro-preview",
                concurrency=GEMINI_CONCURRENCY,
                timeout=60.0,
                provider_type="openai",
            ))
            log.info("Pool: Gemini 3.1 Pro (concurrency=%d)", GEMINI_CONCURRENCY)
        except Exception as e:
            log.warning("Gemini init failed: %s", str(e)[:80])

    # 3. MiniMax M2.7 — native thinking (<think> tags)
    key = os.environ.get("MINIMAX_API_KEY", "")
    if key:
        try:
            _PROVIDER_POOL.append(ProviderSlot(
                name="minimax-m2.7",
                provider=OpenAICompatProvider(
                    api_key=key,
                    base_url="https://api.minimaxi.chat/v1",
                    provider_name="minimax",
                ),
                model="MiniMax-M2.7",
                concurrency=MINIMAX_CONCURRENCY,
                timeout=60.0,
                provider_type="openai",
            ))
            log.info("Pool: MiniMax M2.7 thinking (concurrency=%d)", MINIMAX_CONCURRENCY)
        except Exception as e:
            log.warning("MiniMax init failed: %s", str(e)[:80])

    # 4. Kimi K2.5 — thinking mode
    key = os.environ.get("KIMI_API_KEY", "")
    if key:
        try:
            _PROVIDER_POOL.append(ProviderSlot(
                name="kimi-k2.5",
                provider=OpenAICompatProvider(
                    api_key=key,
                    base_url="https://api.moonshot.ai/v1",
                    provider_name="kimi",
                ),
                model="kimi-k2.5",
                concurrency=KIMI_CONCURRENCY,
                timeout=60.0,
                provider_type="openai",
            ))
            log.info("Pool: Kimi K2.5 thinking (concurrency=%d, Tier 3: 200 max)", KIMI_CONCURRENCY)
        except Exception as e:
            log.warning("Kimi init failed: %s", str(e)[:80])

    total_concurrent = sum(s.concurrency for s in _PROVIDER_POOL)
    log.info("Provider pool: %d providers, %d total concurrent slots", len(_PROVIDER_POOL), total_concurrent)


def _pick_provider() -> ProviderSlot | None:
    """Round-robin provider selection."""
    global _POOL_COUNTER
    _init_provider_pool()
    if not _PROVIDER_POOL:
        return None
    slot = _PROVIDER_POOL[_POOL_COUNTER % len(_PROVIDER_POOL)]
    _POOL_COUNTER += 1
    return slot


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


async def _call_provider(slot: ProviderSlot, messages: list, config: Any) -> str | None:
    """Call a single provider with its semaphore. Returns response text or None."""
    if slot.semaphore is None:
        slot.semaphore = asyncio.Semaphore(slot.concurrency)

    async with slot.semaphore:
        try:
            if slot.provider_type == "google":
                # Gemini uses thinking config via generation_config
                response = await asyncio.wait_for(
                    slot.provider.generate(messages=messages, config=config),
                    timeout=slot.timeout,
                )
            else:
                response = await asyncio.wait_for(
                    slot.provider.generate(messages=messages, config=config),
                    timeout=slot.timeout,
                )
            slot.calls += 1
            return response.content or ""
        except Exception:
            slot.errors += 1
            return None


async def evaluate_topology(
    task: str, topology_dict: dict, slot: ProviderSlot,
) -> float:
    """Execute a topology on a task via a specific provider slot."""
    t0 = time.time()

    # Build graph
    graph = _build_topology_graph(topology_dict)

    # Build messages from first node
    from sage.llm.base import Message, Role, LLMConfig

    nodes = topology_dict.get("nodes", [])
    system_prompt = nodes[0].get("prompt", "You are a helpful assistant.") if nodes else ""

    messages = [
        Message(role=Role.SYSTEM, content=system_prompt),
        Message(role=Role.USER, content=task[:1000]),
    ]
    config = LLMConfig(provider=slot.name, model=slot.model)

    result = await _call_provider(slot, messages, config)

    if result is not None:
        execution_passed = len(result.strip()) > 20
        tokens = len(result) // 4
        _stats.record(success=execution_passed, tokens=tokens, latency=time.time() - t0)
        return _compute_rust_reward(graph, execution_passed)

    # Primary failed — try DeepSeek fallback (slow but reliable)
    if _DEEPSEEK_FALLBACK is not None:
        fb_config = LLMConfig(provider=_DEEPSEEK_FALLBACK.name, model=_DEEPSEEK_FALLBACK.model)
        result = await _call_provider(_DEEPSEEK_FALLBACK, messages, fb_config)
        if result is not None:
            execution_passed = len(result.strip()) > 20
            _stats.record(success=execution_passed, tokens=len(result) // 4, latency=time.time() - t0)
            return _compute_rust_reward(graph, execution_passed)

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

    # Parse stats
    n_valid = len(tasks_and_topos)
    n_invalid = len(completions) - n_valid
    if (n_valid + n_invalid) > 0 and (n_valid + n_invalid) % 8 == 0:
        log.info(
            "Parse: %d valid (%.0f%%) | %d invalid — temp=0.6 target: >50%%",
            n_valid, 100 * n_valid / max(n_valid + n_invalid, 1), n_invalid,
        )

    # All unparseable → 0.0
    rewards = [0.0] * len(completions)

    if not tasks_and_topos:
        return rewards

    # Initialize pool and assign providers round-robin
    _init_provider_pool()
    if not _PROVIDER_POOL:
        log.error("No providers available — all execution rewards = 0.0")
        return rewards

    async def _run_all():
        global _POOL_COUNTER
        coros = []
        for task, topo in tasks_and_topos:
            slot = _PROVIDER_POOL[_POOL_COUNTER % len(_PROVIDER_POOL)]
            _POOL_COUNTER += 1
            coros.append(evaluate_topology(task, topo, slot))
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

    # Log if all failed + per-provider stats
    valid_count = sum(1 for r in rewards if r > 0)
    if valid_count == 0 and len(tasks_and_topos) > 0:
        log.warning("All execution rewards failed — degraded mode (format+structure only)")

    # Per-provider stats every batch
    pool_stats = " | ".join(
        f"{s.name}: {s.calls}ok/{s.errors}err" for s in _PROVIDER_POOL
    )
    if pool_stats:
        log.info("Pool: %s", pool_stats)

    return rewards
