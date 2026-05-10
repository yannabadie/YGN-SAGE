#!/usr/bin/env python3
"""Provider health matrix preflight — P2 (cgpro 2026-05-08).

Per-model diagnostic: status, latency, tokens, cost.  Reports which
providers/models are operational before a paid benchmark run.

Usage:
  python sage-python/scripts/provider_preflight.py \
      --output docs/benchmarks/provider_preflight.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ── Model matrix to test ────────────────────────────────────────────────────

# (provider_key, model_id, tier_label)
MATRIX: list[tuple[str, str, str]] = [
    # Google
    ("google", "gemini-2.5-flash", "fast"),
    ("google", "gemini-3.1-flash-lite-preview", "fast"),
    # DeepSeek
    ("deepseek", "deepseek-v4-flash", "budget"),
    ("deepseek", "deepseek-v4-pro", "reasoner"),
    ("xai", "grok-code-fast-1", "coding"),
    ("kimi", "kimi-k2.6", "reasoner"),
    ("minimax", "MiniMax-M2.7", "reasoner"),
    ("openrouter", "qwen/qwen3.5-plus-02-15", "reasoner"),
    # OpenAI — diagnostic mode (marked excluded in canary until green)
    ("openai", "gpt-5.4", "reasoner"),
    ("openai", "gpt-5.5-pro", "reasoner"),
]


@dataclass
class ProviderResult:
    provider: str
    model_id: str
    status: str = "unknown"  # ok | error | timeout | skipped
    evidence_scope: str = "liveness_only"
    warnings: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    output_length: int = 0
    cost_usd: float | None = None
    error_type: str | None = None
    error_message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _usage_get(usage: Any, key: str, default: Any = 0) -> Any:
    if isinstance(usage, dict):
        return usage.get(key, default)
    return getattr(usage, key, default)


def _cost_from_usage(
    usage: Any,
    provider_key: str,
    model_id: str,
    *,
    lookup_cost_per_token: Any,
) -> float:
    provider_reported = _usage_get(usage, "cost_usd", None)
    if provider_reported is not None:
        return float(provider_reported)

    inp = int(_usage_get(usage, "input_tokens", 0) or 0)
    out = int(_usage_get(usage, "output_tokens", 0) or 0)
    cost_in, cost_out = lookup_cost_per_token(provider_key, model_id)
    return inp * cost_in + out * cost_out


async def _test_one(
    provider_key: str,
    model_id: str,
    api_key: str,
    timeout: float = 60.0,
) -> ProviderResult:
    result = ProviderResult(provider=provider_key, model_id=model_id)
    result.warnings = _preflight_warnings(provider_key, model_id)
    t0 = time.perf_counter()

    try:
        from sage.providers.pydantic_ai_provider import PydanticAIProvider
        from sage.llm.base import Message, Role, LLMConfig

        provider = PydanticAIProvider.for_sage_provider(
            provider_key, model_id, api_key,
        )

        config = LLMConfig(provider=provider_key, model=model_id, max_tokens=50)

        response = await asyncio.wait_for(
            provider.generate(
                messages=[Message(role=Role.USER, content="Reply with exactly: ok")],
                config=config,
            ),
            timeout=timeout,
        )

        if not (response.content or "").strip():
            result.status = "error"
            result.error_type = "EmptyContent"
            result.error_message = "Provider returned an empty final content field"
            result.latency_ms = (time.perf_counter() - t0) * 1000
            return result

        if (response.content or "").strip().lower() != "ok":
            result.warnings.append(
                "Smoke output was non-empty but not exactly 'ok'; "
                "preflight remains liveness-only, not instruction-following evidence."
            )

        result.status = "ok"
        result.output_length = len(response.content or "")
        result.latency_ms = (time.perf_counter() - t0) * 1000

        # Cost from usage
        if hasattr(response, "usage") and response.usage:
            from sage.providers.pydantic_ai_provider import (
                _lookup_cost_per_token,
            )

            result.cost_usd = _cost_from_usage(
                response.usage,
                provider_key,
                model_id,
                lookup_cost_per_token=_lookup_cost_per_token,
            )
        else:
            result.cost_usd = 0.0

    except asyncio.TimeoutError:
        result.status = "timeout"
        result.error_type = "TimeoutError"
        result.latency_ms = (time.perf_counter() - t0) * 1000
    except Exception as exc:
        result.status = "error"
        result.error_type = type(exc).__name__
        result.error_message = str(exc)[:500]
        result.latency_ms = (time.perf_counter() - t0) * 1000

    return result


def _preflight_warnings(provider_key: str, model_id: str) -> list[str]:
    """Provider/model caveats that are known before the live call."""
    warnings: list[str] = []
    if provider_key == "xai" and model_id == "grok-code-fast-1":
        warnings.append(
            "xAI lists grok-code-fast-1 for retirement on 2026-05-15T19:00:00Z; "
            "treat this as short-lived liveness evidence."
        )
    return warnings


async def run_all(
    matrix: list[tuple[str, str, str]],
    api_keys: dict[str, str],
    timeout: float = 60.0,
) -> list[ProviderResult]:
    results: list[ProviderResult] = []
    for provider_key, model_id, tier in matrix:
        api_key = api_keys.get(provider_key, "")
        if not api_key:
            r = ProviderResult(
                provider=provider_key, model_id=model_id,
                status="skipped", error_type="MissingApiKey",
                warnings=_preflight_warnings(provider_key, model_id),
            )
        else:
            r = await _test_one(provider_key, model_id, api_key, timeout)
        results.append(r)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Provider health matrix preflight")
    parser.add_argument(
        "--output", default=None,
        help="Write JSON report to this path",
    )
    parser.add_argument(
        "--timeout", type=float, default=60.0,
        help="Per-provider timeout in seconds",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Print JSON to stdout",
    )
    args = parser.parse_args(argv)

    # Resolve API keys from environment
    api_keys: dict[str, str] = {}
    key_map = {
        "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
        "deepseek": ["DEEPSEEK_API_KEY", "DEEP_SEEK_API_KEY"],
        "xai": ["GROK_API_KEY"],
        "kimi": ["KIMI_API_KEY"],
        "minimax": ["MINIMAX_API_KEY"],
        "openrouter": ["OPEN_ROUTER_API_KEY"],
    }
    for provider, env_vars in key_map.items():
        for var in env_vars:
            val = os.environ.get(var, "")
            if val:
                api_keys[provider] = val
                break

    results = asyncio.run(run_all(MATRIX, api_keys, timeout=args.timeout))

    report = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": [r.as_dict() for r in results],
        "summary": {
            "ok": sum(1 for r in results if r.status == "ok"),
            "error": sum(1 for r in results if r.status == "error"),
            "timeout": sum(1 for r in results if r.status == "timeout"),
            "skipped": sum(1 for r in results if r.status == "skipped"),
        },
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )

    if args.json or not args.output:
        print(json.dumps(report, indent=2))

    # Exit code: non-zero if any error
    return 1 if report["summary"]["error"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
