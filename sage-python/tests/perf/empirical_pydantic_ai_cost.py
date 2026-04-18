"""Empirical test: does Pydantic AI expose provider-reported cost?

Phase 2 of LiteLLM → Pydantic AI migration. Before writing a
PydanticAIProvider wrapper, verify whether we'd regress P0.3 (prefer
provider-reported `response_cost` over local estimate).

Two providers tested:
  1. DeepSeek — OpenAI-compatible via custom base_url (most-used path).
  2. Kimi — native `moonshotai:...` provider class.

For each call we inspect:
  * `result.usage()` — Pydantic AI's public usage API
  * `result.all_messages()` — full conversation including raw model
    response parts. We look for anything that smells like cost:
    provider_details, hidden_params, vendor_details, model_specific,
    extra, raw, etc.
  * `result._usage`, `result._response` — private attributes that may
    expose the underlying SDK response.

Run this script directly (not via pytest — the point is stdout
inspection) with your .env loaded.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path


def _load_env() -> None:
    env = Path(__file__).resolve().parents[3] / ".env"
    if not env.is_file():
        print("NO .env FOUND — aborting", file=sys.stderr)
        sys.exit(1)
    for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
    # Pydantic AI's native moonshotai provider reads MOONSHOTAI_API_KEY
    # (not MOONSHOT_API_KEY). Our .env ships KIMI_API_KEY; mirror it.
    for target in ("MOONSHOT_API_KEY", "MOONSHOTAI_API_KEY"):
        if os.environ.get("KIMI_API_KEY") and not os.environ.get(target):
            os.environ[target] = os.environ["KIMI_API_KEY"]


_load_env()

# Force UTF-8 stdout on Windows so our output is never blocked by cp1252.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


def _dump(obj, name: str, depth: int = 0, max_depth: int = 2) -> None:
    """Print an object's public attributes, shallowly.

    We want to see every field that might hold cost / pricing info.
    """
    indent = "  " * depth
    print(f"{indent}{name} = {type(obj).__module__}.{type(obj).__name__}")
    if depth >= max_depth:
        return
    for attr in sorted(dir(obj)):
        if attr.startswith("__"):
            continue
        try:
            val = getattr(obj, attr)
        except Exception as exc:
            print(f"{indent}  .{attr}: <raise {type(exc).__name__}>")
            continue
        if callable(val):
            continue
        # Highlight cost-adjacent fields
        is_costy = any(
            tag in attr.lower()
            for tag in ("cost", "price", "vendor", "detail", "provider", "hidden", "extra", "meta", "raw")
        )
        marker = " <-- SUSPECT" if is_costy else ""
        preview = repr(val)
        if len(preview) > 200:
            preview = preview[:200] + "...<truncated>"
        print(f"{indent}  .{attr}: {preview}{marker}")


async def test_deepseek() -> None:
    print("\n" + "=" * 70)
    print("DeepSeek — OpenAI-compatible via custom base_url")
    print("=" * 70)
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    model = OpenAIChatModel(
        "deepseek-chat",
        provider=OpenAIProvider(
            base_url="https://api.deepseek.com/v1",
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        ),
    )
    agent = Agent(model)
    try:
        result = await agent.run("Say 'hello' in 3 words max.")
    except Exception as exc:
        print(f"  ERROR: {type(exc).__name__}: {str(exc)[:300]}")
        return

    print(f"  output: {result.output!r}")
    print()
    usage = result.usage()
    _dump(usage, "result.usage()", max_depth=1)
    print()
    # Try to find the raw model response with vendor-specific details.
    msgs = result.all_messages()
    for i, m in enumerate(msgs):
        print(f"  all_messages()[{i}]:")
        _dump(m, f"    msg[{i}]", depth=1, max_depth=2)


async def test_kimi_native() -> None:
    print("\n" + "=" * 70)
    print("Kimi — native moonshotai provider")
    print("=" * 70)
    from pydantic_ai import Agent

    try:
        agent = Agent("moonshotai:kimi-k2-0711-preview")
    except Exception as exc:
        print(f"  ERROR constructing agent: {type(exc).__name__}: {exc}")
        return

    try:
        result = await agent.run("Say 'hello' in 3 words max.")
    except Exception as exc:
        print(f"  ERROR on run: {type(exc).__name__}: {str(exc)[:300]}")
        return

    print(f"  output: {result.output!r}")
    print()
    usage = result.usage()
    _dump(usage, "result.usage()", max_depth=1)
    print()
    # Raw model response inspection
    msgs = result.all_messages()
    print(f"  all_messages count: {len(msgs)}")
    # only last message (model response) is interesting
    if msgs:
        _dump(msgs[-1], "last message", depth=0, max_depth=2)


async def test_genai_prices_integration() -> None:
    """genai-prices is auto-installed with pydantic-ai. Test direct cost calc."""
    print("\n" + "=" * 70)
    print("genai-prices — direct cost calculation for a RunUsage")
    print("=" * 70)
    try:
        from genai_prices import calc_price
    except ImportError as exc:
        print(f"  ImportError: {exc}")
        return

    import inspect
    try:
        print(f"  calc_price signature: {inspect.signature(calc_price)}")
    except (TypeError, ValueError):
        pass

    # Try the minimal "hello" usage from deepseek we just saw
    from pydantic_ai.usage import RunUsage
    u = RunUsage(input_tokens=14, output_tokens=3, requests=1)
    # Try to price it for deepseek-chat
    try:
        priced = calc_price(u, "deepseek/deepseek-chat")
        print(f"  calc_price(deepseek-chat, 14in/3out) = {priced!r}")
        _dump(priced, "  priced", depth=1, max_depth=2)
    except Exception as exc:
        print(f"  calc_price ERROR: {type(exc).__name__}: {exc}")

    # Also openai/gpt-5.4 (our main concern)
    try:
        priced2 = calc_price(RunUsage(input_tokens=1000, output_tokens=500, requests=1), "openai/gpt-5.4")
        print(f"  calc_price(gpt-5.4, 1000in/500out) = {priced2!r}")
    except Exception as exc:
        print(f"  calc_price(gpt-5.4) ERROR: {type(exc).__name__}: {exc}")


async def main() -> None:
    await test_deepseek()
    await test_kimi_native()
    await test_genai_prices_integration()

    print("\n" + "=" * 70)
    print("KEY QUESTIONS TO ANSWER FROM OUTPUT ABOVE:")
    print("=" * 70)
    print("1. Does `result.usage()` expose a cost or price field?")
    print("2. Can we call genai_prices.calc_price(usage, model) directly?")
    print("3. For gpt-5.4 specifically — does genai-prices know this model?")


if __name__ == "__main__":
    asyncio.run(main())
