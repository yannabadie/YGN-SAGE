"""Phase 2 validation: PydanticAIProvider on DeepSeek + Kimi.

Does our wrapper correctly:
1. Build a Pydantic AI model via the factory?
2. Translate a multi-message history (system + user) into Pydantic AI
   ModelRequest list + instructions string?
3. Get a response back, translate to our LLMResponse shape?
4. Populate usage (input/output/total_tokens) + cost_usd from cards.toml?
5. Support tool calls (when a ToolDef is passed)?

Run directly (not pytest) — goal is stdout inspection.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path


def _load_env() -> None:
    env = Path(__file__).resolve().parents[3] / ".env"
    for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]
    for target in ("MOONSHOTAI_API_KEY", "MOONSHOT_API_KEY"):
        if os.environ.get("KIMI_API_KEY") and not os.environ.get(target):
            os.environ[target] = os.environ["KIMI_API_KEY"]


_load_env()

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


async def test_deepseek_basic() -> None:
    from sage.llm.base import LLMConfig, Message, Role
    from sage.providers.pydantic_ai_provider import PydanticAIProvider

    print("\n" + "=" * 70)
    print("DeepSeek — basic generate (system + user, no tools)")
    print("=" * 70)

    provider = PydanticAIProvider.for_sage_provider(
        "deepseek", "deepseek-chat", os.environ.get("DEEPSEEK_API_KEY")
    )
    messages = [
        Message(role=Role.SYSTEM, content="You are terse."),
        Message(role=Role.USER, content="Say 'hello' in 3 words max."),
    ]
    cfg = LLMConfig(provider="deepseek", model="deepseek-chat")
    resp = await provider.generate(messages, config=cfg)

    print(f"  content: {resp.content!r}")
    print(f"  model: {resp.model!r}")
    print(f"  stop_reason: {resp.stop_reason!r}")
    print(f"  usage: {resp.usage!r}")
    print(f"  tool_calls: {resp.tool_calls!r}")
    assert resp.content, "expected non-empty content"
    assert resp.usage, "expected usage populated"
    assert resp.usage["input_tokens"] > 0
    assert resp.usage["output_tokens"] > 0
    print("  ok ✓")


async def test_deepseek_with_tool() -> None:
    from sage.llm.base import LLMConfig, Message, Role, ToolDef
    from sage.providers.pydantic_ai_provider import PydanticAIProvider

    print("\n" + "=" * 70)
    print("DeepSeek — generate with a tool definition (forces tool call)")
    print("=" * 70)

    provider = PydanticAIProvider.for_sage_provider(
        "deepseek", "deepseek-chat", os.environ.get("DEEPSEEK_API_KEY")
    )
    tool = ToolDef(
        name="get_weather",
        description="Get current weather for a city",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
            "required": ["city"],
        },
    )
    messages = [
        Message(role=Role.USER, content="What's the weather in Paris?"),
    ]
    resp = await provider.generate(messages, tools=[tool])
    print(f"  content: {resp.content!r}")
    print(f"  tool_calls: {resp.tool_calls!r}")
    print(f"  usage: {resp.usage!r}")
    # Either content or tool_calls should be set
    assert resp.content or resp.tool_calls, "expected output or tool call"
    print("  ok ✓")


async def test_kimi_basic() -> None:
    from sage.llm.base import LLMConfig, Message, Role
    from sage.providers.pydantic_ai_provider import PydanticAIProvider

    print("\n" + "=" * 70)
    print("Kimi — basic generate via native moonshotai provider")
    print("=" * 70)

    provider = PydanticAIProvider.for_sage_provider(
        "kimi", "kimi-k2-0711-preview", os.environ.get("KIMI_API_KEY")
    )
    messages = [
        Message(role=Role.USER, content="Say 'hi' in 2 words max."),
    ]
    resp = await provider.generate(messages)
    print(f"  content: {resp.content!r}")
    print(f"  model: {resp.model!r}")
    print(f"  usage: {resp.usage!r}")
    assert resp.content, "expected non-empty content"
    print("  ok ✓")


async def test_cost_lookup() -> None:
    from sage.providers.pydantic_ai_provider import _lookup_cost_per_token

    print("\n" + "=" * 70)
    print("Cost lookup from cards.toml")
    print("=" * 70)
    for provider, model in [
        ("deepseek", "deepseek-chat"),
        ("openai", "gpt-5.4"),
        ("google", "gemini-3.1-flash-lite-preview"),
        ("kimi", "kimi-k2.6"),
        ("nonexistent", "no-such-model"),
    ]:
        in_cost, out_cost = _lookup_cost_per_token(provider, model)
        print(f"  {provider}/{model}: in={in_cost:.2e} $/tok, out={out_cost:.2e} $/tok")


async def main() -> None:
    try:
        await test_cost_lookup()
    except Exception as exc:
        print(f"  cost_lookup ERROR: {type(exc).__name__}: {exc}")

    try:
        await test_deepseek_basic()
    except Exception as exc:
        print(f"  DeepSeek basic ERROR: {type(exc).__name__}: {exc}")

    try:
        await test_deepseek_with_tool()
    except Exception as exc:
        print(f"  DeepSeek tool ERROR: {type(exc).__name__}: {exc}")

    try:
        await test_kimi_basic()
    except Exception as exc:
        print(f"  Kimi ERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
