# roadmap-A2 — SWE-bench fast-abort diagnosis (2026-04-24)

**Status:** Diagnosis complete. Root cause identified. Fix partially shipped (`df150a2a` 2026-04-24 14:55). **Post-fix verification pending** — no smoke run after `df150a2a` yet.

**Investigation scope:** the 20% fast-abort rate observed on the 2026-04-23 observe-mode smoke (2/10 aborted <60s).

---

## Findings

### Evidence

2026-04-23 observe smoke (`docs/benchmarks/2026-04-23-diff-verifier-observe-smoke/`):

| Task | Latency | Cost | Error |
|---|---:|---:|---|
| astropy-7746 | 42.7s | $0.93 | empty_patch |
| astropy-14182 | 45.6s | $0.00 | empty_patch |
| astropy-14365 | 60.0s | $0.00 | empty_patch |

3 fast-aborts, not 2. All 3 have `error='empty_patch'`.

2026-04-24 **pre-fix** smoke (09:26, before `df150a2a` at 14:55):

| Task | Latency | Cost | Error |
|---|---:|---:|---|
| astropy-14182 | 57.8s | $0.76 | empty_patch |
| astropy-7746 | 71.5s | $0.84 | empty_patch |
| django-10924 | 87.1s | $0.87 | empty_patch |

Pattern shifted but persists.

### Root cause

All 6 fast-aborts (across both smokes) share the same log signature:

```
HTTP Request: POST https://api.moonshot.ai/v1/chat/completions "HTTP/1.1 400 Bad Request"
Stage 4 multi-agent execution failed: status_code: 400, model_name: kimi-k2.5,
  body: {'message': 'thinking is enabled but reasoning_content is missing
  in assistant tool call message at index 3', ...}
  → falling back to single-agent
AFC is enabled with max remote calls: 10.
...
Stage 4 fallback also failed: Stage 4 fallback returned empty content
  → treating as failure rather than emitting empty patch
```

**Cascade chain:**

1. Stage 4 multi-agent topology routes work to `kimi-k2.5` with `thinking` enabled.
2. Kimi's thinking mode requires `reasoning_content` on **every assistant message that contains tool_calls** (per Moonshot spec — reasoning_content precedes tool_calls in the streamed response, and reconstructed multi-turn messages follow the same convention).
3. `PydanticAIProvider.generate_via_pydantic_ai()` message translation was dropping `Message.thinking` when rebuilding the conversation for the NEXT turn — specifically at "index 3", the 4th message in the stitched history.
4. Kimi returns HTTP 400.
5. Stage 4 falls back to single-agent via `gemini-3.1-flash-lite-preview`, which also returns empty content (the fallback doesn't carry over the tool-use context built up by the multi-agent nodes).
6. Pipeline aborts with `empty_patch`.

The "fast" aspect isn't a timeout — it's **failure short-circuiting** through the multi-agent Stage 4 → fallback → abort chain before any real generation happens.

### Fix (partial, already shipped)

Commit `df150a2a` (2026-04-24 14:55) — **AFTER both observed smokes**:

1. `sage-python/src/sage/providers/pydantic_ai_provider.py:247-274` — outgoing message path: emit `ThinkingPart` **before** `TextPart`/`ToolCallPart` so PydanticAI's OpenAI model profile (which sets `openai_chat_send_back_thinking_parts='field'` for Moonshot/DeepSeek-thinking) serializes it back as `reasoning_content` on the assistant message.

2. `sage-python/src/sage/phases/act.py:201-374` — 3 sites propagate `response.thinking` into `Message.thinking` so the next turn's message history carries it.

3. `sage-python/src/sage/providers/pydantic_ai_provider.py:315-320 + :341` — response path: extract `ThinkingPart` from Kimi's response into `LLMResponse.thinking`.

### Gap: post-fix verification missing

No smoke has run since `df150a2a`. Evidence that the fix actually closes the bug is absent. Two paths to close:

1. **Inexpensive**: add a unit test that reconstructs the exact message sequence (multi-turn tool-call with thinking) and asserts the outgoing Pydantic AI `ModelMessage` serializes `reasoning_content` at every assistant tool-call turn. No API calls.

2. **Expensive but definitive**: re-run a ≥10-task SWE-bench smoke (task #118 covers this at N=50). Budget ~$20-40.

### Additional observation — CEGAR repair path

`sage-python/src/sage/agent_loop_execution.py:227-231` builds a message list for the CEGAR verification-failure repair pass **without** threading `thinking` onto the assistant message:

```python
messages = [
    Message(role=Role.SYSTEM, content=system_prompt),
    Message(role=Role.ASSISTANT, content=content),  # ← no thinking=
    Message(role=Role.USER, content=repair_prompt),
]
```

If this path is routed to Kimi with thinking enabled, it will hit the same HTTP 400. The 2026-04-23/24 smokes didn't trigger this because CEGAR isn't the dominant path; it's a latent second-order risk.

**Suggested defensive fix:** thread the caller's `thinking` value onto the synthetic assistant message in `agent_loop_execution.py:229`. ~1 LOC + a unit test.

---

## Status summary

| Item | State |
|---|---|
| Root cause identified | ✅ Kimi `reasoning_content` missing in multi-turn tool-call history |
| Fix shipped | ✅ Partial — `df150a2a` covers the primary outgoing message path |
| Fix verified | ❌ No post-fix smoke run |
| CEGAR repair path latent risk | ⚠️ Not covered by `df150a2a`; 1-LOC defensive fix worth adding |

**Next steps (non-blocking for roadmap-A2 close):**

1. (cheap) Add unit test reproducing the 4-turn tool-call message flow; assert `reasoning_content` present at every assistant tool-call turn after PydanticAI translation.
2. (cheap) CEGAR defensive fix — thread `thinking` onto `agent_loop_execution.py:229` synthetic assistant message.
3. (gated on budget) Run N=10 observe smoke post-fix; confirm fast-aborts drop to 0.
4. Fold N=10 result into task #118 paired N=50 smoke when that kicks off.

The diagnosis phase of roadmap-A2 is **done**. Remaining work is verification (unit test + micro-smoke), which falls under either roadmap-A3 (task #118) or a dedicated verification commit.
