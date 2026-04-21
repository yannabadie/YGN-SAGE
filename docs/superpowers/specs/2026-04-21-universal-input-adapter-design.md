# Universal Input Adapter — Design Spec

**Date:** 2026-04-21
**Status:** Draft, awaiting user review before implementation
**Goal:** YGN-SAGE accepts **any** input format — SWE-bench instance, BigCodeBench
task, raw chat prompt, ExoCortex research query — through a single normalized
entry point. No per-source hard-coded prompt template.

---

## Problem statement

Today each bench hand-rolls a `_TASK_TEMPLATE` in its bench module
(`sage.bench.swebench_bench._TASK_TEMPLATE`, `sage.bench.bigcodebench_bench`'s
prompt composition, etc.). Each bench encodes its own:

- Mandatory workflow (e.g. "You MUST make three execute_bash calls")
- Output format expectations (unified diff / fenced python code / JSON / free text)
- Tool hints

The same agent loop then receives these three different styles of prompts and
tries to cope. Consequences:

1. **Inconsistent tool affordance.** Commit `3622ac5` fixed the structural
   tool listing (native `ToolRegistry.describe_for_prompt()`), but per-bench
   prompt text still overrides — SWE-bench's "MUST bash" clause explains the
   `search_exocortex=0 calls` on the 2026-04-21 N=50 smoke.
2. **Chat is impossible today.** A user prompt "Summarize this paper" cannot
   be run through `system.run()` without the agent drowning in SWE-bench's
   "emit a unified diff" expectations. There is no "bench=chat" template and
   creating one would duplicate the antipattern.
3. **Benchmark instructions leak into every downstream.** The Windows-CRLF
   fix, Directive-#3 SSL gating, and the eval-harness code are all
   conflated with the per-bench prompt logic — SWE-bench-specific concerns
   spread across six Python modules.

## Proposed architecture

### TaskInput dataclass

```python
from dataclasses import dataclass, field
from enum import Enum

class ResponseFormat(str, Enum):
    TEXT = "text"            # Free-form prose. Default for chat.
    CODE = "code"            # Fenced python code block.
    PATCH = "patch"          # Unified diff patch (SWE-bench, codemods).
    JSON = "json"            # Structured JSON following a schema.
    SEARCH_REPLACE = "search_replace"  # Aider-style <<<<<<< SEARCH ... blocks.

@dataclass
class TaskInput:
    """The single normalized input to `AgentSystem.run()`.

    Any source (chat, SWE-bench, BCB, ExoCortex query, external API) gets
    converted to this shape before the pipeline sees it.
    """
    prompt: str                          # The user request, natural language.
    response_format: ResponseFormat = ResponseFormat.TEXT
    hints: dict = field(default_factory=dict)  # Structured context.
    instructions: str = ""               # Source-specific workflow additions.
    tools_filter: list[str] | None = None      # Restrict to these tools only.
    expected_length_hint: int = 0        # "Output ~50 words" etc. 0 = no hint.
    source: str = "chat"                 # "chat" | "swebench" | "bcb" | ...
```

### Normalizers (one per source)

Each bench / interface provides a pure function `normalize(raw) -> TaskInput`.
No global state. No inheritance hierarchy. Just mapping.

```python
# sage.input.chat
def normalize_chat(user_message: str) -> TaskInput:
    return TaskInput(
        prompt=user_message,
        response_format=ResponseFormat.TEXT,
        source="chat",
    )

# sage.input.swebench
def normalize_swebench(instance: dict) -> TaskInput:
    return TaskInput(
        prompt=instance["problem_statement"],
        response_format=ResponseFormat.PATCH,
        hints={
            "repo": instance["repo"],
            "base_commit": instance["base_commit"],
            "version": instance.get("version", "unknown"),
            "hints_text": instance.get("hints_text", ""),
        },
        instructions=SWEBENCH_WORKFLOW,  # Kept as a module-level constant.
        source="swebench",
    )

# sage.input.bcb
def normalize_bcb(task: dict) -> TaskInput:
    return TaskInput(
        prompt=task["prompt"],
        response_format=ResponseFormat.CODE,
        hints={
            "code_prompt": task.get("code_prompt", ""),
            "test": task.get("test", ""),
            "entry_point": task.get("entry_point", ""),
        },
        instructions=BCB_WORKFLOW,
        source="bcb",
    )
```

### Prompt builder (in `perceive` phase)

`perceive()` already composes the system_prompt from `config.system_prompt` +
validation-level augmentation + (since commit `3622ac5`) the
`ToolRegistry.describe_for_prompt()` block. Extend it to layer the TaskInput
on top:

```python
def build_prompts(task_input: TaskInput, loop: AgentLoop) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) given a normalized TaskInput."""
    # Base system prompt (same for everyone).
    sp = loop.config.system_prompt

    # Tool affordance (native, already wired).
    tools = loop._tools.describe_for_prompt(task_input.tools_filter)
    if tools:
        sp = f"{sp}\n\n{tools}"

    # Response-format expectations.
    sp = f"{sp}\n\n{_response_format_block(task_input.response_format, task_input.expected_length_hint)}"

    # Source-specific instructions (benches opt-in; chat gets none).
    if task_input.instructions:
        sp = f"{sp}\n\n## Workflow\n{task_input.instructions}"

    # The user prompt is the prompt field + any structured hints rendered as
    # a secondary context block.
    up = task_input.prompt
    if task_input.hints:
        hints_md = _render_hints_block(task_input.hints)
        up = f"{up}\n\n{hints_md}"

    return sp, up
```

### Pipeline entry point

```python
class AgentSystem:
    async def run(self, task: str | TaskInput) -> str:
        """Universal entry. Accepts chat string, TaskInput, or bench dict."""
        if isinstance(task, str):
            task_input = normalize_chat(task)
        elif isinstance(task, TaskInput):
            task_input = task
        else:
            raise TypeError(f"Unsupported task type {type(task)}")
        return await self.pipeline.run(task_input)
```

Each bench's `generate_patches()` call becomes one line:

```python
task_input = normalize_swebench(instance)
patch = await self.system.run(task_input)
```

## What changes per layer

| Layer | Today | After |
|-------|-------|-------|
| `swebench_bench._TASK_TEMPLATE` | 62-line Python string | Deleted. Replaced by `SWEBENCH_WORKFLOW` constant (10 lines) + `normalize_swebench()` (15 lines). |
| `bigcodebench_bench` prompt composition | Inlined | Moved to `sage.input.bcb.normalize_bcb()`. |
| Chat entry point | **Does not exist** | `normalize_chat(str) → TaskInput`. |
| `perceive()` prompt composition | Hardcoded `loop.config.system_prompt + validation augmentation` | Layered: system prompt + tool affordance + response format + source instructions + user prompt with hints block. |
| Response format enforcement | `_TASK_TEMPLATE` says "Output your final patch in a ```diff fence" | `_response_format_block(ResponseFormat.PATCH)` renders the same text. Reusable across any future patch-emitting bench. |

## What does NOT change

- The agent loop (`AgentLoop.run()`) takes a `task: str` today. I'd keep that
  signature at the loop level — `perceive()` consumes TaskInput and returns a
  built prompt string to `loop`. Minimally invasive.
- The 12 topology templates, the 6-path generation, the Rust SystemRouter,
  the ModelAssigner — untouched. This is a front-end reshape.
- All existing tests keep passing: normalizers produce the SAME string that
  the current `_TASK_TEMPLATE.format(...)` produces on a given instance. A
  regression test compares byte-identical output for a known SWE-bench sample.

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Changing SWE-bench prompt text invalidates the 2026-04-21 baseline | The first commit keeps the prompt byte-identical; the TaskInput layer is a reshape, not a content change. The AVR enrichment (commit `9eb05b0`) is separately represented via `instructions` + `hints`. |
| Per-bench normalizers scatter | Keep all three normalizers in `sage.input.*` (new package). One file per source, max 40 lines. Docs in the package `__init__.py`. |
| Chat mode needs tool restriction | `tools_filter` on TaskInput — a chat session might want only `search_exocortex` and no `execute_bash`. Set `tools_filter=["search_exocortex"]` in `normalize_chat()` by default; allow caller to expand. |
| Response format enforcement varies per model | Already handled by existing logic (validation_level, PATCH extraction in `_extract_patch`). `_response_format_block()` only adds the natural-language instruction; the strict extraction stays where it is. |
| Chat sessions don't know what "mandatory workflow" they need | Default: none. Chat mode is free-form. If a chat user asks "fix this bug in file X", the agent uses tools (search_exocortex, execute_bash if repo is mounted), no bench-specific template. |

## Migration plan

- **Commit 1**: Introduce `sage.input` package with `TaskInput`, `ResponseFormat`,
  and `normalize_chat()`. No bench changes yet. 20 unit tests.
- **Commit 2**: Migrate SWE-bench. Old `_TASK_TEMPLATE` becomes
  `SWEBENCH_WORKFLOW` fed through the new path. Regression test: a known
  instance produces byte-identical prompt text.
- **Commit 3**: Migrate BCB. Same pattern.
- **Commit 4**: Add `AgentSystem.run()` string/TaskInput overload.
- **Commit 5**: Chat CLI prototype (`python -m sage.chat`). Interactive REPL
  that calls `normalize_chat` and prints responses.

Each commit is independently testable and reversible.

## Out of scope (for now)

- **Streaming responses.** Chat mode probably wants token streaming; the
  loop already has `generate_stream` plumbing (`StreamingLLMProvider`
  protocol). Wiring into the new CLI is a follow-up.
- **Multi-turn chat state.** WorkingMemory already handles this per-run. A
  true persistent conversation would need session state — a separate design.
- **Graphical chat UI.** Out of scope for SAGE; the CLI is enough for dogfood.

## Expected outcomes

- `search_exocortex` can be called from chat mode naturally (no anti-affordance
  bash mandate). Directly measurable on a chat session that mentions a library.
- Per-bench prompt drift (each bench invents its own "MUST do X" pattern)
  replaced by one configurable TaskInput contract.
- New benches (HumanEval+, SWE-bench Pro, custom) require only a 40-line
  normalizer, not a full prompt-template authoring pass.
- The 2026-04-21 tool-affordance fix (`3622ac5`) composes cleanly with
  source-specific instructions instead of being overridden by them.

## Questions for user before implementation

1. **Chat tools filter.** In chat mode, should the agent have full tool
   access by default (execute_bash available) or restricted (no bash, only
   `search_exocortex` + maybe a web-fetch tool)? Default matters — chat that
   runs bash is powerful but surprising.
2. **Chat persistence.** Multi-turn chat needs session state (history,
   prior tool outputs). Use WorkingMemory (per-run) or a new `ChatSession`
   object? Affects where the history lives.
3. **Response format default for chat.** `ResponseFormat.TEXT` seems right
   but some chat users want `CODE` when they ask for code. Auto-detect from
   the prompt text or require an explicit mode switch?
4. **Migration pace.** 5 commits as described, or bundle 1+2+3+4 into one
   bigger commit and iterate on chat (5) separately? I'd prefer the
   incremental pace so regressions are attributable to one change.

---

**Status:** awaiting user answers on the 4 questions before starting
Commit 1.
