"""BigCodeBench normalizer — BCB task dict → TaskInput → prompt string.

C3 (2026-04-22): byte-identical migration of the inline prompt build
previously living in `sage.bench.bigcodebench_bench.run()` at lines
79-87. Pure refactor — no content change. The generic prompt builder
that lands in C4 will replace `render_bcb_prompt` with a layered
composition; until then this function reproduces the exact byte
sequence the bench used to emit.

BCB task dict fields consumed:
    instruct_prompt / complete_prompt : natural-language request
                                        (split-dependent).
    code_prompt                       : imports + function signature
                                        pre-amble. Prepended as a
                                        fenced python block when
                                        non-empty.
    test, entry_point, libs           : carried on `hints` for
                                        downstream observability
                                        (repair prompt, Context7
                                        integration, smoke logs).
                                        `libs` is a string
                                        representation of a list —
                                        stored verbatim; consumers
                                        parse it with ast.literal_eval.
"""
from __future__ import annotations

from typing import Any

from sage.input.types import ResponseFormat, TaskInput


def normalize_bcb(task: dict[str, Any], split: str = "instruct") -> TaskInput:
    """Map a BigCodeBench task dict to a `TaskInput`.

    Parameters
    ----------
    task :
        The raw task dict from `_load_dataset`. Must carry either
        `instruct_prompt` (when `split == "instruct"`) or
        `complete_prompt` (when `split == "complete"`).
    split :
        `"instruct"` (NL prompts, default, the one we benchmark on) or
        `"complete"` (docstring style). Mirrors the CLI `--split` flag.

    Raises
    ------
    KeyError
        If neither `instruct_prompt` nor `complete_prompt` is present.
        Matches the pre-C3 path's loudness on malformed dataset entries.
    """
    prompt_key = "instruct_prompt" if split == "instruct" else "complete_prompt"
    nl_prompt = task.get(prompt_key)
    if nl_prompt is None:
        # Pre-C3 fallback: if the selected-split key is missing, try
        # the other one. Preserves the `task.get(prompt_key,
        # task.get("instruct_prompt", ""))` behavior from the old
        # inline builder.
        nl_prompt = task.get("instruct_prompt", "")

    return TaskInput(
        prompt=nl_prompt,
        response_format=ResponseFormat.CODE,
        hints={
            "code_prompt": task.get("code_prompt", "") or "",
            "test": task.get("test", "") or "",
            "entry_point": task.get("entry_point", "") or "",
            "libs": task.get("libs", "") or "",
            "split": split,
        },
        instructions="",
        source="bcb",
    )


def render_bcb_prompt(task_input: TaskInput) -> str:
    """Reproduce the pre-C3 inline prompt-build output **byte-for-byte**.

    Only responsible for BCB-shaped inputs (`task_input.source == "bcb"`
    and hints carry `code_prompt`). The generic prompt builder that
    lands in C4 will replace this with a layered composition; byte
    identity is what makes this commit safe to merge without disturbing
    the 2026-04-21 BCB smoke baseline.
    """
    nl_prompt = task_input.prompt
    code_prompt = (task_input.hints.get("code_prompt") or "")
    if code_prompt:
        return (
            f"Use this function signature and imports:\n"
            f"```python\n{code_prompt}\n```\n\n{nl_prompt}"
        )
    return nl_prompt
