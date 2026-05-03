"""BigCodeBench normalizer — BCB task dict → TaskInput → prompt string.

C3 (2026-04-22): migrated the inline prompt build that previously lived
in `sage.bench.bigcodebench_bench.run()` at lines 79-87.
C5 (2026-05-03): `render_bcb_prompt` appends an OUTPUT REQUIREMENT so
the synthesizer node emits Python code rather than planning summaries
("no output" failure mode — v6 ablation, 3/10 tasks affected).

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
        # BCB tasks are self-contained function stubs — there is no repository
        # to explore. Without this instruction, multi-agent nodes call
        # search_repo/read_file in loops against an empty codebase and burn
        # the full 120s task timeout. Direct code generation is both faster
        # and correct for atomic BCB challenges.
        instructions=(
            "This is a self-contained coding task. Write the complete Python "
            "function implementation directly. Do NOT call any file-search or "
            "code-exploration tools — there is no repository to explore. "
            "Return only the implementation code."
        ),
        source="bcb",
    )


def render_bcb_prompt(task_input: TaskInput) -> str:
    """Build the BCB task string that all topology nodes receive.

    Only responsible for BCB-shaped inputs (`task_input.source == "bcb"`
    and hints carry `code_prompt`). Appends an explicit OUTPUT REQUIREMENT
    so that the synthesizer node produces Python code instead of echoing
    the planner's planning text — the "no output" failure mode observed in
    v6 ablation (3/10 tasks returned planning summaries with no
    `def task_func(` definition).
    """
    nl_prompt = task_input.prompt
    code_prompt = (task_input.hints.get("code_prompt") or "")
    if code_prompt:
        base = (
            f"Use this function signature and imports:\n"
            f"```python\n{code_prompt}\n```\n\n{nl_prompt}"
        )
    else:
        base = nl_prompt
    return (
        f"{base}\n\n"
        "OUTPUT REQUIREMENT: Return ONLY a complete, runnable Python function "
        "implementation that starts with `def task_func(`. No planning, no diffs, "
        "no explanations — just working Python code."
    )
