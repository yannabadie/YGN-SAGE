"""SWE-bench normalizer — instance dict → TaskInput → prompt string.

C2a (2026-04-21): byte-identical migration of the previous inline
`_TASK_TEMPLATE` / `_build_task_prompt` in `sage.bench.swebench_bench`.
The template text is preserved verbatim — including the
"MUST make at least THREE distinct execute_bash calls"
anti-affordance that the 2026-04-21 ExoCortex audit flagged. C2b
softens that clause in a separate commit so the attribution window
for any tool-usage shift is unambiguous.

Downstream wiring (C4) will teach `perceive()` to consume `TaskInput`
directly. Until then, `render_swebench_prompt()` reproduces the exact
byte sequence the bench used to emit, and `swebench_bench._build_task_prompt`
calls through it.
"""
from __future__ import annotations

from typing import Any

from sage.input.types import ResponseFormat, TaskInput


SWEBENCH_SYSTEM_TEMPLATE = """\
You are an expert software engineer working inside a checked-out repository \
clone. Your job is to resolve a GitHub issue by producing a minimal unified \
diff patch.

## Repository
- **Repo:** {repo}
- **Version:** {version}
- **Base commit:** {base_commit}
- **Working directory:** the repo is checked out in the current directory. \
Use relative paths (no absolute paths, no /tmp/...).

## Issue Description

{problem_statement}

{hints_section}\
## Mandatory Workflow — Follow Every Step

You have these tools available:
- **execute_bash** — run any shell command (cat, grep, find, git log, sed, python, pytest)
- Memory + knowledge tools registered at boot (if any)

You MUST make at least THREE distinct execute_bash calls *before* writing any \
patch. One-shot patches are almost always wrong — the line numbers and \
context lines will not match the real source, and the harness will reject \
the diff.

1. **Locate** the code mentioned in the issue.
   Example: `grep -RIn "ClassName\\|function_name" src/ tests/ | head -40`
2. **Read** the full function/class being modified (not just the snippet in the issue).
   Example: `sed -n '200,260p' src/package/module.py`
3. **Check tests** that reference the target. They often reveal the contract.
   Example: `grep -RIn "function_name" tests/ | head -20`
4. **Verify** hunk line numbers immediately before emitting them.
   Example: `grep -n "^def function_name" src/package/module.py`
5. Reason about the minimal change. If unsure, read more. Never guess line numbers.
6. Write the patch.

## Patch Format — Strict

Output your final patch in unified diff format inside a fenced ```diff block:

```diff
diff --git a/path/to/file.py b/path/to/file.py
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -<start>,<count> +<start>,<count> @@ <optional context>
 unchanged line
-removed line
+added line
 unchanged line
```

Hard requirements:
- `diff --git` headers and `--- a/` / `+++ b/` paths MUST use forward slashes.
- Every context and removed line MUST match the real source exactly — you \
verified this with execute_bash.
- Hunk ranges (`@@ -s,c +s,c @@`) MUST be correct. Re-check with grep -n.
- Keep the change minimal. Do not refactor unrelated code."""


def normalize_swebench(instance: dict[str, Any]) -> TaskInput:
    """Map a SWE-bench instance dict to a `TaskInput`.

    Instance keys consumed: `problem_statement` (required), `repo`
    (required), `version` (optional, defaults to `"unknown"`),
    `base_commit` (required), `hints_text` (optional).

    The `problem_statement` becomes the `prompt` field; the rest
    flows through `hints`. `instructions` is set to the full
    SWE-bench workflow template so C4 can render it through the
    generic prompt builder while C2a's byte-identical renderer stays
    wired to `_build_task_prompt`.
    """
    return TaskInput(
        prompt=instance["problem_statement"],
        response_format=ResponseFormat.PATCH,
        hints={
            "repo": instance["repo"],
            "version": instance.get("version", "unknown"),
            "base_commit": instance["base_commit"],
            "hints_text": instance.get("hints_text", "") or "",
        },
        instructions=SWEBENCH_SYSTEM_TEMPLATE,
        source="swebench",
    )


def render_swebench_prompt(task_input: TaskInput) -> str:
    """Reproduce the pre-C2a `_build_task_prompt` output **byte-for-byte**.

    This function is only responsible for SWE-bench-shaped inputs (i.e.
    `task_input.source == "swebench"` and hints carry the four expected
    keys). The generic prompt builder that lands in C4 will replace this
    with a layered composition; until then, byte-identity is what makes
    the refactor safe to merge without disturbing the running smoke
    baseline.
    """
    hints = task_input.hints
    hints_text = (hints.get("hints_text") or "").strip()
    hints_section = (
        f"## Hints (from the issue comments)\n\n{hints_text}\n\n"
        if hints_text
        else ""
    )

    return SWEBENCH_SYSTEM_TEMPLATE.format(
        repo=hints["repo"],
        version=hints.get("version", "unknown"),
        base_commit=hints["base_commit"],
        problem_statement=task_input.prompt,
        hints_section=hints_section,
    )
