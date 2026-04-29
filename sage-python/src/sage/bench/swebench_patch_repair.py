"""Patch validation + programmatic/LLM repair for SWE-bench predictions.

Two-stage repair pipeline for malformed unified-diff patches emitted
by SAGE's agent loop. Motivated by the 2026-04-21 v15 smoke: 2/10
tasks (`astropy-7746`, `django-10914`) failed with `git apply` / `patch`
rejecting the hunk headers or context lines (LLM hallucination on
large files).

Pipeline:
  1. **Programmatic counts-fix** — ``_fix_hunk_header_counts(patch)``
     recomputes ``@@ -s,c +s,c @@`` from the real body line counts.
     Zero-cost, deterministic. Catches ``astropy-7746``-class errors
     (wrong counts) but not ``django-10914``-class errors (stale
     context lines that no longer exist in the file).

  2. **LLM repair** — ``repair_patch_via_llm()`` does a single one-shot
     direct LLM call (no tools, no agent state) with the ``git apply``
     stderr as feedback. Aider's "apply-with-feedback" pattern.

Validator:
  ``validate_patch_apply(patch, repo_dir)`` runs ``git apply --check``
  in the cloned repo. Returns ``(ok, stderr)``. 10-second timeout caps
  pathological cases.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import subprocess
from typing import Any

log = logging.getLogger(__name__)

# ``@@ -<start>[,<count>] +<start>[,<count>] @@ <trailer>``
# When count is omitted, unified-diff spec says count=1. We treat
# missing count as 1 on parse and always write an explicit count on
# output to avoid ambiguity.
_HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<src_start>\d+)(?:,(?P<src_count>\d+))? "
    r"\+(?P<dst_start>\d+)(?:,(?P<dst_count>\d+))? @@(?P<trailer>.*)$"
)


def _fix_hunk_header_counts(patch: str) -> str:
    """Recompute ``-s,c +s,c`` counts from the hunk body.

    Deterministic fix for the ``astropy-7746``-class failure where
    the LLM emits ``@@ -1264,28 +1279,35 @@`` but the hunk body
    doesn't actually contain 28 source / 35 destination lines. GNU
    patch and ``git apply`` both reject these; the counts can be
    recomputed directly from the body.

    Counting rules per the unified-diff spec (RFC 2822-like):
      - ``' '`` (context): counts in both src_count and dst_count
      - ``'-'`` (removed): counts only in src_count
      - ``'+'`` (added):   counts only in dst_count
      - ``'\\'`` (no-newline marker): ignored
      - Empty line: treated as context ``' '`` (most tools do this)

    No-op on well-formed patches (recomputed counts match original).
    """
    lines = patch.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _HUNK_HEADER_RE.match(line)
        if m is None:
            out.append(line)
            i += 1
            continue

        # Find end of this hunk body. Body line MUST start with one of
        # ``' '`` / ``'-'`` / ``'+'`` / ``'\\'`` per the unified-diff
        # spec. An empty line or any other char ends the body — this
        # handles both the trailing ``""`` artefact of ``split("\n")``
        # and the "prose between hunks" edge case where the LLM wraps
        # its diff in explanatory text.
        body_start = i + 1
        body_end = body_start
        while body_end < len(lines):
            nxt = lines[body_end]
            if not nxt:
                break
            c0 = nxt[0]
            if c0 not in (" ", "-", "+", "\\"):
                # Could be the start of another hunk / diff header or
                # prose. Stop.
                break
            body_end += 1

        body = lines[body_start:body_end]

        src_count = 0
        dst_count = 0
        for b in body:
            c = b[0]  # body guaranteed non-empty with a known prefix above
            if c == " ":
                src_count += 1
                dst_count += 1
            elif c == "-":
                src_count += 1
            elif c == "+":
                dst_count += 1
            # c == "\\" -> "\ No newline at end of file" -> doesn't count

        src_start = int(m.group("src_start"))
        dst_start = int(m.group("dst_start"))
        trailer = m.group("trailer") or ""
        new_header = (
            f"@@ -{src_start},{src_count} "
            f"+{dst_start},{dst_count} @@{trailer}"
        )
        out.append(new_header)
        out.extend(body)
        i = body_end

    return "\n".join(out)


def validate_patch_apply(patch: str, repo_dir: str) -> tuple[bool, str]:
    """Run ``git apply --check`` against ``repo_dir``.

    Returns ``(ok, stderr_text)``. ``--check`` validates without
    modifying the working tree. 10-second timeout.

    ``repo_dir`` must be a git clone at the instance's ``base_commit``
    (which is what ``SWEBenchBench._setup_repo`` already produces).
    """
    if not patch.strip():
        return False, "empty patch"
    try:
        result = subprocess.run(
            ["git", "apply", "--check", "-"],
            input=patch.encode("utf-8"),
            cwd=repo_dir,
            capture_output=True,
            timeout=10,
        )
        ok = result.returncode == 0
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        return ok, stderr
    except subprocess.TimeoutExpired:
        return False, "git apply --check timed out after 10s"
    except FileNotFoundError:
        return False, "git not found on PATH"
    except Exception as exc:  # noqa: BLE001 — surface any validator fault
        return False, f"validator exception: {type(exc).__name__}: {exc}"


_REPAIR_PROMPT_TEMPLATE = """\
The unified-diff patch below was rejected by `git apply`. Fix it.

## Original issue

{problem_statement}

## Patch that failed to apply

```diff
{broken_patch}
```

## `git apply --check` stderr

```
{error_msg}
```

Output ONLY a corrected unified diff inside a single ```diff fenced \
block. Fix hunk-header counts, realign context lines to the real \
source, or rewrite the hunks if the line numbers are unrecoverable. \
No prose, no explanation — just the diff.\
"""


async def repair_patch_via_llm(
    llm: Any,
    problem_statement: str,
    broken_patch: str,
    error_msg: str,
    timeout: float = 60.0,
) -> str:
    """One-shot LLM repair of a failed patch. No tools, no agent state.

    Aider-style "apply-with-feedback" — the LLM sees the stderr from
    ``git apply`` and emits a corrected diff. Direct call on the
    underlying ``LLMProvider`` (not through the agent loop) so the
    exploration state from the original run doesn't bleed in.

    ``llm`` must expose ``async def generate(messages, ...)``
    returning an object with a ``.content`` attribute (see
    ``sage.llm.base.LLMProvider`` protocol).

    Returns the extracted diff text, or ``""`` on failure / empty
    response.
    """
    from sage.llm.base import Message, Role

    try:
        from sage.bench.swebench_bench import _extract_patch
    except ImportError:
        # Defensive: if module path changes, degrade to raw content.
        def _extract_patch(s: str) -> str:  # type: ignore[misc]
            return s

    repair_prompt = _REPAIR_PROMPT_TEMPLATE.format(
        problem_statement=(problem_statement or "")[:2000],
        broken_patch=broken_patch,
        error_msg=(error_msg or "unknown error")[:1000],
    )

    try:
        response = await asyncio.wait_for(
            llm.generate(
                messages=[Message(role=Role.USER, content=repair_prompt)],
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        log.warning("[patch-repair] LLM repair timed out after %.0fs", timeout)
        return ""
    except Exception as exc:  # noqa: BLE001
        log.warning("[patch-repair] LLM repair failed: %s", exc)
        return ""

    return _extract_patch(response.content or "")


def _normalize_line_endings(patch: str) -> str:
    """Convert any ``\\r\\n`` / ``\\r`` → ``\\n`` so ``git apply`` and
    GNU ``patch`` don't reject the diff as "corrupt patch at line N".

    Motivation (2026-04-24 A6): astropy-6938 in the 2026-04-24 observe
    smoke emitted a valid-looking diff that Docker's ``patch`` rejected
    with ``corrupt patch at line 18``. Byte inspection showed ``\\r\\n``
    line endings — the emission path normalizes CRLF → LF on the agent
    response (``swebench_bench._extract_patch_from_response:216``), but
    the predictions get re-serialised by Python text-mode ``open()`` on
    Windows, which CRLF-translates on the way back out. The SWE-bench
    harness then writes the ``patch.diff`` inside the Docker context
    via the same text-mode open, propagating the CRLF inside the
    container where GNU ``patch`` chokes.

    Normalizing here is belt-and-suspenders: even if the upstream write
    path introduces CRLF, the repair pipeline starts from LF-only bytes
    and every downstream validation (``git apply --check``, LLM repair
    feedback) sees clean input. Idempotent on LF-only input.
    """
    if "\r" not in patch:
        return patch
    return patch.replace("\r\n", "\n").replace("\r", "\n")


async def try_repair_patch(
    patch: str,
    repo_dir: str | None,
    llm: Any,
    problem_statement: str,
    instance_id: str = "",
    llm_timeout: float = 60.0,
    run_frame_builder: Any | None = None,
    run_id: str | None = None,
    node_run_id: str | None = None,
    event_seq: int | None = None,
    source_id: str = "swebench:patch_repair",
) -> tuple[str, str]:
    """Validate + repair pipeline. Returns ``(final_patch, repair_stage)``.

    ``repair_stage`` ∈ ``{"", "unchanged", "programmatic_counts",
    "llm_repair", "crlf_normalized", "failed"}``:
      - ``""``: validation was skipped (empty patch or no repo_dir).
      - ``"unchanged"``: original patch validated, no repair needed.
      - ``"crlf_normalized"``: pure CRLF → LF fix alone resolved the
        patch. Common on Windows (A6, 2026-04-24).
      - ``"programmatic_counts"``: ``_fix_hunk_header_counts`` fixed it
        (possibly after CRLF normalization).
      - ``"llm_repair"``: LLM one-shot fixed it.
      - ``"failed"``: no stage produced a valid patch — original
        returned unchanged so the downstream swebench evaluator can
        still classify as apply-error (same behavior as pre-fix).
    """
    if not patch or not repo_dir:
        return patch, ""

    def _finish(final_patch: str, repair_stage: str) -> tuple[str, str]:
        _record_repair_delta(
            patch=final_patch,
            repair_stage=repair_stage,
            run_frame_builder=run_frame_builder,
            run_id=run_id,
            node_run_id=node_run_id,
            event_seq=event_seq,
            source_id=source_id,
        )
        return final_patch, repair_stage

    # Stage 0 (A6, 2026-04-24): LF normalization. CRLF bytes in the
    # patch are a Windows-emission artefact that both ``git apply`` and
    # GNU ``patch`` reject as "corrupt patch". Belt-and-suspenders even
    # if the upstream write path already handled it.
    normalized = _normalize_line_endings(patch)
    crlf_was_fixed = normalized != patch
    patch = normalized

    ok, err = validate_patch_apply(patch, repo_dir)
    if ok:
        if crlf_was_fixed:
            log.info(
                "[%s] CRLF normalization resolved patch", instance_id or "?",
            )
            return _finish(patch, "crlf_normalized")
        return _finish(patch, "unchanged")

    log.info(
        "[%s] patch validation failed: %s",
        instance_id or "?", err[:200],
    )

    # Stage 1: programmatic fix (free).
    fixed = _fix_hunk_header_counts(patch)
    if fixed != patch:
        ok2, err2 = validate_patch_apply(fixed, repo_dir)
        if ok2:
            log.info("[%s] programmatic counts-fix resolved patch", instance_id or "?")
            return _finish(fixed, "programmatic_counts")
        err = err2 or err  # latest error feeds LLM repair

    # Stage 2: LLM repair (costs one call).
    if llm is None:
        return _finish(patch, "failed")

    repaired = await repair_patch_via_llm(
        llm, problem_statement, patch, err, timeout=llm_timeout,
    )
    if not repaired:
        return _finish(patch, "failed")

    ok3, err3 = validate_patch_apply(repaired, repo_dir)
    if ok3:
        log.info("[%s] LLM repair resolved patch", instance_id or "?")
        return _finish(repaired, "llm_repair")

    log.info(
        "[%s] LLM repair still invalid: %s",
        instance_id or "?", (err3 or "no stderr")[:200],
    )
    return _finish(patch, "failed")


def _record_repair_delta(
    *,
    patch: str,
    repair_stage: str,
    run_frame_builder: Any | None,
    run_id: str | None,
    node_run_id: str | None,
    event_seq: int | None,
    source_id: str,
) -> None:
    if os.environ.get("SAGE_ORACLE") != "1" or run_frame_builder is None:
        return
    from sage.runtime.evidence.producers.diff import produce_diff_verifier_deltas

    patch_hash = hashlib.sha256(patch.encode("utf-8", errors="replace")).hexdigest()
    verify_outcome = "clean" if repair_stage and repair_stage != "failed" else "unsupported_no_opinion"
    producer_result = produce_diff_verifier_deltas(
        run_id=run_id or run_frame_builder.run_id,
        node_run_id=node_run_id,
        event_seq=event_seq,
        source_id=source_id,
        verify_result={
            "outcome": verify_outcome,
            "mismatches": [],
            "reasons": [verify_outcome],
            "patch_hash": patch_hash,
        },
        repair_result={
            "repair_stage": repair_stage,
            "patch_hash": patch_hash,
        },
    )
    for delta in producer_result.deltas:
        run_frame_builder.record_delta(delta)


__all__ = [
    "_fix_hunk_header_counts",
    "validate_patch_apply",
    "repair_patch_via_llm",
    "try_repair_patch",
]
