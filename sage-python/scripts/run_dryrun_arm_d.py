#!/usr/bin/env python3
"""Cycle-13 E Tier 2.1 — Arm D smoke runner.

Reads N task metadata from `instances.json` (produced by
`swebench_pro_fetch.py`), runs each task through `python -m sage.cli
run --jsonl`, captures the `final_result` event, extracts a unified
diff (if any), formats as SWE-bench Pro record, writes
`predictions.json` + per-task event traces.

Per cgpro DESIGN E (2026-05-05, conv `cgpro_pi_mono_pivot_20260505`,
verdict GO_TIER_1_PLUS_2 sub-stage 2.1):

  Tier 2.1 acceptance: 1/1 graded real Arm D task minimum, 2/2 only
  if Docker/image/runtime is not the bottleneck. Hard cutoff: if
  Docker pull/eval > 15 min OR API spend > $5, stop.

Modes:
  --mock              Synthetic empty patch per task (NO API spend).
                      Validates fetch script + format_patch wiring.
                      Tier 2.0 expansion. Acceptance: shape-valid
                      predictions.json produced.
  (default)           Real LLM via `python -m sage.cli run --jsonl`.
                      Default model: budget tier (deepseek-v4-flash);
                      override via SAGE_LLM_TIER env var.
                      Acceptance: shape-valid predictions.json +
                      per-task RuntimeEventLog file present.

Output:
  <output-dir>/predictions.json     — Pro grader input format
  <output-dir>/per_task/<id>.events.jsonl  — full event trace per task
  <output-dir>/summary.json         — aggregated metrics

Grading is OUT OF SCOPE here. The grader (`swe_bench_pro_eval.py`
in scaleapi/SWE-bench_Pro-os) requires Docker daemon running OR
Modal account + per-instance dockerfiles + run_scripts. Run grader
separately on the produced predictions.json.

Usage:
  # Mock smoke (no API):
  python -m sage_python.scripts.run_dryrun_arm_d \\
      --instances-json sage-python/data/swebench_pro/n10/instances.json \\
      --limit 1 --mock \\
      --output-dir sage-python/data/swebench_pro/arm_d_smoke_mock_n1

  # Real smoke (1 task, ~$0.50-1):
  python -m sage_python.scripts.run_dryrun_arm_d \\
      --instances-json sage-python/data/swebench_pro/n10/instances.json \\
      --limit 1 --budget-usd 5.0 \\
      --output-dir sage-python/data/swebench_pro/arm_d_smoke_real_n1
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
import hashlib
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Slice 7 of `canary-stage-timing-budget` (2026-05-11): use the canonical
# SWE-bench prompt builder + patch extractor. Without these, the canary
# sent a bare "Produce a unified diff" string and the agent returned
# synthesizer reasoning text — 0/5 patches across the first real N=5
# run (cumulative_cost_usd=$0.36, no provider_call_timeouts, 0 diffs).
# These helpers live in sage-python source (library code, not runtime
# contracts) so the harness can stay decoupled from runtime internals.
from sage.bench.swebench_bench import _extract_patch as _swebench_extract_patch
from sage.input.swebench import normalize_swebench, render_swebench_prompt

log = logging.getLogger("sage.bench.run_dryrun_arm_d")

# ── Slice 9 / `canary-patch-focused-prompt-profile` (cgpro DESIGN 2026-05-11) ──
# After slice 8 (cwd=cloned_repo + dotenv + TRACE_RAW), the canary still
# hits 0/5 patches because the canonical SWEBENCH_SYSTEM_TEMPLATE
# mandates "≥3 distinct tool calls" before emission. With stochastic
# topology selection (debate ~60% via bandit), debater roles can't
# satisfy the tool-call quota and the agent exits with EMPTY_STEP_SENTINEL.
#
# This slice adds a canary-local SWE-bench prompt PROFILE selectable
# via ``--swebench-prompt-profile {canonical,patch_focused}``:
#
# - ``canonical`` (default): pass-through to ``render_swebench_prompt``
#   from ``sage.input.swebench`` — byte-identical to slice 7. Existing
#   callers unaffected.
#
# - ``patch_focused``: canary-local template that
#     * keeps the expert-software-engineer framing,
#     * keeps the repo-grounding (says "the repo is checked out in
#       your working directory"),
#     * RECOMMENDS (not requires) typed tools,
#     * REMOVES the "MUST make at least THREE distinct tool calls
#       before emitting a patch" mandate,
#     * keeps the Patch Format section STRICTER (unified diff in
#       fenced ```diff block with diff --git / --- a/ / +++ b/ /
#       matching context line headers).
#
# Per cgpro DESIGN NON_GOALS: no topology override, no system_hint
# force, no edit to sage.input.swebench.
_PROMPT_PROFILE_CANONICAL = "canonical"
_PROMPT_PROFILE_PATCH_FOCUSED = "patch_focused"
_PROMPT_PROFILES: tuple[str, ...] = (
    _PROMPT_PROFILE_CANONICAL,
    _PROMPT_PROFILE_PATCH_FOCUSED,
)
_DEFAULT_PROMPT_PROFILE = _PROMPT_PROFILE_CANONICAL


# Inline template for patch_focused (defined here so this slice does
# NOT edit sage/input/swebench.py per cgpro DESIGN). The `{...}`
# placeholders are filled at render time from the instance dict.
_PATCH_FOCUSED_TEMPLATE = """\
You are an expert software engineer working inside a checked-out repository clone. Your job is to resolve a GitHub issue by producing a minimal, surgical unified diff patch.

## Repository
- **Repo:** {repo}
- **Version:** {version}
- **Base commit:** {base_commit}
- **Working directory:** the repo is checked out in your current working directory. Use relative paths (no absolute paths, no /tmp/...).

## Issue Description

{problem_statement}

{hints_section}\
## Tools (recommended, NOT required)

You have these tools available. They are useful for verifying line numbers and context lines BEFORE emitting your patch. There is NO minimum number of tool calls required:

- **read_file(path, max_bytes)** — read a text file in the checked-out repo.
- **search_repo(query, path, max_results, regex)** — search the repo for a pattern.
- **list_files(path, pattern, max)** — glob files under a relative root.
- **git_diff(path, staged, extra_args)** — show the current working-tree diff.
- **apply_patch(diff, check_only)** — apply a unified diff via `git apply` (check_only=True for dry-run).

If you are confident about the fix without consulting these tools, you may emit your patch directly. If you are uncertain about hunk line numbers, USE the tools — fabricated hunk numbers will be rejected by the grader.

## Patch Format — STRICT

Your final output MUST be a unified diff inside a fenced ```diff block. Without the fenced block, the harness will record an empty patch.

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
- `diff --git` header MUST use forward slashes.
- `--- a/` and `+++ b/` paths MUST use forward slashes and MUST match the `diff --git` paths.
- Every context line and removed line MUST match the real source verbatim. If you guess, the patch will fail `git apply`.
- Hunk headers (`@@ -s,c +s,c @@`) MUST be correct. If unsure, use `read_file` or `search_repo` to verify before emitting.
- Keep the change minimal. Do not refactor unrelated code.
- Output ONLY the fenced ```diff block as your final answer. Reasoning text BEFORE the block is allowed; reasoning text AFTER is not.
"""


def _render_patch_focused_prompt(task: dict[str, Any]) -> str:
    """Render the patch_focused prompt profile from an instance dict.

    Uses the same instance fields as ``sage.input.swebench.normalize_swebench``:
    ``problem_statement`` (required), ``repo``, ``base_commit``,
    ``version`` (defaults "unknown"), ``hints_text`` (optional).
    """
    repo = task.get("repo") or "<unknown>"
    base_commit = task.get("base_commit") or "<unknown>"
    version = task.get("version") or "unknown"
    problem_statement = task.get("problem_statement") or ""
    hints_text = (task.get("hints_text") or "").strip()
    hints_section = (
        f"## Hints (from the issue comments)\n\n{hints_text}\n\n"
        if hints_text
        else ""
    )
    return _PATCH_FOCUSED_TEMPLATE.format(
        repo=repo,
        version=version,
        base_commit=base_commit,
        problem_statement=problem_statement,
        hints_section=hints_section,
    )


def _build_prompt(task: dict[str, Any], profile: str) -> tuple[str, dict[str, Any]]:
    """Render the prompt for ``task`` under the given ``profile``.

    Returns ``(prompt_text, metadata)`` where metadata always carries:
    - ``prompt_profile``: the profile name actually used
    - ``prompt_sha256``: SHA-256 of the rendered prompt bytes (so the
      gate can attribute outcomes to specific prompt versions)
    - ``topology_override_used``: always False in this slice (we never
      force topology)
    - ``system_hint_forced``: always False in this slice
    """
    if profile == _PROMPT_PROFILE_PATCH_FOCUSED:
        text = _render_patch_focused_prompt(task)
    elif profile == _PROMPT_PROFILE_CANONICAL:
        text = render_swebench_prompt(normalize_swebench(task))
    else:
        raise ValueError(
            f"Unknown prompt profile {profile!r}; expected one of {_PROMPT_PROFILES}"
        )
    prompt_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    metadata = {
        "prompt_profile": profile,
        "prompt_sha256": prompt_sha256,
        "topology_override_used": False,
        "system_hint_forced": False,
    }
    return text, metadata

# ── Slice 8 / `canary-real-repo-context` (cgpro DESIGN 2026-05-11) ──
# After slice 7 still produced 0/5 patches, root cause shifted: the
# SWEBENCH_SYSTEM_TEMPLATE mandates ≥3 tool calls against the working
# directory before the agent emits a patch. The canary subprocess
# inherited the YGN repo root as CWD (no per-task checkout), so
# read_file / search_repo / list_files / run_tests all failed and the
# agent exhausted its step budget with the EMPTY_STEP_SENTINEL.
#
# Slice 8 clones each SWE-bench Pro instance's repo at base_commit into
# a per-task tempdir and launches the sage CLI subprocess with
# cwd=repo_dir so the tools see real source. Cleanup runs in
# try/finally per task, plus an atexit registry catches any leftovers
# from interrupted runs.

# Module-level registry of tempdirs the canary created. Populated by
# _setup_repo_for_canary, drained by _cleanup_repo_dir; anything still
# present at process exit is removed by _atexit_cleanup_canary_repos
# (registered once at import time below).
_CANARY_REPO_TMPDIRS: set[str] = set()

# Git CLI exit code budget. Shallow clone + checkout typically completes
# in 10–60 s on a healthy network; a single big repo (e.g. ansible)
# can push past 120 s. 180 s is a defensible upper bound for the
# initial clone; 60 s is enough for the fetch fallback (since we only
# fetch one specific commit at --depth 1).
_GIT_CLONE_TIMEOUT_S = 180.0
_GIT_FETCH_TIMEOUT_S = 120.0
_GIT_CHECKOUT_TIMEOUT_S = 60.0


def _atexit_cleanup_canary_repos() -> None:
    """Remove every tempdir still listed at process exit.

    Called once via ``atexit.register``. The try/finally inside
    ``_run_one_task`` removes its tempdir under normal control flow;
    this catches the interrupt case (SIGINT / unhandled exception
    above the finally block / asyncio cancellation that bubbled past
    cleanup).
    """
    for path in list(_CANARY_REPO_TMPDIRS):
        try:
            shutil.rmtree(path, ignore_errors=True)
        except Exception:  # noqa: BLE001 — best-effort, atexit handler
            pass
        _CANARY_REPO_TMPDIRS.discard(path)


import atexit  # noqa: E402  — registered immediately so any tempdir
                #                survives an interrupted main()
atexit.register(_atexit_cleanup_canary_repos)


def _setup_repo_for_canary(instance: dict[str, Any]) -> dict[str, Any]:
    """Clone ``instance.repo`` at ``instance.base_commit`` into a tempdir.

    Returns a dict with the metadata the canary needs to record per
    task:

    - ``repo_context_status`` — "ready" on success, otherwise a
      machine-readable failure tag.
    - ``repo_dir`` — the tempdir path (or ``None`` on failure).
    - ``repo_url`` — the GitHub URL the clone targeted.
    - ``base_commit`` — echoed back from the instance dict.
    - ``checkout_sha`` — what ``git rev-parse HEAD`` returns after the
      checkout step (should equal ``base_commit`` on success).
    - ``clone_elapsed_ms`` — wall-clock for the clone+checkout sequence.
    - ``fetch_fallback_used`` — bool. ``True`` when the shallow clone
      did not contain ``base_commit`` and we ran an explicit
      ``git fetch --depth 1 origin <base_commit>``.
    - ``failure_reason`` — populated on non-ready statuses.

    The caller must pass ``repo_dir`` as ``cwd`` to the subprocess that
    runs SAGE, and call ``_cleanup_repo_dir(repo_dir)`` in a finally.
    """
    repo = (instance.get("repo") or "").strip()
    base_commit = (instance.get("base_commit") or "").strip()
    metadata: dict[str, Any] = {
        "repo_context_status": "missing_inputs",
        "repo_dir": None,
        # ``tmp_root`` is the ``mkdtemp`` prefix root. Kept in the
        # metadata so the caller can always clean up even when the
        # ``repo_dir`` (a subdirectory of tmp_root) was never created
        # or got partially populated. ``None`` only when no tempdir
        # was created (missing-inputs early-return below).
        "tmp_root": None,
        "repo_url": None,
        "base_commit": base_commit or None,
        "checkout_sha": None,
        "clone_elapsed_ms": 0,
        "fetch_fallback_used": False,
        "failure_reason": None,
    }
    if not repo or not base_commit:
        metadata["failure_reason"] = (
            f"missing_inputs repo={repo!r} base_commit={base_commit!r}"
        )
        return metadata

    repo_url = f"https://github.com/{repo}.git"
    metadata["repo_url"] = repo_url

    tmp_root = tempfile.mkdtemp(prefix="sage_canary_repo_")
    _CANARY_REPO_TMPDIRS.add(tmp_root)
    metadata["tmp_root"] = tmp_root
    # Place the checkout inside the prefix dir so the dir's basename is
    # readable for diagnostics if cleanup ever skips.
    repo_dir = os.path.join(tmp_root, repo.split("/")[-1])

    start = time.monotonic()
    try:
        clone = subprocess.run(  # noqa: S603 — git is trusted; args validated above
            ["git", "clone", "--no-tags", "--depth", "1", repo_url, repo_dir],
            capture_output=True,
            timeout=_GIT_CLONE_TIMEOUT_S,
            check=False,
        )
        if clone.returncode != 0:
            metadata["repo_context_status"] = "clone_failed"
            stderr_tail = (clone.stderr or b"").decode(
                "utf-8", errors="replace"
            )[-2000:]
            metadata["failure_reason"] = f"git_clone_exit={clone.returncode} stderr={stderr_tail}"
            return metadata

        # Detached HEAD on the target commit. If the shallow clone does
        # not contain it (commit older than the tip), git checkout
        # exits non-zero and we run the fetch fallback.
        checkout = subprocess.run(  # noqa: S603
            ["git", "-C", repo_dir, "checkout", "--detach", base_commit],
            capture_output=True,
            timeout=_GIT_CHECKOUT_TIMEOUT_S,
            check=False,
        )
        if checkout.returncode != 0:
            metadata["fetch_fallback_used"] = True
            fetch = subprocess.run(  # noqa: S603
                ["git", "-C", repo_dir, "fetch", "--depth", "1", "origin", base_commit],
                capture_output=True,
                timeout=_GIT_FETCH_TIMEOUT_S,
                check=False,
            )
            if fetch.returncode != 0:
                metadata["repo_context_status"] = "fetch_failed"
                stderr_tail = (fetch.stderr or b"").decode(
                    "utf-8", errors="replace"
                )[-2000:]
                metadata["failure_reason"] = (
                    f"git_fetch_exit={fetch.returncode} stderr={stderr_tail}"
                )
                return metadata
            checkout2 = subprocess.run(  # noqa: S603
                ["git", "-C", repo_dir, "checkout", "--detach", base_commit],
                capture_output=True,
                timeout=_GIT_CHECKOUT_TIMEOUT_S,
                check=False,
            )
            if checkout2.returncode != 0:
                metadata["repo_context_status"] = "checkout_failed"
                stderr_tail = (checkout2.stderr or b"").decode(
                    "utf-8", errors="replace"
                )[-2000:]
                metadata["failure_reason"] = (
                    f"git_checkout_after_fetch_exit={checkout2.returncode} "
                    f"stderr={stderr_tail}"
                )
                return metadata

        # Verify what we actually checked out — drift here would be a
        # serious silent bug.
        head = subprocess.run(  # noqa: S603
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            capture_output=True,
            timeout=_GIT_CHECKOUT_TIMEOUT_S,
            check=False,
        )
        if head.returncode != 0:
            metadata["repo_context_status"] = "rev_parse_failed"
            metadata["failure_reason"] = (
                f"git_rev_parse_exit={head.returncode}"
            )
            return metadata
        metadata["checkout_sha"] = head.stdout.decode("utf-8").strip()

        metadata["repo_context_status"] = "ready"
        metadata["repo_dir"] = repo_dir
        return metadata
    except subprocess.TimeoutExpired as exc:
        metadata["repo_context_status"] = "timeout"
        metadata["failure_reason"] = f"timeout cmd={exc.cmd[:3]!r}"
        return metadata
    except (OSError, FileNotFoundError) as exc:
        metadata["repo_context_status"] = "os_error"
        metadata["failure_reason"] = f"os_error {exc!r}"
        return metadata
    finally:
        metadata["clone_elapsed_ms"] = int((time.monotonic() - start) * 1000)


def _load_ygn_dotenv_into(env: dict[str, str]) -> int:
    """Read ``<_REPO_ROOT>/.env`` and merge into ``env`` if absent.

    Returns the count of keys loaded. Keys already present in ``env``
    are NOT overwritten — the subprocess environment wins over the
    on-disk file. No-op if the .env file does not exist.

    Slice 8 follow-up (2026-05-11): with the canary now running each
    subprocess in cwd=cloned_repo, python-dotenv's auto-discovery
    (which walks UP from cwd) cannot find ``<YGN>/.env``. Pre-loading
    here preserves the historical behavior where the subprocess
    starts from the YGN repo root.
    """
    env_file = _REPO_ROOT / ".env"
    if not env_file.is_file():
        return 0
    n_loaded = 0
    try:
        text = env_file.read_text(encoding="utf-8")
    except OSError:
        return 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in env:
            continue
        # Strip surrounding quotes (single or double) if present.
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        env[key] = value
        n_loaded += 1
    return n_loaded


def _cleanup_repo_dir(repo_dir: str | None, *, tmp_root: str | None = None) -> str:
    """Remove the tempdir tree and drop it from the cleanup registry.

    Returns one of:
    - ``"removed"`` on success
    - ``"missing"`` when the path is ``None`` or does not exist
    - ``"failed"`` when ``shutil.rmtree`` raised even with ignore_errors

    The cleanup target is the tempdir prefix root (``tmp_root``), not
    just the ``<repo>`` subdir, so we get the whole ``sage_canary_repo_*``
    parent gone. When called with only ``repo_dir`` we infer the prefix
    by climbing one level (the canonical layout placed by
    ``_setup_repo_for_canary``).
    """
    candidate: str | None
    if tmp_root is not None:
        candidate = tmp_root
    elif repo_dir:
        candidate = os.path.dirname(repo_dir.rstrip(os.sep + "/"))
    else:
        candidate = None

    if not candidate or not os.path.exists(candidate):
        # Still drop from registry in case the tempdir was already
        # cleaned by atexit on a prior interrupt.
        if candidate is not None:
            _CANARY_REPO_TMPDIRS.discard(candidate)
        return "missing"

    try:
        shutil.rmtree(candidate, ignore_errors=True)
        _CANARY_REPO_TMPDIRS.discard(candidate)
        # ignore_errors=True can leave directories on Windows under
        # readonly bits (.git/objects/...). Best-effort second pass.
        if os.path.exists(candidate):
            return "failed"
        return "removed"
    except OSError:
        return "failed"

# Default model when running in real mode. Per cgpro plan §"Models per
# arm": cycle-13 main run forces SAGE_LLM_TIER=reasoner (Opus 4.7) for
# fair vs Claude Code. The smoke uses `budget` (deepseek-v4-flash) by
# default for cost — overridable via env or --tier.
_DEFAULT_TIER = "budget"

# Block `canary-stage-timing-budget` (cgpro DESIGN 2026-05-11, conv
# `cgpro_ygn_sage_global_analysis_20260510`) slice 3.
#
# Named timeout profiles. The B2 step 4 N=1 canary timed out at 300s on
# a substantial Vuls Trivy upgrade task without extracting a patch.
# cgpro recommended a 600-1200s envelope for the graded-patch-generation
# profile; 900s is the midpoint and gives the agent room without
# inviting unbounded reasoner_thinking_overflow.
#
# `default` preserves the historical 120s — the existing per-task budget
# for non-graded smokes (mock runs, plumbing checks, A5 timing triage
# unit work) — so unflagged callers are byte-equivalent to pre-slice-3.
_TIMEOUT_PROFILES: dict[str, float] = {
    "default": 120.0,
    "graded_patch_generation": 900.0,
}
_DEFAULT_PROFILE = "default"

# Path to the Pro patch format helper (loaded as module).
_FORMAT_PATCH_PATH = (
    Path(__file__).parent / "swebench_pro_format_patch.py"
).resolve()
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CANARY_MANIFEST_PATH = (
    _REPO_ROOT / "docs" / "benchmarks" / "cycle-13-canary-manifest.md"
)

_PREDICTION_AUDIT_SCHEMA_VERSION = "swebench_pro_canary_prediction_v1"
_PREDICTION_AUDIT_FIELDS = (
    "_verifier_repair_budget_usd",
    "_diff_verifier_mismatches",
    "_diff_verifier_outcome",
    "model_id_final",
    "provider_final",
    "_observed_model_ids",
    "_observed_providers",
    "_observed_event_cost_usd",
    "total_cost_usd",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _git_status_short() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "status", "--short"],
            cwd=_REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _extract_manifest_commit(manifest_text: str) -> str | None:
    match = re.search(
        r"\|\s*Commit SHA\s*\|\s*`?([^`|]+)`?\s*\|",
        manifest_text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).strip()


def _event_value(event: dict[str, Any], key: str) -> Any:
    payload = event.get("payload")
    if isinstance(payload, dict) and payload.get(key) is not None:
        return payload.get(key)
    return event.get(key)


def _event_audit_from_file(events_path: Path) -> dict[str, Any]:
    model_id_final: str | None = None
    provider_final: str | None = None
    observed_model_ids: set[str] = set()
    observed_providers: set[str] = set()
    assigned_providers: set[str] = set()
    execution_providers: set[str] = set()
    assigned_model_ids: set[str] = set()
    execution_model_ids: set[str] = set()
    provider_policy_failure_seen = False
    observed_cost_usd = 0.0

    if not events_path.is_file():
        return {
            "model_id_final": None,
            "provider_final": None,
            "_observed_model_ids": [],
            "_observed_providers": [],
            "_assigned_model_ids": [],
            "_assigned_providers": [],
            "_execution_model_ids": [],
            "_execution_providers": [],
            "_provider_policy_failure_seen": False,
            "_observed_event_cost_usd": 0.0,
        }

    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            ev_type = event.get("event_type")
            if ev_type == "failure":
                kind = _event_value(event, "kind")
                error_type = _event_value(event, "error_type")
                if kind == "provider_policy" or error_type == "provider_policy_violation":
                    provider_policy_failure_seen = True
            if ev_type in {
                "routing_decision",
                "model_assigned",
                "node_started",
                "node_completed",
            }:
                model_id = _event_value(event, "model_id")
                if isinstance(model_id, str) and model_id:
                    model_id_final = model_id
                    observed_model_ids.add(model_id)
                    if ev_type == "model_assigned":
                        assigned_model_ids.add(model_id)
                    if ev_type in {"node_started", "node_completed"}:
                        execution_model_ids.add(model_id)
                provider_id = _event_value(event, "provider_id") or _event_value(
                    event, "provider"
                )
                if isinstance(provider_id, str) and provider_id:
                    provider_final = provider_id
                    observed_providers.add(provider_id)
                    if ev_type == "model_assigned":
                        assigned_providers.add(provider_id)
                    if ev_type in {"node_started", "node_completed"}:
                        execution_providers.add(provider_id)
            cost_usd = _event_value(event, "cost_usd")
            if isinstance(cost_usd, (int, float)) and not isinstance(cost_usd, bool):
                observed_cost_usd += float(cost_usd)

    return {
        "model_id_final": model_id_final,
        "provider_final": provider_final,
        "_observed_model_ids": sorted(observed_model_ids),
        "_observed_providers": sorted(observed_providers),
        "_assigned_model_ids": sorted(assigned_model_ids),
        "_assigned_providers": sorted(assigned_providers),
        "_execution_model_ids": sorted(execution_model_ids),
        "_execution_providers": sorted(execution_providers),
        "_provider_policy_failure_seen": provider_policy_failure_seen,
        "_observed_event_cost_usd": observed_cost_usd,
    }


# ── Slice 10A topology / control-surface audit (cgpro VERIFY RF#C MODIFY) ──
# Captures the topology selection rationale + control-surface
# completeness markers so paired-reruns can attribute outcomes to
# prompt/profile changes vs bandit-Thompson noise. Does NOT alter
# any runtime contract — pure event-log post-processing.

_SENTINEL_MARKER_RUNTIME = "[sage: agent exited after"


def _topology_audit_from_file(events_path: Path) -> dict[str, Any]:
    """Extract topology + per-node + routing-vs-execution + oracle +
    control-surface metadata from a per-task events.jsonl.

    All fields are derived from events that already exist; this
    function does not add new event types or runtime behavior.
    """
    if not events_path.is_file():
        return {
            "topology_template": None,
            "topology_id": None,
            "node_count": None,
            "edge_count": None,
            "routing_model_id": None,
            "routing_confidence": None,
            "routing_source": None,
            "routing_system": None,
            "routing_domain": None,
            "nodes": [],
            "oracle": None,
            "control_surface": {
                "routing_decision_emitted": False,
                "topology_selected_emitted": False,
                "model_assigned_for_all_nodes": False,
                "oracle_verdict_emitted": False,
                "cli_complete_emitted": False,
            },
            "provider_policy_substitution_detected": False,
        }

    topology_template = None
    topology_id = None
    node_count = None
    edge_count = None
    routing_model_id = None
    routing_confidence = None
    routing_source = None
    routing_system = None
    routing_domain = None

    nodes_by_id: dict[str, dict[str, Any]] = {}

    oracle_payload: dict[str, Any] | None = None

    has_routing = False
    has_topology = False
    has_oracle = False
    has_cli_complete = False

    with events_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            ev_type = event.get("event_type")

            if ev_type == "routing_decision":
                has_routing = True
                routing_model_id = _event_value(event, "model_id")
                routing_confidence = _event_value(event, "confidence")
                routing_source = _event_value(event, "routing_source")
                routing_system = _event_value(event, "system")
                routing_domain = _event_value(event, "domain")

            elif ev_type == "topology_selected":
                has_topology = True
                topology_template = _event_value(event, "template_type")
                topology_id = _event_value(event, "topology_id")
                node_count = _event_value(event, "node_count")
                edge_count = _event_value(event, "edge_count")

            elif ev_type == "model_assigned":
                nid = str(_event_value(event, "node_id") or "")
                if not nid:
                    continue
                node = nodes_by_id.setdefault(nid, {"node_id": nid})
                node["assigned_role"] = _event_value(event, "node_role")
                node["assigned_model_id"] = _event_value(event, "model_id")
                node["assigned_provider_id"] = _event_value(event, "provider_id")

            elif ev_type == "node_completed":
                nid = str(_event_value(event, "node_id") or "")
                if not nid:
                    continue
                node = nodes_by_id.setdefault(nid, {"node_id": nid})
                node["completed_role"] = _event_value(event, "node_role")
                node["completed_model_id"] = _event_value(event, "model_id")
                node["completed_provider_id"] = _event_value(event, "provider_id")
                node["latency_ms"] = _event_value(event, "latency_ms")
                node["cost_usd"] = _event_value(event, "cost_usd")
                node["output_length"] = _event_value(event, "output_length")
                # Sentinel detection: the runtime emits payload as a string
                # in raw mode; if it matches the sentinel prefix, mark it.
                payload = event.get("payload")
                if isinstance(payload, str):
                    node["is_sentinel"] = _SENTINEL_MARKER_RUNTIME in payload
                else:
                    node["is_sentinel"] = (node.get("output_length") or 0) <= 51

            elif ev_type == "oracle_verdict":
                has_oracle = True
                payload = event.get("payload")
                if isinstance(payload, dict):
                    oracle_payload = {
                        "trainable": payload.get("trainable"),
                        "verdict_source": payload.get("verdict_source"),
                        "quality_label": payload.get("quality_label"),
                        "score": payload.get("score"),
                        "reason_codes": payload.get("reason_codes"),
                    }

            elif ev_type == "cli_complete":
                has_cli_complete = True

    nodes_sorted = sorted(nodes_by_id.values(), key=lambda n: int(n.get("node_id", "0") or "0"))

    # Provider-policy substitution: the routing layer picked one model,
    # but the actual execution used different model(s). Per the
    # slice 9 forensic finding, this happens silently — no dedicated
    # event. Slice 10A surfaces it as a derived flag pending the
    # full I-11 witness chain (10D).
    substitution_detected = False
    if routing_model_id:
        executed_models = {
            n.get("completed_model_id")
            for n in nodes_by_id.values()
            if isinstance(n.get("completed_model_id"), str)
        }
        if executed_models and routing_model_id not in executed_models:
            substitution_detected = True

    all_nodes_assigned = bool(
        node_count is not None
        and len(nodes_by_id) >= int(node_count)
        and all("assigned_model_id" in n for n in nodes_by_id.values())
    )

    return {
        "topology_template": topology_template,
        "topology_id": topology_id,
        "node_count": node_count,
        "edge_count": edge_count,
        "routing_model_id": routing_model_id,
        "routing_confidence": routing_confidence,
        "routing_source": routing_source,
        "routing_system": routing_system,
        "routing_domain": routing_domain,
        "nodes": nodes_sorted,
        "oracle": oracle_payload,
        "control_surface": {
            "routing_decision_emitted": has_routing,
            "topology_selected_emitted": has_topology,
            "model_assigned_for_all_nodes": all_nodes_assigned,
            "oracle_verdict_emitted": has_oracle,
            "cli_complete_emitted": has_cli_complete,
        },
        "provider_policy_substitution_detected": substitution_detected,
    }


# ── Synthetic patch generators (mock mode) ───────────────────────────────────


def _synthetic_empty_patch() -> str:
    """Empty patch: agent gave up. Pro grader treats as non-resolution.

    Per cgpro DESIGN E trap Q5 (validate_record accepts empty patch):
    this proves the runner produces shape-valid output even when the
    agent fails entirely.
    """
    return ""


def _synthetic_minimal_patch(instance_id: str) -> str:
    """Minimal but well-formed unified diff for shape validation.

    Per `gather_patches.py` in scaleapi/SWE-bench_Pro-os: the grader
    accepts plain-text patches. This synthetic patch is shape-valid
    but won't actually resolve any real test (designed to fail
    gracefully under the grader, not pass).
    """
    return (
        f"diff --git a/synthetic.py b/synthetic.py\n"
        f"index 0000000..1111111 100644\n"
        f"--- a/synthetic.py\n"
        f"+++ b/synthetic.py\n"
        f"@@ -1,1 +1,1 @@\n"
        f"-# placeholder for {instance_id}\n"
        f"+# patched by ygn-sage arm-d smoke (mock mode)\n"
    )


# ── Patch extraction from agent output ───────────────────────────────────────


_DIFF_HEADER_RE = re.compile(
    r"^diff --git a/.+ b/.+$",
    re.MULTILINE,
)


def _extract_patch_from_text(text: str) -> str:
    """Find the first unified-diff block in `text` and return it.

    Heuristic — matches a `diff --git ...` header and returns from
    that line to the end (or up to a fenced code block close if
    present). Empty-string returned when no diff found.

    This is the Tier 2.1 dumb-extractor. Cycle-13 main run can use
    a more sophisticated extractor (sage's existing diff parsing) if
    this proves insufficient.
    """
    if not text:
        return ""

    # Strip markdown code fences if the diff is wrapped.
    fence_match = re.search(
        r"```(?:diff|patch)?\n(.*?)```",
        text,
        re.DOTALL,
    )
    candidate = fence_match.group(1) if fence_match else text

    header_match = _DIFF_HEADER_RE.search(candidate)
    if not header_match:
        return ""

    return candidate[header_match.start():].strip() + "\n"


# ── Sage CLI subprocess (real mode) ──────────────────────────────────────────


# B2 bug 3 (2026-06-10): single source for the canary's diff-verifier mode.
# The subprocess env AND the launcher-side `_annotate_diff_verifier` call
# both read this constant, so the env the child sees can never drift from
# the mode the launcher annotates with (the 2026-05-12 N=5 had the env set
# but no launcher-side consumer at all — every `_diff_verifier_outcome` was
# None because the verifier lives in SWEBenchBench, which the canary never
# instantiates).
_CANARY_DIFF_VERIFIER_MODE = "observe"


def _task_subprocess_env(tier: str) -> dict[str, str]:
    """Build the per-task subprocess environment (extracted for testability —
    B2 contract test 6 asserts SAGE_DIFF_VERIFIER_MODE propagation)."""
    env = os.environ.copy()
    env["SAGE_LLM_TIER"] = tier
    env["SAGE_DIFF_VERIFIER_MODE"] = _CANARY_DIFF_VERIFIER_MODE
    env["SAGE_OTEL_EXPORTER"] = "none"
    return env


async def _run_sage_cli(
    task_text: str,
    budget_usd: float,
    output_events_path: Path,
    tier: str,
    provider_allowlist: tuple[str, ...] = (),
    provider_denylist: tuple[str, ...] = (),
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    """Invoke `python -m sage.cli run --jsonl` as subprocess.

    Uses the JSONL stdin protocol (matching the direct CLI path):
    writes ``{"command":"prompt","args":{"task":...}}`` to stdin,
    then drains stdout events + stderr in parallel (cgpro 2026-05-09
    fix — previously passed the task as a positional arg, which
    bypasses the canonical JSONL prompt channel and deadlocks on
    stderr buffering).
    """
    output_events_path.parent.mkdir(parents=True, exist_ok=True)

    env = _task_subprocess_env(tier)
    env.setdefault("SAGE_BOOT_BYPASS_EPOCH_GUARD", "1")
    env.setdefault("SAGE_BOOT_BYPASS_REASON",
        "cycle-13 E Tier 2.1 arm D smoke run; bypass disables atexit "
        "save so smokes do not pollute ~/.sage/ across consecutive runs")
    env.setdefault("SAGE_OPERATOR_ID", "ygn-sage-arm-d-smoke")
    # Slice 8 follow-up #2: SAGE CLI default writes redacted payloads
    # (``payload: None`` on stdout for runtime events). The canary
    # needs the final_result payload to extract a unified diff — under
    # the default, agent_output is always "" and the extractor always
    # returns empty. SAGE_TRACE_RAW=1 keeps credential redaction but
    # surfaces arbitrary text content so the canary can read the
    # agent's output. Per sage-python/src/sage/runtime/event_log/
    # writer.py:169 + redaction.py — credential-shaped patterns are
    # still stripped before serialization.
    env.setdefault("SAGE_TRACE_RAW", "1")
    # Slice 8 follow-up: when ``cwd`` is set to a cloned repo (not the
    # YGN root), python-dotenv auto-discovery inside sage cannot find
    # ``<YGN>/.env`` because it walks UP from cwd. Pre-load YGN's .env
    # into the subprocess env so API keys reach the sage CLI
    # regardless of cwd. Existing entries in os.environ win, so this
    # is a strict no-op when keys are already loaded.
    _load_ygn_dotenv_into(env)

    cmd = [
        sys.executable, "-m", "sage.cli", "run", "--jsonl",
        "--budget-usd", str(budget_usd),
    ]
    if provider_allowlist:
        cmd.extend(["--provider-allowlist", ",".join(provider_allowlist)])
    if provider_denylist:
        cmd.extend(["--provider-denylist", ",".join(provider_denylist)])
    log.info("Spawning sage CLI: %s", " ".join(cmd))

    start = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=str(cwd) if cwd is not None else None,
    )

    # Exercise the canonical inbound JSONL protocol, not the legacy
    # raw-stdin fallback.
    assert proc.stdin is not None
    prompt_command = {
        "command": "prompt",
        "args": {
            "task": task_text,
            "budget_usd": budget_usd,
        },
    }
    proc.stdin.write(
        (json.dumps(prompt_command, separators=(",", ":")) + "\n").encode("utf-8")
    )
    await proc.stdin.drain()
    proc.stdin.close()

    # Drain stderr in parallel to prevent pipe deadlock.
    async def _drain_stderr() -> bytes:
        if proc.stderr is None:
            return b""
        chunks: list[bytes] = []
        async for chunk in proc.stderr:
            chunks.append(chunk)
        return b"".join(chunks)

    stderr_task = asyncio.create_task(_drain_stderr())

    final_result_payload: Any = None
    cli_complete_payload: dict[str, Any] | None = None
    cli_complete_run_id: str | None = None
    model_id_final: str | None = None
    provider_final: str | None = None

    # Per SAGE CLI v0 protocol (docs/contracts/SAGE_CLI_PROTOCOL.md):
    # ``cli_complete`` is the terminal frame for a run. The subprocess
    # MAY hold stdout / its process slot open beyond that point (the
    # B2 step 2 / step 4 / slice-3 N=1 evidence on 2026-05-11 saw a
    # ~707s post-cli_complete idle wait before ``async for`` hit EOF
    # — the runner was blocked on the pipe, not doing useful work).
    # When we observe ``cli_complete`` we stop reading and terminate
    # the subprocess ourselves so the wall-clock budget bounds real
    # agent work, not subprocess teardown.
    saw_cli_complete = False
    exit_code: int | None = None
    latency_s: float = 0.0
    stderr_bytes: bytes = b""

    try:
        with output_events_path.open("w", encoding="utf-8", newline="\n") as event_log:
            assert proc.stdout is not None
            async for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    continue
                event_log.write(line + "\n")
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("Non-JSON line on stdout: %r", line[:80])
                    continue
                ev_type = event.get("event_type")
                payload = event.get("payload", {})
                if isinstance(payload, dict):
                    if ev_type == "routing_decision":
                        model_id_final = payload.get("model_id") or model_id_final
                    elif ev_type in {"model_assigned", "node_started", "node_completed"}:
                        model_id_final = payload.get("model_id") or model_id_final
                        provider_final = (
                            payload.get("provider_id")
                            or payload.get("provider")
                            or provider_final
                        )
                if ev_type == "final_result":
                    final_result_payload = payload
                elif ev_type == "cli_complete":
                    cli_complete_payload = payload if isinstance(payload, dict) else {}
                    event_run_id = event.get("run_id")
                    cli_complete_run_id = (
                        event_run_id if isinstance(event_run_id, str) else None
                    )
                    saw_cli_complete = True
                    break  # Terminal frame; do NOT wait for stdout EOF

        latency_s = time.monotonic() - start
        if saw_cli_complete and proc.returncode is None:
            # Subprocess may still be alive holding stdout; force closure
            # so wall-clock measures the contract (cli_complete), not
            # the runtime's teardown idle time.
            proc.terminate()
            try:
                exit_code = await asyncio.wait_for(proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                proc.kill()
                exit_code = await proc.wait()
        else:
            exit_code = await proc.wait()

        # Stderr drain: once the subprocess has exited (or been killed
        # above) its stderr pipe receives EOF and the drain task
        # finishes naturally. Wait briefly; otherwise treat stderr as
        # best-effort and move on.
        try:
            stderr_bytes = await asyncio.wait_for(stderr_task, timeout=5)
        except asyncio.TimeoutError:
            stderr_task.cancel()
            with suppress(asyncio.CancelledError):
                await stderr_task
    except asyncio.CancelledError:
        if proc.returncode is None:
            proc.terminate()
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=5)
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
        stderr_task.cancel()
        with suppress(asyncio.CancelledError):
            await stderr_task
        raise

    if cli_complete_payload is None:
        stderr_text = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
        log.error(
            "CLI ended without cli_complete; exit_code=%s stderr=%s",
            exit_code, stderr_text[-4000:],
        )

    event_audit = _event_audit_from_file(output_events_path)

    return {
        "exit_code": exit_code,
        "latency_ms": int(latency_s * 1000),
        "final_result_payload": final_result_payload,
        "cli_complete_payload": cli_complete_payload,
        "total_cost_usd": (
            (cli_complete_payload or {}).get("total_cost_usd")
            if cli_complete_payload
            else None
        ),
        "run_id": cli_complete_run_id,
        "model_id_final": event_audit.get("model_id_final") or model_id_final,
        "provider_final": event_audit.get("provider_final") or provider_final,
        "_observed_model_ids": event_audit.get("_observed_model_ids", []),
        "_observed_providers": event_audit.get("_observed_providers", []),
        "_assigned_model_ids": event_audit.get("_assigned_model_ids", []),
        "_assigned_providers": event_audit.get("_assigned_providers", []),
        "_execution_model_ids": event_audit.get("_execution_model_ids", []),
        "_execution_providers": event_audit.get("_execution_providers", []),
        "_provider_policy_failure_seen": event_audit.get(
            "_provider_policy_failure_seen",
            False,
        ),
        "_observed_event_cost_usd": event_audit.get("_observed_event_cost_usd", 0.0),
        "trace_dir": (
            cli_complete_payload.get("trace_dir")
            if isinstance(cli_complete_payload, dict)
            else None
        ),
        "stderr": stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else "",
    }


# ── Per-task runner ──────────────────────────────────────────────────────────


def _learning_evidence_not_requested() -> dict[str, Any]:
    return {
        "claimed": False,
        "status": "skipped",
        "reason_code": "not_claimed",
    }


def _safe_artifact_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return stem or "task"


def _learning_evidence_no_go(
    *,
    reason_code: str,
    detail: str,
    run_id: str | None,
    source_trace_dir: str | Path | None,
    archived_trace_dir: Path | None,
    expect_default_pipeline_learn: bool,
) -> dict[str, Any]:
    return {
        "claimed": True,
        "status": "no_go",
        "reason_code": reason_code,
        "detail": detail,
        "mode": "evidence-boundary",
        "expect_default_pipeline_learn": expect_default_pipeline_learn,
        "run_id": run_id,
        "source_trace_dir": str(source_trace_dir) if source_trace_dir else None,
        "trace_dir": str(archived_trace_dir) if archived_trace_dir else None,
        "records": 0,
    }


def _validate_learning_evidence(
    trace_dir: str | Path | None,
    run_id: str | None,
    *,
    archive_trace_dir: Path,
    expect_default_pipeline_learn: bool,
) -> dict[str, Any]:
    """Run the post-run learning side-effect evidence boundary.

    The task has already completed. This controls benchmark artifact
    acceptability only; it does not authorize or block runtime learning.
    """
    if not trace_dir or not run_id:
        return _learning_evidence_no_go(
            reason_code="missing_trace_identity",
            detail="cli_complete did not provide both run_id and trace_dir",
            run_id=run_id,
            source_trace_dir=trace_dir,
            archived_trace_dir=archive_trace_dir,
            expect_default_pipeline_learn=expect_default_pipeline_learn,
        )

    source_trace_dir = Path(trace_dir)
    if not source_trace_dir.is_dir():
        return _learning_evidence_no_go(
            reason_code="trace_dir_missing",
            detail=f"trace_dir not found: {source_trace_dir}",
            run_id=run_id,
            source_trace_dir=source_trace_dir,
            archived_trace_dir=archive_trace_dir,
            expect_default_pipeline_learn=expect_default_pipeline_learn,
        )

    try:
        if archive_trace_dir.exists():
            shutil.rmtree(archive_trace_dir)
        shutil.copytree(source_trace_dir, archive_trace_dir)
    except OSError as exc:
        return _learning_evidence_no_go(
            reason_code="trace_archive_failed",
            detail=f"{type(exc).__name__}: {exc}",
            run_id=run_id,
            source_trace_dir=source_trace_dir,
            archived_trace_dir=archive_trace_dir,
            expect_default_pipeline_learn=expect_default_pipeline_learn,
        )

    cmd = [
        sys.executable,
        "-m",
        "sage.runtime.credit_assignment.validate",
        str(archive_trace_dir),
        "--run-id",
        run_id,
        "--mode",
        "evidence-boundary",
    ]
    if expect_default_pipeline_learn:
        cmd.append("--expect-default-pipeline-learn")

    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _learning_evidence_no_go(
            reason_code="validator_error",
            detail=f"{type(exc).__name__}: {exc}",
            run_id=run_id,
            source_trace_dir=source_trace_dir,
            archived_trace_dir=archive_trace_dir,
            expect_default_pipeline_learn=expect_default_pipeline_learn,
        )

    status = "pass" if proc.returncode == 0 else "no_go"
    return {
        "claimed": True,
        "status": status,
        "reason_code": "validated" if status == "pass" else "validator_failed",
        "mode": "evidence-boundary",
        "expect_default_pipeline_learn": expect_default_pipeline_learn,
        "run_id": run_id,
        "source_trace_dir": str(source_trace_dir),
        "trace_dir": str(archive_trace_dir),
        "validator_exit_code": proc.returncode,
        "validator_command": cmd,
        "validator_stdout": proc.stdout[-4000:],
        "validator_stderr": proc.stderr[-4000:],
    }


def _load_format_patch_module() -> Any:
    """Load `swebench_pro_format_patch` from scripts/ as a sibling module."""
    spec = importlib.util.spec_from_file_location(
        "swebench_pro_format_patch", _FORMAT_PATCH_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("swebench_pro_format_patch", mod)
    spec.loader.exec_module(mod)
    return mod


def _prediction_audit_record(
    record: dict[str, Any],
    summary: dict[str, Any],
    *,
    task_index: int,
) -> dict[str, Any]:
    audit_record = dict(record)
    audit_record["_audit_schema_version"] = _PREDICTION_AUDIT_SCHEMA_VERSION
    audit_record["_task_index"] = task_index
    audit_record["_mock"] = bool(summary.get("mock"))
    audit_record["_timeout"] = bool(summary.get("timeout"))
    audit_record["_exit_code"] = summary.get("exit_code")
    audit_record["_latency_ms"] = summary.get("latency_ms")
    audit_record["_events_path"] = summary.get("events_path")
    for field in _PREDICTION_AUDIT_FIELDS:
        audit_record[field] = summary.get(field)
    return audit_record


def _write_predictions_jsonl(
    records: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, (record, summary) in enumerate(zip(records, summaries, strict=True)):
            handle.write(
                json.dumps(
                    _prediction_audit_record(record, summary, task_index=index),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def _write_aggregate_events(
    summaries: list[dict[str, Any]],
    *,
    output_dir: Path,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as destination:
        for summary in summaries:
            rel_path = summary.get("events_path")
            if not isinstance(rel_path, str):
                continue
            source = output_dir / rel_path
            if not source.is_file():
                continue
            with source.open("r", encoding="utf-8") as handle:
                for line in handle:
                    destination.write(line.rstrip("\r\n") + "\n")


def _load_grader_gate(grader_preflight_path: Path | None) -> dict[str, Any]:
    if grader_preflight_path is None:
        return {
            "status": "BLOCKED",
            "reason": "grader_preflight_artifact_not_supplied",
            "path": None,
        }
    if not grader_preflight_path.is_file():
        return {
            "status": "BLOCKED",
            "reason": "grader_preflight_artifact_missing",
            "path": str(grader_preflight_path),
        }
    try:
        data = json.loads(grader_preflight_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "BLOCKED",
            "reason": "grader_preflight_artifact_unreadable",
            "path": str(grader_preflight_path),
            "detail": f"{type(exc).__name__}: {exc}",
        }
    decision = data.get("decision")
    # The preflight script (`scripts/swebench_pro_grader_preflight.py`)
    # emits decision string `READY_MODAL` for the remote-Modal grading
    # path (introduced 2026-05-10 in commit a7474306), but this gate was
    # added earlier in commit ec0b775e with the speculatively-typed
    # string `READY_REMOTE_MODAL`. Result: every Modal preflight ever
    # produced (e.g. `docs/benchmarks/2026-05-12-b2-n5-graded/
    # grader_preflight.json`) was rejected by the gate as
    # `grader_preflight_not_ready`. Accept both forms going forward,
    # and also accept the explicit boolean fields the preflight always
    # emits — that is the source of truth, the decision string is just
    # a human-readable summary.
    ready = (
        bool(data.get("local_grading_ready"))
        or bool(data.get("modal_grading_ready"))
        or decision in {"READY_LOCAL_DOCKER", "READY_MODAL", "READY_REMOTE_MODAL"}
    )
    return {
        "status": "PASS" if ready else "BLOCKED",
        "reason": None if ready else "grader_preflight_not_ready",
        "path": str(grader_preflight_path),
        "sha256": _sha256_file(grader_preflight_path),
        "decision": decision,
        "blockers": data.get("blockers", []),
    }


def _load_ci_gate(ci_green_artifact: Path | None, *, git_head: str | None) -> dict[str, Any]:
    if ci_green_artifact is None:
        return {
            "status": "BLOCKED",
            "reason": "ci_green_artifact_not_supplied",
            "path": None,
        }
    if not ci_green_artifact.is_file():
        return {
            "status": "BLOCKED",
            "reason": "ci_green_artifact_missing",
            "path": str(ci_green_artifact),
        }
    try:
        data = json.loads(ci_green_artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "BLOCKED",
            "reason": "ci_green_artifact_unreadable",
            "path": str(ci_green_artifact),
            "detail": f"{type(exc).__name__}: {exc}",
        }
    status = str(data.get("status") or data.get("conclusion") or "").lower()
    commit = data.get("commit") or data.get("head_sha")
    passed = status in {"pass", "passed", "success", "green"} and (
        git_head is None or commit == git_head
    )
    reasons: list[str] = []
    if status not in {"pass", "passed", "success", "green"}:
        reasons.append("ci_status_not_green")
    if git_head is not None and commit != git_head:
        reasons.append("ci_commit_mismatch")
    return {
        "status": "PASS" if passed else "BLOCKED",
        "reason": None if passed else ",".join(reasons),
        "path": str(ci_green_artifact),
        "sha256": _sha256_file(ci_green_artifact),
        "commit": commit,
        "reported_status": status,
    }


def _write_launch_manifest(
    *,
    output_dir: Path,
    instances_json: Path,
    manifest_path: Path | None,
    budget_usd: float,
    global_budget_usd: float,
    task_timeout_s: float,
    profile: str = _DEFAULT_PROFILE,
    profile_timeout_override: bool = False,
    provider_allowlist: tuple[str, ...],
    provider_denylist: tuple[str, ...],
    grader_preflight_path: Path | None,
    ci_green_artifact: Path | None,
) -> dict[str, Any]:
    git_head = _git_head()
    git_status_short = _git_status_short()
    manifest_exists = manifest_path is not None and manifest_path.is_file()
    manifest_text = (
        manifest_path.read_text(encoding="utf-8") if manifest_exists and manifest_path else ""
    )
    manifest_commit = _extract_manifest_commit(manifest_text) if manifest_text else None
    manifest_reasons: list[str] = []
    if not manifest_exists:
        manifest_reasons.append("manifest_missing")
    elif manifest_commit in {None, "", "<SET_AT_LAUNCH>"}:
        manifest_reasons.append("manifest_commit_not_frozen")
    elif git_head is not None and manifest_commit != git_head:
        manifest_reasons.append("manifest_commit_mismatch")

    if manifest_exists and manifest_path is not None:
        shutil.copyfile(manifest_path, output_dir / "launch_manifest.md")

    launch_manifest = {
        "schema_version": "swebench_pro_canary_launch_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo": {
            "path": str(_REPO_ROOT),
            "head": git_head,
            "dirty": bool(git_status_short),
            "status_short": git_status_short,
        },
        "inputs": {
            "instances_json": {
                "path": str(instances_json),
                "sha256": _sha256_file(instances_json),
            },
            "manifest": {
                "path": str(manifest_path) if manifest_path else None,
                "exists": bool(manifest_exists),
                "sha256": _sha256_file(manifest_path) if manifest_exists and manifest_path else None,
                "declared_commit": manifest_commit,
                "copied_to": "launch_manifest.md" if manifest_exists else None,
            },
        },
        "budget": {
            "budget_usd_per_task": budget_usd,
            "global_budget_usd": global_budget_usd,
            "task_timeout_s": task_timeout_s,
            "effective_profile": profile,
            "profile_timeout_default_s": _TIMEOUT_PROFILES.get(profile),
            "profile_timeout_override": profile_timeout_override,
        },
        "providers": {
            "allowlist": list(provider_allowlist),
            "denylist": list(provider_denylist),
        },
        "manifest_gate": {
            "status": "PASS" if not manifest_reasons else "BLOCKED",
            "reasons": manifest_reasons,
        },
        "grading_gate": _load_grader_gate(grader_preflight_path),
        "ci_gate": _load_ci_gate(ci_green_artifact, git_head=git_head),
    }
    (output_dir / "launch_manifest.json").write_text(
        json.dumps(launch_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    return launch_manifest


def _combine_canary_decision(gates: dict[str, dict[str, Any]]) -> str:
    statuses = {name: gate.get("status") for name, gate in gates.items()}
    if any(status == "NO_GO" for status in statuses.values()):
        return "NO_GO"
    if any(status == "BLOCKED" for status in statuses.values()):
        return "BLOCKED"
    if all(status == "PASS" for status in statuses.values()):
        return "PENDING_REVIEW"
    return "BLOCKED"


def _provider_gate(
    summaries: list[dict[str, Any]],
    *,
    mock: bool,
    provider_allowlist: tuple[str, ...],
    provider_denylist: tuple[str, ...],
) -> dict[str, Any]:
    if mock:
        return {
            "status": "BLOCKED",
            "reason": "mock_mode_no_provider_observation",
            "observed_providers": [],
            "provider_allowlist": list(provider_allowlist),
            "provider_denylist": list(provider_denylist),
        }
    observed_set: set[str] = set()
    assigned_set: set[str] = set()
    execution_set: set[str] = set()
    policy_failure_seen = False
    for summary in summaries:
        raw_observed = summary.get("_observed_providers")
        if isinstance(raw_observed, list):
            observed_set.update(str(item) for item in raw_observed if item)
        elif isinstance(raw_observed, tuple):
            observed_set.update(str(item) for item in raw_observed if item)
        raw_assigned = summary.get("_assigned_providers")
        if isinstance(raw_assigned, (list, tuple)):
            assigned_set.update(str(item) for item in raw_assigned if item)
        raw_execution = summary.get("_execution_providers")
        if isinstance(raw_execution, (list, tuple)):
            execution_set.update(str(item) for item in raw_execution if item)
        policy_failure_seen = policy_failure_seen or bool(
            summary.get("_provider_policy_failure_seen")
        )
        provider_final = summary.get("provider_final")
        if provider_final:
            observed_set.add(str(provider_final))
    observed = sorted(observed_set)
    assigned = sorted(assigned_set)
    execution = sorted(execution_set)
    missing = [
        str(summary.get("instance_id"))
        for summary in summaries
        if (
            not summary.get("_provider_policy_failure_seen")
            and (not summary.get("provider_final") or not summary.get("model_id_final"))
        )
    ]
    denyset = set(provider_denylist)
    allowset = set(provider_allowlist)
    assigned_denied = [provider for provider in assigned if provider in denyset]
    execution_denied = [provider for provider in execution if provider in denyset]
    assigned_outside_allowlist = [
        provider for provider in assigned if allowset and provider not in allowset
    ]
    execution_outside_allowlist = [
        provider for provider in execution if allowset and provider not in allowset
    ]
    assigned_policy_violation = bool(assigned_denied or assigned_outside_allowlist)
    execution_policy_violation = bool(execution_denied or execution_outside_allowlist)
    if execution_policy_violation or missing:
        status = "NO_GO"
    elif assigned_policy_violation and not policy_failure_seen:
        status = "NO_GO"
    else:
        status = "PASS"
    reason = None
    if status == "NO_GO":
        reason = "provider_audit_failed"
    elif assigned_policy_violation and policy_failure_seen:
        reason = "runtime_provider_policy_enforced"
    return {
        "status": status,
        "reason": reason,
        "observed_providers": observed,
        "assigned_providers": assigned,
        "execution_providers": execution,
        "provider_policy_failure_seen": policy_failure_seen,
        "missing_provider_or_model": missing,
        "assigned_denied_providers": assigned_denied,
        "execution_denied_providers": execution_denied,
        "assigned_outside_allowlist": assigned_outside_allowlist,
        "execution_outside_allowlist": execution_outside_allowlist,
        "provider_allowlist": list(provider_allowlist),
        "provider_denylist": list(provider_denylist),
    }


def _categorize_timeout_from_events(
    events_path: Path,
    *,
    task_timeout_s: float,
) -> dict[str, Any]:
    """Block A5 (cgpro DESIGN 2026-05-10): categorize a runner timeout.

    Reads per-task events file and forwards typed event lists to
    `sage.bench.event_ledger.categorize_timeout`. Returns the
    categorization dict (last_stage / elapsed_ms_by_stage /
    provider_attempted / model_id_final / provider_final / reason_code)
    so the runner summary can carry it as a first-class field.

    Per cgpro DESIGN correction: heuristic is TIME-based (uses
    cli_progress.payload.elapsed_ms which is cumulative since task
    start), not count-based; provider_attempted is true only when
    node_started events exist (model_assigned alone is not proof of a
    call attempt).
    """
    progress_events: list[dict[str, Any]] = []
    model_assigned_events: list[dict[str, Any]] = []
    node_started_events: list[dict[str, Any]] = []
    routing_decision_events: list[dict[str, Any]] = []

    if events_path.is_file():
        for raw in events_path.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                ev = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            event_type = ev.get("event_type")
            if event_type == "cli_progress":
                progress_events.append(ev)
            elif event_type == "model_assigned":
                model_assigned_events.append(ev)
            elif event_type == "node_started":
                node_started_events.append(ev)
            elif event_type == "routing_decision":
                routing_decision_events.append(ev)

    from sage.bench.event_ledger import categorize_timeout

    return categorize_timeout(
        progress_events=progress_events,
        model_assigned_events=model_assigned_events,
        node_started_events=node_started_events,
        routing_decision_events=routing_decision_events,
        elapsed_total_ms=task_timeout_s * 1000.0,
    )


def _resolve_total_cost(
    *,
    cli_total_cost_usd: float | None,
    observed_event_cost_usd: float,
    had_llm_execution: bool,
    cli_complete_expected: bool,
) -> tuple[float, str, dict[str, Any] | None]:
    """Resolve the authoritative per-task cost (B2 bug 2, 2026-05-12 canary).

    The 2026-05-12 N=5 exposed that `cli_complete.total_cost_usd` is lossy
    outside the happy path: hard failures report $0 while the JSONL trace
    carries real `node_completed.cost_usd`, and even successes under-report
    (tutanota db90ac26: $0.134 reported vs $0.266 observed). The eeb3a7fb fix
    recovered cost on the TIMEOUT path only; this helper generalizes the same
    rule to all three paths (success / failure / timeout).

    Contract (cgpro stop-condition: never double-count): the two sources are
    never summed — the larger one wins, with an explicit source label:
      - "cli_complete"                          cli payload covers observed
      - "event_audit_observed_event_cost_usd"   trace sum exceeds/replaces it
      - "no_cost_evidence"                      neither source has a figure
    A cost_integrity_warning is emitted when the audit had to override or
    when LLM execution evidence exists with zero recorded cost.
    `cli_complete_expected=False` (timeout path) suppresses the
    missing-payload warning: the subprocess was killed, absence is normal.
    """
    cli_total = (
        float(cli_total_cost_usd)
        if isinstance(cli_total_cost_usd, (int, float))
        and not isinstance(cli_total_cost_usd, bool)
        else None
    )
    observed = float(observed_event_cost_usd or 0.0)
    epsilon = 1e-9

    if cli_total is not None and cli_total + epsilon >= observed:
        return cli_total, "cli_complete", None

    if observed > 0.0:
        if cli_total is None:
            warning = (
                {
                    "reason_code": "cli_complete_cost_missing",
                    "detail": (
                        "cli_complete carried no total_cost_usd but the JSONL "
                        "trace recorded real per-node cost — recovered from "
                        "the event audit. Budget reports must use this "
                        "figure, not $0.00."
                    ),
                }
                if cli_complete_expected
                else None
            )
        else:
            warning = {
                "reason_code": "cli_complete_cost_underreport",
                "detail": (
                    f"cli_complete reported total_cost_usd={cli_total:.6f} "
                    f"but the event audit observed {observed:.6f} — using "
                    "the larger event-derived figure (2.6x under-report "
                    "class caught by cgpro on the 2026-05-12 N=5)."
                ),
                "cli_complete_total_cost_usd": cli_total,
                "observed_event_cost_usd": observed,
            }
        return observed, "event_audit_observed_event_cost_usd", warning

    warning = None
    if had_llm_execution:
        warning = {
            "reason_code": "llm_execution_observed_zero_cost",
            "detail": (
                "node_completed / model_assigned events observed in "
                "trace but no cost_usd > 0 was recorded — provider "
                "cost accounting may have been lost. "
                "Do NOT report this as a clean $0.00 spend in any "
                "downstream budget audit."
            ),
        }
    if cli_total is not None:
        return cli_total, "cli_complete", warning
    return 0.0, "no_cost_evidence", warning


def _annotate_diff_verifier(
    *,
    patch: str,
    repo_dir: str | None,
    mode: str,
) -> dict[str, Any]:
    """Run the pre-emission diff-context verifier on an extracted patch
    (B2 bug 3, 2026-05-12 canary): the env flag was correctly propagated to
    the subprocess but NOBODY consumed it on the canary path — the verifier
    lives in SWEBenchBench._run_one_instance, which the canary never
    instantiates. This launcher-side annotation closes that gap.

    Every skip path is an EXPLICIT outcome string (never None): the canary
    manifest stop condition #5 keys off "zero tasks produce
    `_diff_verifier_outcome`", which null fields would re-trigger silently.
    Serialization mirrors swebench_bench (compact mismatch dicts, no
    expected/actual bodies; crash maps to unsupported_no_opinion).
    """
    if mode not in {"observe", "repair"}:
        return {
            "_diff_verifier_outcome": "skipped_mode_off",
            "_diff_verifier_mismatches": None,
            "_diff_verifier_reasons": None,
        }
    if not patch:
        return {
            "_diff_verifier_outcome": "skipped_no_patch",
            "_diff_verifier_mismatches": None,
            "_diff_verifier_reasons": None,
        }
    if not repo_dir or not Path(repo_dir).is_dir():
        return {
            "_diff_verifier_outcome": "skipped_no_repo_dir",
            "_diff_verifier_mismatches": None,
            "_diff_verifier_reasons": None,
        }
    from sage.bench.swebench_diff_verifier import verify_diff_context_with_reasons

    try:
        result = verify_diff_context_with_reasons(patch, Path(repo_dir))
    except Exception as exc:  # noqa: BLE001 - observability annotation must not kill the task
        log.warning("diff verifier crashed during canary annotation: %s", exc)
        return {
            "_diff_verifier_outcome": "unsupported_no_opinion",
            "_diff_verifier_mismatches": [],
            "_diff_verifier_reasons": ["unsupported_no_opinion"],
        }
    return {
        "_diff_verifier_outcome": str(result.outcome),
        "_diff_verifier_mismatches": [
            {
                "file": m.file,
                "hunk_index": m.hunk_index,
                "old_start": m.old_start,
                "old_count": m.old_count,
                "kind": m.kind,
                "match_ratio": m.match_ratio,
            }
            for m in result.mismatches
        ],
        "_diff_verifier_reasons": [str(reason) for reason in result.reasons],
    }


def _timeout_task_result(
    task: dict[str, Any],
    output_dir: Path,
    *,
    prefix: str,
    fmt_module: Any,
    task_timeout_s: float,
    expect_default_pipeline_learn: bool,
) -> dict[str, Any]:
    instance_id = task["instance_id"]
    per_task_events = output_dir / "per_task" / f"{instance_id}.events.jsonl"
    per_task_events.parent.mkdir(parents=True, exist_ok=True)
    needs_leading_newline = False
    if per_task_events.is_file() and per_task_events.stat().st_size > 0:
        with per_task_events.open("rb") as existing:
            existing.seek(-1, os.SEEK_END)
            needs_leading_newline = existing.read(1) not in {b"\n", b"\r"}
    timeout_line = json.dumps(
            {
                "event_type": "runner_timeout",
                "instance_id": instance_id,
                "task_timeout_s": task_timeout_s,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    with per_task_events.open("a", encoding="utf-8", newline="\n") as handle:
        if needs_leading_newline:
            handle.write("\n")
        handle.write(timeout_line + "\n")
    event_audit = _event_audit_from_file(per_task_events)
    # Block A5 categorization — read AFTER the runner_timeout line is
    # appended so the events file is in its final shape.
    timeout_categorization = _categorize_timeout_from_events(
        per_task_events,
        task_timeout_s=task_timeout_s,
    )
    # cgpro NEXT_BLOCK_ID=COST_TRACKING_ZERO_COST_RUNNER_TIMEOUT_FIX
    # (DESIGN_LOCKED 2026-05-12 real-canary VERIFY): on the runner-
    # timeout path, the subprocess was killed before `cli_complete`
    # could report `total_cost_usd`. But the JSONL trace DOES
    # contain `node_completed.cost_usd` for any LLM calls that
    # finished before the timeout. Use the event-derived sum
    # (`_observed_event_cost_usd`) as the authoritative
    # `total_cost_usd` on the timeout path, otherwise the summary
    # reports a false $0.00 spend while real API budget was
    # consumed. cgpro acceptance criterion #5: emit an explicit
    # cost_integrity_warning when LLM execution evidence exists
    # but cost would have been reported as zero.
    # B2 bug 2 (2026-06-10): the inline timeout-only recovery (eeb3a7fb) is
    # now the shared _resolve_total_cost helper, used by all three result
    # paths. cli_complete_expected=False — the subprocess was killed, so a
    # missing payload is normal here, not an integrity anomaly.
    timeout_total_cost, timeout_cost_source, timeout_cost_integrity_warning = (
        _resolve_total_cost(
            cli_total_cost_usd=None,
            observed_event_cost_usd=float(
                event_audit.get("_observed_event_cost_usd", 0.0) or 0.0
            ),
            had_llm_execution=bool(
                event_audit.get("_execution_model_ids")
                or event_audit.get("_observed_model_ids")
            ),
            cli_complete_expected=False,
        )
    )
    summary = {
        "instance_id": instance_id,
        "exit_code": None,
        "latency_ms": int(task_timeout_s * 1000),
        "total_cost_usd": timeout_total_cost,
        "_total_cost_usd_source": timeout_cost_source,
        "cost_integrity_warning": timeout_cost_integrity_warning,
        "extracted_patch_present": False,
        "extracted_patch_chars": 0,
        "mock": False,
        "timeout": True,
        "events_path": str(per_task_events.relative_to(output_dir)),
        "_verifier_repair_budget_usd": None,
        "_diff_verifier_mismatches": None,
        # B2 bug 3: explicit skip outcome on timeout — the subprocess was
        # killed before any patch could exist; null would silently
        # re-trigger manifest stop condition #5.
        "_diff_verifier_outcome": "skipped_timeout",
        "_diff_verifier_reasons": None,
        "model_id_final": (
            timeout_categorization.get("model_id_final")
            or event_audit.get("model_id_final")
        ),
        "provider_final": (
            timeout_categorization.get("provider_final")
            or event_audit.get("provider_final")
        ),
        "_observed_model_ids": event_audit.get("_observed_model_ids", []),
        "_observed_providers": event_audit.get("_observed_providers", []),
        "_assigned_model_ids": event_audit.get("_assigned_model_ids", []),
        "_assigned_providers": event_audit.get("_assigned_providers", []),
        "_execution_model_ids": event_audit.get("_execution_model_ids", []),
        "_execution_providers": event_audit.get("_execution_providers", []),
        "_provider_policy_failure_seen": event_audit.get(
            "_provider_policy_failure_seen",
            False,
        ),
        "_observed_event_cost_usd": event_audit.get("_observed_event_cost_usd", 0.0),
        # Block A5: timeout categorization (cgpro DESIGN 2026-05-10).
        # Distinguishes scoring_boot_impossible / reasoner_thinking_overflow /
        # provider_call_timeout / stage_deadlock so a timeout reports
        # something actionable rather than just "120s exceeded".
        "timeout_categorization": {
            "last_stage": timeout_categorization["last_stage"],
            "elapsed_ms_by_stage": timeout_categorization["elapsed_ms_by_stage"],
            "provider_attempted": timeout_categorization["provider_attempted"],
            "reason_code": timeout_categorization["reason_code"],
        },
        "learning_evidence_boundary": _learning_evidence_no_go(
            reason_code="task_timeout",
            detail=f"task exceeded timeout_s={task_timeout_s}",
            run_id=None,
            source_trace_dir=None,
            archived_trace_dir=None,
            expect_default_pipeline_learn=expect_default_pipeline_learn,
        ),
        # Slice 10A: topology audit also stamped on the timeout path so
        # paired-rerun analysis can compare timeout-vs-success runs at
        # the same control-surface granularity.
        "topology_audit": _topology_audit_from_file(per_task_events),
    }
    return {
        "summary": summary,
        "record": fmt_module.format_patch(instance_id, "", prefix=prefix),
    }


async def _run_one_task(
    task: dict[str, Any],
    output_dir: Path,
    *,
    mock: bool,
    budget_usd: float,
    tier: str,
    provider_allowlist: tuple[str, ...],
    provider_denylist: tuple[str, ...],
    prefix: str,
    fmt_module: Any,
    claim_default_pipeline_learning_evidence: bool,
    expect_default_pipeline_learn: bool,
    swebench_prompt_profile: str = _DEFAULT_PROMPT_PROFILE,
) -> dict[str, Any]:
    """Run one task end-to-end. Returns a per-task summary dict."""
    instance_id = task["instance_id"]
    log.info("Running task %s (mock=%s)", instance_id, mock)

    per_task_events = output_dir / "per_task" / f"{instance_id}.events.jsonl"

    if mock:
        # No subprocess — synthesize a patch + minimal "summary".
        patch = _synthetic_minimal_patch(instance_id)
        # B2 (2026-06-10): mock summaries carry the same explicit audit
        # fields as real ones so a --mock dry-run can prove the three
        # B2_RERUN_UNBLOCKERS field classes are populated end-to-end.
        # No repo worktree exists in mock, so the annotation resolves to
        # the explicit "skipped_no_repo_dir" outcome (never None).
        mock_verifier_annotation = _annotate_diff_verifier(
            patch=patch,
            repo_dir=None,
            mode=_CANARY_DIFF_VERIFIER_MODE,
        )
        summary = {
            "instance_id": instance_id,
            "exit_code": 0,
            "latency_ms": 0,
            "total_cost_usd": 0.0,
            "_total_cost_usd_source": "no_cost_evidence",
            "cost_integrity_warning": None,
            "extracted_patch_present": bool(patch),
            "extracted_patch_chars": len(patch),
            "mock": True,
            "timeout": False,
            "events_path": str(per_task_events.relative_to(output_dir)),
            "_diff_verifier_mismatches": mock_verifier_annotation[
                "_diff_verifier_mismatches"
            ],
            "_diff_verifier_outcome": mock_verifier_annotation[
                "_diff_verifier_outcome"
            ],
            "_diff_verifier_reasons": mock_verifier_annotation[
                "_diff_verifier_reasons"
            ],
            "learning_evidence_boundary": _learning_evidence_not_requested(),
        }
        # Write a dummy events file so per_task dir is uniform.
        per_task_events.parent.mkdir(parents=True, exist_ok=True)
        per_task_events.write_text(
            json.dumps(
                {
                    "event_type": "synthetic_mock",
                    "instance_id": instance_id,
                    "note": "mock mode — no real CLI invocation",
                }
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
    else:
        # Slice 8 / `canary-real-repo-context` (cgpro DESIGN 2026-05-11):
        # clone the instance's repo at base_commit and run the sage CLI
        # subprocess with cwd=repo_dir so read_file / search_repo /
        # list_files / run_tests resolve against the real source tree.
        # Without this, the SWEBENCH_SYSTEM_TEMPLATE Mandatory Workflow
        # (≥3 tool calls before emitting a patch) forces the agent to
        # call tools that fail → step budget exhausted → EMPTY_STEP_SENTINEL.
        repo_context = _setup_repo_for_canary(task)
        repo_dir = repo_context.get("repo_dir")
        # The tempdir prefix to clean up after the run is the
        # ``mkdtemp`` root, kept in ``repo_context["tmp_root"]`` even
        # when the clone/checkout failed and ``repo_dir`` is None.
        # Fall back to deriving from repo_dir if tmp_root is absent
        # (e.g. test stubs that don't simulate the full metadata).
        tmp_root_to_clean = repo_context.get("tmp_root")
        if tmp_root_to_clean is None and isinstance(repo_dir, str):
            tmp_root_to_clean = os.path.dirname(repo_dir)

        if repo_context.get("repo_context_status") != "ready":
            log.warning(
                "[%s] Repo context NOT ready (%s); subprocess will run "
                "from YGN repo root and tools will likely fail. "
                "Continuing so the failure is observable in events.",
                instance_id,
                repo_context.get("repo_context_status"),
            )

        # Slice 9: prompt profile dispatch.
        # - canonical: render_swebench_prompt(normalize_swebench(task))
        #   from sage.input.swebench (byte-identical to slice 7+8)
        # - patch_focused: canary-local template, drops the "≥3 tool
        #   calls" mandate while keeping repo-grounding + strict diff
        prompt, prompt_metadata = _build_prompt(task, swebench_prompt_profile)
        cleanup_status = "not_attempted"
        try:
            cli_result = await _run_sage_cli(
                prompt,
                budget_usd=budget_usd,
                output_events_path=per_task_events,
                tier=tier,
                provider_allowlist=provider_allowlist,
                provider_denylist=provider_denylist,
                cwd=repo_dir if isinstance(repo_dir, str) else None,
            )
            # Extract patch from final_result. B2 bug 3 (2026-06-10): the
            # extraction AND the diff-context verification moved INSIDE this
            # try-block — the verifier compares hunk context against real
            # file bytes, so it must run before the finally below removes
            # the cloned worktree (previously extraction ran after cleanup
            # and the verifier had nothing to read even if wired).
            agent_output = ""
            payload = cli_result.get("final_result_payload") or {}
            if isinstance(payload, str):
                agent_output = payload
            elif isinstance(payload, dict):
                for key in ("result", "output", "text", "content", "answer"):
                    val = payload.get(key)
                    if isinstance(val, str) and val.strip():
                        agent_output = val
                        break

            # Use the swebench_bench extractor (handles raw diffs, fenced
            # ```diff blocks, mixed text-with-embedded-diff, sentinel
            # rejection, Unix line-ending normalization). The local
            # ``_extract_patch_from_text`` is kept for older callers that
            # only need the dumb header-scan path.
            patch = _swebench_extract_patch(agent_output)
            diff_verifier_annotation = _annotate_diff_verifier(
                patch=patch,
                repo_dir=repo_dir if isinstance(repo_dir, str) else None,
                mode=_CANARY_DIFF_VERIFIER_MODE,
            )
        finally:
            if tmp_root_to_clean is not None:
                cleanup_status = _cleanup_repo_dir(
                    repo_dir, tmp_root=tmp_root_to_clean
                )
        # Stamp final metadata on the repo_context record so it lands
        # in the per-task summary verbatim.
        repo_context["subprocess_cwd"] = (
            repo_dir if isinstance(repo_dir, str) else None
        )
        repo_context["repo_dir_cleanup_status"] = cleanup_status
        if claim_default_pipeline_learning_evidence:
            cli_complete_payload = cli_result.get("cli_complete_payload")
            cli_outcome = (
                cli_complete_payload.get("outcome")
                if isinstance(cli_complete_payload, dict)
                else None
            )
            archive_trace_dir = (
                output_dir
                / "per_task"
                / f"{_safe_artifact_stem(instance_id)}.trace"
            )
            if cli_outcome != "success":
                learning_evidence = _learning_evidence_no_go(
                    reason_code="cli_outcome_not_success",
                    detail=f"cli_complete outcome was {cli_outcome!r}",
                    run_id=cli_result.get("run_id"),
                    source_trace_dir=cli_result.get("trace_dir"),
                    archived_trace_dir=archive_trace_dir,
                    expect_default_pipeline_learn=expect_default_pipeline_learn,
                )
            else:
                learning_evidence = _validate_learning_evidence(
                    cli_result.get("trace_dir"),
                    cli_result.get("run_id"),
                    archive_trace_dir=archive_trace_dir,
                    expect_default_pipeline_learn=expect_default_pipeline_learn,
                )
        else:
            learning_evidence = _learning_evidence_not_requested()

        # B2 bug 2 (2026-06-10): the nominal path used to trust
        # cli_complete.total_cost_usd alone — hard failures reported $0
        # while the trace carried real cost, and successes under-reported
        # (real $0.79 vs reported $0.30 on the 2026-05-12 N=5, 2.6x).
        nominal_total_cost, nominal_cost_source, nominal_cost_warning = (
            _resolve_total_cost(
                cli_total_cost_usd=cli_result.get("total_cost_usd"),
                observed_event_cost_usd=float(
                    cli_result.get("_observed_event_cost_usd", 0.0) or 0.0
                ),
                had_llm_execution=bool(
                    cli_result.get("_execution_model_ids")
                    or cli_result.get("_observed_model_ids")
                ),
                cli_complete_expected=True,
            )
        )
        summary = {
            "instance_id": instance_id,
            "exit_code": cli_result["exit_code"],
            "latency_ms": cli_result["latency_ms"],
            "total_cost_usd": nominal_total_cost,
            "_total_cost_usd_source": nominal_cost_source,
            "cost_integrity_warning": nominal_cost_warning,
            "extracted_patch_present": bool(patch),
            "extracted_patch_chars": len(patch),
            "mock": False,
            "timeout": False,
            "events_path": str(per_task_events.relative_to(output_dir)),
            "model_id_final": cli_result.get("model_id_final"),
            "provider_final": cli_result.get("provider_final"),
            "_observed_model_ids": cli_result.get("_observed_model_ids", []),
            "_observed_providers": cli_result.get("_observed_providers", []),
            "_assigned_model_ids": cli_result.get("_assigned_model_ids", []),
            "_assigned_providers": cli_result.get("_assigned_providers", []),
            "_execution_model_ids": cli_result.get("_execution_model_ids", []),
            "_execution_providers": cli_result.get("_execution_providers", []),
            "_provider_policy_failure_seen": cli_result.get(
                "_provider_policy_failure_seen",
                False,
            ),
            "_observed_event_cost_usd": cli_result.get("_observed_event_cost_usd", 0.0),
            "_verifier_repair_budget_usd": None,
            "_diff_verifier_mismatches": diff_verifier_annotation[
                "_diff_verifier_mismatches"
            ],
            "_diff_verifier_outcome": diff_verifier_annotation[
                "_diff_verifier_outcome"
            ],
            "_diff_verifier_reasons": diff_verifier_annotation[
                "_diff_verifier_reasons"
            ],
            "stderr_chars": len(cli_result.get("stderr", "")),
            "learning_evidence_boundary": learning_evidence,
            "repo_context": {
                "status": repo_context["repo_context_status"],
                "repo_url": repo_context["repo_url"],
                "base_commit": repo_context["base_commit"],
                "checkout_sha": repo_context["checkout_sha"],
                "clone_elapsed_ms": repo_context["clone_elapsed_ms"],
                "fetch_fallback_used": repo_context["fetch_fallback_used"],
                "subprocess_cwd": repo_context.get("subprocess_cwd"),
                "repo_dir_cleanup_status": repo_context.get("repo_dir_cleanup_status"),
                "failure_reason": repo_context["failure_reason"],
            },
            "prompt_metadata": prompt_metadata,
            # Slice 10A (RF#C MODIFY): topology + control-surface audit
            # so paired-reruns can attribute outcomes to prompt/profile
            # vs bandit-Thompson noise. NOT making topology deterministic.
            "topology_audit": _topology_audit_from_file(per_task_events),
        }

    record = fmt_module.format_patch(instance_id, patch, prefix=prefix)
    return {"summary": summary, "record": record}


# ── Main ────────────────────────────────────────────────────────────────────


async def run(
    instances_json: Path,
    output_dir: Path,
    *,
    mock: bool,
    limit: int,
    budget_usd: float,
    tier: str,
    prefix: str,
    global_budget_usd: float = 25.0,
    task_timeout_s: float = 120.0,
    profile: str = _DEFAULT_PROFILE,
    profile_timeout_override: bool = False,
    manifest_path: Path | None = None,
    grader_preflight_path: Path | None = None,
    ci_green_artifact: Path | None = None,
    provider_allowlist: tuple[str, ...] = ("google", "deepseek"),
    provider_denylist: tuple[str, ...] = ("openai",),
    claim_default_pipeline_learning_evidence: bool = False,
    expect_default_pipeline_learn: bool = False,
    swebench_prompt_profile: str = _DEFAULT_PROMPT_PROFILE,
) -> int:
    if mock and claim_default_pipeline_learning_evidence:
        log.error(
            "--claim-default-pipeline-learning-evidence requires real mode; "
            "mock has no runtime trace"
        )
        return 2

    fmt_module = _load_format_patch_module()

    instances_text = instances_json.read_text(encoding="utf-8")
    instances = json.loads(instances_text)
    if not isinstance(instances, list) or not instances:
        log.error("instances.json empty or wrong shape (expected list)")
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)
    launch_manifest = _write_launch_manifest(
        output_dir=output_dir,
        instances_json=instances_json,
        manifest_path=manifest_path,
        budget_usd=budget_usd,
        global_budget_usd=global_budget_usd,
        task_timeout_s=task_timeout_s,
        profile=profile,
        profile_timeout_override=profile_timeout_override,
        provider_allowlist=provider_allowlist,
        provider_denylist=provider_denylist,
        grader_preflight_path=grader_preflight_path,
        ci_green_artifact=ci_green_artifact,
    )

    # Apply limit
    selected = instances[:limit] if limit > 0 else instances
    log.info(
        "Selected %d/%d tasks (mock=%s, tier=%s, budget_usd=%.2f, prefix=%s)",
        len(selected), len(instances), mock, tier, budget_usd, prefix,
    )

    started_at = datetime.now(timezone.utc)
    cumulative_cost = 0.0
    cumulative_latency_ms = 0
    summaries: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    budget_stop_reasons: list[str] = []

    for i, task in enumerate(selected):
        if not mock and cumulative_cost >= global_budget_usd:
            budget_stop_reasons.append("global_budget_exhausted_before_task")
            log.warning(
                "Global budget exhausted: cumulative_cost=$%.2f >= $%.2f. "
                "Stopping early at task %d/%d.",
                cumulative_cost, global_budget_usd, i, len(selected),
            )
            break
        if not mock and cumulative_cost + budget_usd > global_budget_usd:
            budget_stop_reasons.append("task_budget_would_exceed_global_cap")
            log.warning(
                "Task budget would exceed global cap: $%.2f + $%.2f > $%.2f. "
                "Stopping early at task %d/%d.",
                cumulative_cost, budget_usd, global_budget_usd, i, len(selected),
            )
            break
        try:
            result = await asyncio.wait_for(
                _run_one_task(
                    task,
                    output_dir,
                    mock=mock,
                    budget_usd=budget_usd,
                    tier=tier,
                    provider_allowlist=provider_allowlist,
                    provider_denylist=provider_denylist,
                    prefix=prefix,
                    fmt_module=fmt_module,
                    claim_default_pipeline_learning_evidence=claim_default_pipeline_learning_evidence,
                    expect_default_pipeline_learn=expect_default_pipeline_learn,
                    swebench_prompt_profile=swebench_prompt_profile,
                ),
                timeout=task_timeout_s,
            )
        except asyncio.TimeoutError:
            log.error("Task %s exceeded timeout_s=%.1f", task["instance_id"], task_timeout_s)
            result = _timeout_task_result(
                task,
                output_dir,
                prefix=prefix,
                fmt_module=fmt_module,
                task_timeout_s=task_timeout_s,
                expect_default_pipeline_learn=expect_default_pipeline_learn,
            )
        summaries.append(result["summary"])
        records.append(result["record"])
        cost = result["summary"].get("total_cost_usd") or 0.0
        cumulative_cost += cost if isinstance(cost, (int, float)) else 0.0
        cumulative_latency_ms += result["summary"].get("latency_ms", 0)

    # Write predictions.json (Pro shape)
    predictions_path = output_dir / "predictions.json"
    fmt_module.write_predictions(records, predictions_path)
    predictions_jsonl_path = output_dir / "predictions.jsonl"
    _write_predictions_jsonl(records, summaries, predictions_jsonl_path)
    events_path = output_dir / "events.jsonl"
    _write_aggregate_events(summaries, output_dir=output_dir, output_path=events_path)

    # Write summary
    summary_doc: dict[str, Any] = {
        "run_started_at_utc": started_at.isoformat(),
        "run_ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "mock" if mock else "real",
        "tier": tier if not mock else None,
        "tasks_run": len(summaries),
        "tasks_in_set": len(instances),
        "cumulative_cost_usd": cumulative_cost,
        "cumulative_latency_ms": cumulative_latency_ms,
        "budget": {
            "budget_usd_per_task": budget_usd,
            "global_budget_usd": global_budget_usd,
            "task_timeout_s": task_timeout_s,
            "effective_profile": profile,
            "profile_timeout_default_s": _TIMEOUT_PROFILES.get(profile),
            "profile_timeout_override": profile_timeout_override,
            "stop_reasons": budget_stop_reasons,
        },
        "prompt": {
            "swebench_prompt_profile": swebench_prompt_profile,
            "topology_override_used": False,
            "system_hint_forced": False,
        },
        "predictions_path": str(predictions_path.relative_to(output_dir)),
        "predictions_jsonl_path": str(predictions_jsonl_path.relative_to(output_dir)),
        "events_path": str(events_path.relative_to(output_dir)),
        "per_task_dir": "per_task/",
        "patches_extracted": sum(
            1 for s in summaries if s["extracted_patch_present"]
        ),
        "patches_empty": sum(
            1 for s in summaries if not s["extracted_patch_present"]
        ),
        "task_summaries": summaries,
    }
    evidence_items = [
        item
        for item in (summary.get("learning_evidence_boundary") for summary in summaries)
        if isinstance(item, dict)
    ]
    summary_doc["learning_evidence_gate"] = {
        "claimed": claim_default_pipeline_learning_evidence,
        "status": (
            "NO_GO"
            if any(item.get("status") == "no_go" for item in evidence_items)
            else "PASS" if claim_default_pipeline_learning_evidence else "NOT_CLAIMED"
        ),
        "expect_default_pipeline_learn": expect_default_pipeline_learn,
        "passed": sum(1 for item in evidence_items if item.get("status") == "pass"),
        "failed": sum(1 for item in evidence_items if item.get("status") == "no_go"),
        "skipped": sum(1 for item in evidence_items if item.get("status") == "skipped"),
    }
    budget_gate = {
        "status": "NO_GO" if cumulative_cost > global_budget_usd else "PASS",
        "cumulative_cost_usd": cumulative_cost,
        "global_budget_usd": global_budget_usd,
        "stop_reasons": budget_stop_reasons,
    }
    timeout_gate = {
        "status": (
            "NO_GO"
            if sum(1 for summary in summaries if summary.get("timeout")) > 1
            else "PASS"
        ),
        "timeouts": sum(1 for summary in summaries if summary.get("timeout")),
    }
    summary_doc["budget_gate"] = budget_gate
    summary_doc["timeout_gate"] = timeout_gate
    provider_gate = _provider_gate(
        summaries,
        mock=mock,
        provider_allowlist=provider_allowlist,
        provider_denylist=provider_denylist,
    )
    summary_doc["launch_manifest_path"] = "launch_manifest.json"
    summary_doc["acceptance_gate_results"] = {
        "manifest_gate": launch_manifest["manifest_gate"],
        "budget_gate": budget_gate,
        "timeout_gate": timeout_gate,
        "provider_gate": provider_gate,
        "grading_gate": launch_manifest["grading_gate"],
        "ci_gate": launch_manifest["ci_gate"],
    }
    summary_doc["canary_decision"] = _combine_canary_decision(
        summary_doc["acceptance_gate_results"]
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary_doc, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )

    log.info(
        "Done. predictions=%d patches_extracted=%d cost=$%.4f latency=%.1fs",
        len(records),
        summary_doc["patches_extracted"],
        cumulative_cost,
        cumulative_latency_ms / 1000,
    )
    if claim_default_pipeline_learning_evidence and (
        not summaries
        or any(
            (summary.get("learning_evidence_boundary") or {}).get("status") == "no_go"
            for summary in summaries
        )
    ):
        log.error("Learning side-effect evidence boundary failed")
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances-json",
        type=Path,
        required=True,
        help="path to instances.json from swebench_pro_fetch.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="output dir for predictions.json + per_task/ + summary.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="run at most N tasks (default 1, smoke discipline)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="synthetic patches (NO API spend); validates wire-up only",
    )
    parser.add_argument(
        "--budget-usd",
        type=float,
        default=5.0,
        help="per-task spend cap forwarded to sage CLI (default 5.0)",
    )
    parser.add_argument(
        "--global-budget-usd",
        type=float,
        default=25.0,
        help="global spend cap for this runner invocation (default 25.0)",
    )
    parser.add_argument(
        "--task-timeout-s",
        type=float,
        default=None,
        help=(
            "per-task wall-clock timeout in seconds. Sentinel ``None`` "
            "means: resolve from --profile. Explicit value overrides the "
            "profile-driven default and is reported as a profile_override."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=sorted(_TIMEOUT_PROFILES),
        default=_DEFAULT_PROFILE,
        help=(
            "Named timeout profile. ``default`` keeps 120s for plumbing "
            "smokes; ``graded_patch_generation`` gives the agent 900s for "
            "real-LLM SWE-bench Pro canaries (cgpro DESIGN 2026-05-11 "
            "envelope 600-1200s)."
        ),
    )
    parser.add_argument(
        "--swebench-prompt-profile",
        choices=list(_PROMPT_PROFILES),
        default=_DEFAULT_PROMPT_PROFILE,
        help=(
            "SWE-bench prompt profile. ``canonical`` (default) uses "
            "sage.input.swebench.render_swebench_prompt (with the "
            "mandatory tool-call workflow). ``patch_focused`` is a "
            "canary-local template (cgpro DESIGN 2026-05-11 NEW_SLICE "
            "canary-patch-focused-prompt-profile) that drops the "
            "'≥3 tool calls before patch' mandate while keeping "
            "repo-grounding and a STRICT unified-diff output contract. "
            "Designed to coexist with adaptive topology selection — no "
            "system_hint or topology override."
        ),
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=_DEFAULT_CANARY_MANIFEST_PATH,
        help=(
            "human canary manifest to copy/hash into launch artifacts "
            f"(default {_DEFAULT_CANARY_MANIFEST_PATH})"
        ),
    )
    parser.add_argument(
        "--grader-preflight-path",
        type=Path,
        default=None,
        help="optional SWE-bench Pro grader preflight artifact for GO gating",
    )
    parser.add_argument(
        "--ci-green-artifact",
        type=Path,
        default=None,
        help="optional machine-readable CI-green artifact for GO gating",
    )
    parser.add_argument(
        "--provider-allowlist",
        default="google,deepseek",
        help="comma-separated provider allowlist forwarded to sage run and audit",
    )
    parser.add_argument(
        "--provider-denylist",
        default="openai",
        help="comma-separated provider denylist forwarded to sage run and audit",
    )
    parser.add_argument(
        "--tier",
        default=_DEFAULT_TIER,
        help=f"SAGE_LLM_TIER override (default {_DEFAULT_TIER})",
    )
    parser.add_argument(
        "--prefix",
        default="ygn-sage-arm-d-smoke",
        help="prefix label written into each Pro record",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--claim-default-pipeline-learning-evidence",
        action="store_true",
        help=(
            "Claim oracle-enabled default-pipeline learning evidence; after "
            "each real task, archive the canonical RuntimeEventLog trace, run "
            "the evidence-boundary validator, and fail the harness if it fails."
        ),
    )
    parser.add_argument(
        "--expect-default-pipeline-learn",
        action="store_true",
        help=(
            "With --claim-default-pipeline-learning-evidence, require the "
            "minimal current default Stage 5 learning decision set."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not args.instances_json.is_file():
        log.error(
            "--instances-json not found: %s. "
            "Run `swebench_pro_fetch.py` first.",
            args.instances_json,
        )
        return 2
    if (
        args.expect_default_pipeline_learn
        and not args.claim_default_pipeline_learning_evidence
    ):
        parser.error(
            "--expect-default-pipeline-learn requires "
            "--claim-default-pipeline-learning-evidence"
        )
    if args.mock and args.claim_default_pipeline_learning_evidence:
        parser.error(
            "--claim-default-pipeline-learning-evidence requires real mode; "
            "remove --mock"
        )

    profile_timeout = _TIMEOUT_PROFILES[args.profile]
    if args.task_timeout_s is None:
        effective_task_timeout_s = profile_timeout
        timeout_override = False
    else:
        effective_task_timeout_s = float(args.task_timeout_s)
        timeout_override = True
        log.info(
            "Explicit --task-timeout-s=%.1f overrides profile %r default %.1f",
            effective_task_timeout_s,
            args.profile,
            profile_timeout,
        )

    return asyncio.run(
        run(
            args.instances_json,
            args.output_dir,
            mock=args.mock,
            limit=args.limit,
            budget_usd=args.budget_usd,
            global_budget_usd=args.global_budget_usd,
            task_timeout_s=effective_task_timeout_s,
            profile=args.profile,
            profile_timeout_override=timeout_override,
            manifest_path=args.manifest_path,
            grader_preflight_path=args.grader_preflight_path,
            ci_green_artifact=args.ci_green_artifact,
            provider_allowlist=tuple(
                item.strip() for item in args.provider_allowlist.split(",") if item.strip()
            ),
            provider_denylist=tuple(
                item.strip() for item in args.provider_denylist.split(",") if item.strip()
            ),
            tier=args.tier,
            prefix=args.prefix,
            claim_default_pipeline_learning_evidence=(
                args.claim_default_pipeline_learning_evidence
            ),
            expect_default_pipeline_learn=args.expect_default_pipeline_learn,
            swebench_prompt_profile=args.swebench_prompt_profile,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
