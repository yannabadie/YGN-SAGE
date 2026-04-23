"""SWE-Bench evaluation adapter for YGN-SAGE.

Loads SWE-Bench instances (Lite: 300, Verified: 500) from HuggingFace,
feeds each issue to SAGE's agent loop to generate a patch, then evaluates
via the official swebench Docker-based harness.

Requirements:
    pip install swebench datasets docker

Evaluation flow:
    1. Load dataset from HuggingFace (princeton-nlp/SWE-bench_Lite or _Verified)
    2. For each instance: feed problem_statement to SAGE -> capture generated patch
    3. Write predictions in swebench JSONL format
    4. Build Docker images (swebench harness)
    5. Run evaluation in Docker containers (applies patch, runs tests)
    6. Grade results: resolved = all FAIL_TO_PASS tests now pass

Platform notes:
    - swebench's harness requires Linux `resource` module. On Windows, a stub
      is injected at import time. The actual evaluation runs inside Docker
      (Linux containers via Docker Desktop / WSL2), so this is safe.
    - Docker Desktop with WSL2 backend is REQUIRED on Windows.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import os
import platform
import sys
import time
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any

from sage.bench.runner import BenchReport, TaskResult
from sage.bench.truth_pack import BenchmarkManifest, TaskTrace

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Windows compatibility: stub the `resource` module before importing swebench
# ---------------------------------------------------------------------------
if platform.system() != "Linux" and "resource" not in sys.modules:
    _resource_stub = types.ModuleType("resource")
    setattr(_resource_stub, "RLIMIT_NOFILE", 7)
    setattr(_resource_stub, "getrlimit", lambda _x: (1024, 1048576))
    setattr(_resource_stub, "setrlimit", lambda _x, _y: None)
    sys.modules["resource"] = _resource_stub

# ---------------------------------------------------------------------------
# Dataset names on HuggingFace
# ---------------------------------------------------------------------------
_DATASET_MAP = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full": "princeton-nlp/SWE-bench",
    # Scale AI's Pro split — enterprise-scale long-horizon tasks
    # (OpenSAGE reports 59% on this; it is the target for Sprint 5).
    "pro": "ScaleAI/SWE-bench_Pro",
}

# swebench prediction keys
_KEY_INSTANCE_ID = "instance_id"
_KEY_MODEL = "model_name_or_path"
_KEY_PREDICTION = "model_patch"
# SAGE-specific metadata key (leading underscore = "not part of the official
# SWE-bench harness spec"). Persisted through write_predictions so that
# per-bucket analysis reads the jsonl directly instead of grepping agent
# logs (2026-04-23 emission-format smoke finding #4). The harness tolerates
# extra fields — it reads predictions.jsonl by key, not by schema.
_KEY_EXTRACTION_METHOD = "_extraction_method"

# Sentinel string returned by agent_loop when the LLM produces no content for
# N consecutive steps. MUST stay in sync with phases/learn.py. We classify
# sentinels separately from real patches so reporting doesn't overcount.
_SENTINEL_MARKER = "[sage: agent exited after"


def _classify_prediction(pred: str | None | dict) -> str:
    """Return 'real' | 'sentinel' | 'empty' for a prediction.

    Real patches are non-empty non-sentinel strings. Sentinels indicate the
    agent hit step budget with no content. Empty patches indicate generation
    errors or timeouts.

    Accepts either the raw patch string (legacy) or the full prediction dict
    (preferred, since 2026-04-18 D7 audit). The dict form checks the
    structured_failure metadata first — `_extract_patch` strips sentinel
    text now, so the patch alone can no longer distinguish sentinel-emptied
    from real-empty. The `_structured_failure` field carries that signal
    forward for accurate bucketing in the summary header.
    """
    if isinstance(pred, dict):
        failure = pred.get("_structured_failure")
        if failure == "step_budget_exhausted":
            return "sentinel"
        patch = pred.get(_KEY_PREDICTION)
    else:
        patch = pred
    if not patch:
        return "empty"
    if _SENTINEL_MARKER in patch:
        # Legacy path — pre-D7, sentinel text was emitted as patch directly.
        # Kept for backward compat with old jsonl files.
        return "sentinel"
    return "real"


# ---------------------------------------------------------------------------
# Dataset loading (uses HuggingFace `datasets` directly)
# ---------------------------------------------------------------------------

def _ssl_bypass() -> None:
    """Disable SSL verification for corporate proxy environments."""
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context  # type: ignore[assignment]
    os.environ.setdefault("CURL_CA_BUNDLE", "")
    os.environ.setdefault("REQUESTS_CA_BUNDLE", "")


def load_swebench_dataset(
    dataset: str = "lite",
    split: str = "test",
) -> list[dict[str, Any]]:
    """Load a SWE-Bench dataset from HuggingFace.

    Args:
        dataset: "lite" (300), "verified" (500), or "full" (2294).
        split: HuggingFace split name (default: "test").

    Returns:
        List of instance dicts with keys: repo, instance_id, base_commit,
        patch, test_patch, problem_statement, hints_text, version,
        FAIL_TO_PASS, PASS_TO_PASS, etc.
    """
    _ssl_bypass()
    from datasets import load_dataset as hf_load

    hf_name = _DATASET_MAP.get(dataset, dataset)
    log.info("Loading %s (split=%s) from HuggingFace...", hf_name, split)
    ds = hf_load(hf_name, split=split)
    instances = [dict(row) for row in ds]
    log.info("Loaded %d instances from %s", len(instances), hf_name)
    return instances


# ---------------------------------------------------------------------------
# Prompt engineering for SWE-Bench tasks
# ---------------------------------------------------------------------------
#
# NOTE: AgentSystem.run() takes a single task string (no separate system
# prompt slot at the benchmark boundary). Everything the model needs to know
# about tool-use and iteration must be inside the task text — hence the
# merged template below.
#
# Lessons from 2026-04-08 diagnostic (tool_turn_count=1 for all 5 tasks):
# 1. Routing classified SWE-bench tasks as S2 — fixed via pipeline.run(system_hint=3).
# 2. The richer system-prompt text wasn't wired into the task — fixed below.
# 3. Models rushed to patch after one tool call — the template now MANDATES
#    at least three exploration steps before any patch is written, and forces
#    line-number verification before emitting hunk headers.

from sage.input.swebench import normalize_swebench, render_swebench_prompt


def _build_task_prompt(instance: dict[str, Any]) -> str:
    """Build the task prompt for a SWE-bench instance.

    Thin wrapper around the `sage.input.swebench` normalizer path.
    `AgentSystem.run()` still takes a single string today; C4 will
    teach `perceive()` to consume the full `TaskInput` and render
    the same text through a generic builder. Until then, this keeps
    the bench's external behavior byte-identical.
    """
    return render_swebench_prompt(normalize_swebench(instance))


def _extract_patch(response: str) -> str:
    """Extract a unified diff patch from agent response.

    Handles various output formats:
    - Raw diff (starts with diff --git or ---)
    - Markdown code blocks (```diff ... ```)
    - Mixed text with embedded diff

    Always normalizes to Unix line endings and ensures trailing newline
    (required by `git apply` in Linux Docker containers).
    """
    if not response:
        return ""

    # D7 fix (2026-04-18 audit docs/audits/2026-04-18-astropy-14995-*):
    # sentinel text must NOT be emitted as a patch. Before this guard,
    # astropy-14995 produced a 52-char "PATCH" that was literally
    # "[sage: agent exited after 20 steps with no content]\n" — Docker
    # eval rejects it, but the jsonl counted it as a patch attempt.
    # Return empty so the classifier buckets it as `empty` and reports
    # surface the real failure mode (step-budget exhaustion).
    if _SENTINEL_MARKER in response:
        return ""

    # Normalize line endings (Windows -> Unix) before processing
    response = response.replace("\r\n", "\n").replace("\r", "\n")
    response = response.strip()

    # Case 1: Already a clean diff
    if response.startswith("diff --git") or response.startswith("---"):
        return response + "\n" if not response.endswith("\n") else response

    # Case 2: Markdown code block
    for marker in ["```diff", "```patch", "```"]:
        if marker in response:
            start = response.find(marker)
            start = response.find("\n", start) + 1
            end = response.find("```", start)
            if end > start:
                candidate = response[start:end].strip()
                if candidate and ("---" in candidate or "diff --git" in candidate):
                    return candidate + "\n" if not candidate.endswith("\n") else candidate

    # Case 3: Find diff content anywhere in the response
    lines = response.split("\n")
    diff_lines: list[str] = []
    in_diff = False
    for line in lines:
        if line.startswith("diff --git") or line.startswith("---"):
            in_diff = True
        if in_diff:
            diff_lines.append(line)
        # Stop if we hit a blank line after a diff section ends
        if in_diff and not line.strip() and diff_lines and not diff_lines[-1].startswith(
            ("diff", "---", "+++", "@@", " ", "+", "-")
        ):
            break

    if diff_lines:
        patch = "\n".join(diff_lines).strip()
        # Ensure trailing newline (required by git apply)
        return patch + "\n" if not patch.endswith("\n") else patch

    # F11 audit fix (2026-04-19 docs/audits/2026-04-18-astropy-14995-*):
    # Smoke v10 (F10 synthesizer-no-tools) revealed that a synthesizer
    # without tools blindly forwards whatever the coder emitted — if
    # coder emitted a ```bash grep ...``` exploration command, that
    # gets classified as a "real patch" (non-empty, non-sentinel). The
    # previous fallback returned the full response on the assumption
    # that swebench's evaluator would reject invalid diffs — but our
    # internal classifier counts it as real before evaluator sees it.
    # A real unified diff MUST contain at least one hunk header (@@)
    # OR a diff --git / --- header. If the response has none, it's
    # not a diff and we return "" so D7's `_structured_failure` path
    # captures it as EMPTY.
    if "@@" not in response and "diff --git" not in response and "\n---" not in response:
        return ""

    # Fallback: return the entire response (swebench will reject if not a valid diff)
    # Ensure trailing newline
    return response + "\n" if response and not response.endswith("\n") else response


# ---------------------------------------------------------------------------
# SEARCH/REPLACE emission helpers (Track 2 task 2.2)
#
# Implements the SEARCH/REPLACE patch emission format described in
# docs/superpowers/plans/2026-04-21-semantic-quality-plan.md. This is an
# alternative to raw unified-diff emission that sidesteps the line-counting
# hallucination trap (LLMs are bad at computing hunk headers). The helpers
# are wired into generate_patches by task 2.4; here they are pure functions
# so the tests in tests/test_search_replace_extraction.py can drive them.
#
# Block syntax (Aider / OpenHands convention)::
#
#     ## File: path/to/module.py
#     <<<<<<< SEARCH
#     <exact existing text>
#     =======
#     <replacement text>
#     >>>>>>> REPLACE
#
# Contract points enforced by the tests:
# - search_text and replace_text include the trailing "\n" byte-for-byte.
# - Missing ## File: -> scan repo_dir for a unique file containing
#   search_text; drop the block on 0 or >=2 matches.
# - Malformed block (missing ======= or >>>>>>> REPLACE) -> drop, keep
#   parsing later blocks.
# - Fuzzy match (difflib ratio >= 0.95) only when exact match fails; the
#   hunk's minus side uses the ACTUAL file lines so ``git apply --check``
#   accepts the patch.
# ---------------------------------------------------------------------------

_SR_FILE_MARKER = "## File:"
_SR_SEARCH_MARKER = "<<<<<<< SEARCH"
_SR_SEPARATOR = "======="
_SR_REPLACE_MARKER = ">>>>>>> REPLACE"
_SR_FUZZY_THRESHOLD = 0.95

# ---------------------------------------------------------------------------
# Emission-format env-var gate (Track 2 task 2.3)
# ---------------------------------------------------------------------------
# The bench supports two patch-emission formats:
#   - "unified"         — the legacy ```diff fenced unified-diff format.
#   - "search-replace"  — the SEARCH/REPLACE block format plumbed by T2.1+T2.2.
# ``SAGE_EMISSION_FORMAT`` is the operator-facing toggle. Default is
# "unified" until the paired smoke in T2.5 validates the switch (per the
# decision gate on docs/superpowers/plans/2026-04-21-semantic-quality-plan.md).
_EMISSION_FORMAT_ENV = "SAGE_EMISSION_FORMAT"
_VALID_EMISSION_FORMATS = frozenset({"unified", "search-replace"})


def _get_emission_format() -> str:
    """Read ``SAGE_EMISSION_FORMAT``. Returns ``"unified"`` (default) or
    ``"search-replace"``. Unknown values log WARN and fall back to
    ``"unified"``. The default MUST stay ``"unified"`` until the paired
    smoke validates the switch — red-team plan §5-style decision gate."""
    raw = os.environ.get(_EMISSION_FORMAT_ENV)
    if raw is None:
        return "unified"
    v = raw.strip().lower()
    if v in _VALID_EMISSION_FORMATS:
        return v
    log.warning(
        "%s=%r is not a valid emission format (allowed: %s); falling back to 'unified'",
        _EMISSION_FORMAT_ENV,
        raw,
        sorted(_VALID_EMISSION_FORMATS),
    )
    return "unified"

# Defensive caps for the ``## File:``-less content-scan fallback. SWE-bench
# repos (django, sympy, matplotlib, ...) ship multi-MB binary blobs, cached
# wheels, PNG fixtures — slurping them whole into memory on every block is
# wasteful and memory-risky. The suffix allow-list is intentionally
# conservative: source-code / config / doc extensions only. The size cap
# at 2 MB is comfortably above any real source file but well below any
# data-file-shaped outlier we would want to decode UTF-8.
_SR_MAX_SCAN_BYTES = 2_000_000
_SR_SCAN_SUFFIXES = {
    ".py",
    ".pyi",
    ".pyx",
    ".rst",
    ".md",
    ".txt",
    ".cfg",
    ".toml",
    ".ini",
    ".yaml",
    ".yml",
    ".json",
}


def _extract_search_replace_blocks(
    response: str,
    repo_dir: str | Path,
) -> list[tuple[str, str, str]]:
    """Parse SEARCH/REPLACE blocks from an LLM response.

    Returns a list of ``(file_path, search_text, replace_text)`` tuples in
    the order they appear in ``response``. ``search_text`` and
    ``replace_text`` include the trailing newline of their last line.

    Path resolution:
    - Preferred: the nearest preceding ``## File: <path>`` line gives the
      path verbatim (kept as written in the response).
    - Fallback: if no marker precedes the block, the helper scans
      ``repo_dir`` for files whose contents contain ``search_text``
      verbatim. Exactly one match -> use that file (as a forward-slash
      posix path relative to ``repo_dir``). Zero or >=2 matches -> drop
      the block.

    Malformed blocks (missing ``=======`` separator or ``>>>>>>> REPLACE``
    terminator, or nested SEARCH markers) are dropped; parsing continues
    for subsequent blocks. Empty or whitespace-only response -> ``[]``.
    """
    if not response or not response.strip():
        return []

    repo_path = Path(repo_dir)
    blocks: list[tuple[str, str, str]] = []

    # Line-oriented state machine. A regex-based implementation would
    # span malformed/well-formed block pairs (test case e) and swallow
    # the good block into a single fake match.
    lines = response.splitlines(keepends=True)
    # Ensure each captured line ends in "\n" so the tests' byte-for-byte
    # assertions hold (splitlines(keepends=True) keeps whatever terminator
    # was present; the final line may lack one if the response ended
    # without a trailing newline — we normalise to "\n" on append).
    state = "OUTSIDE"  # OUTSIDE | IN_SEARCH | IN_REPLACE
    current_path: str | None = None  # from most-recent ## File: marker
    pending_path: str | None = None  # path captured when SEARCH opened
    search_buf: list[str] = []
    replace_buf: list[str] = []

    for raw_line in lines:
        # Strip only the trailing newline so we can match markers by equality
        # without a regex, but keep the original line available for buffer
        # appends. Markers always occupy a full line in the conventional
        # format, but we tolerate trailing whitespace (e.g. accidental " \n").
        line_no_nl = raw_line.rstrip("\n").rstrip("\r")
        stripped = line_no_nl.strip()

        if state == "OUTSIDE":
            if stripped.startswith(_SR_FILE_MARKER):
                # Path is the rest of the line after ``## File:``.
                current_path = stripped[len(_SR_FILE_MARKER):].strip()
            elif stripped == _SR_SEARCH_MARKER:
                # Enter IN_SEARCH; "consume" the current_path so that a
                # bare block following immediately cannot reuse it.
                state = "IN_SEARCH"
                pending_path = current_path
                current_path = None
                search_buf = []
                replace_buf = []
            # Any other line outside a block is narrative prose - ignore.
            continue

        if state == "IN_SEARCH":
            if stripped == _SR_SEPARATOR:
                state = "IN_REPLACE"
                continue
            if stripped == _SR_SEARCH_MARKER or stripped == _SR_REPLACE_MARKER:
                # Nested SEARCH or premature REPLACE -> malformed; drop.
                state = "OUTSIDE"
                pending_path = None
                search_buf = []
                replace_buf = []
                # If this is actually a new SEARCH opening, re-enter.
                if stripped == _SR_SEARCH_MARKER:
                    state = "IN_SEARCH"
                    pending_path = current_path
                    current_path = None
                continue
            # Normal content line inside the SEARCH section.
            search_buf.append(raw_line if raw_line.endswith("\n") else raw_line + "\n")
            continue

        if state == "IN_REPLACE":
            if stripped == _SR_REPLACE_MARKER:
                # Well-formed block: resolve path and emit.
                path = pending_path
                search_text = "".join(search_buf)
                replace_text = "".join(replace_buf)
                if path is None:
                    # Content-scan fallback.
                    path = _scan_repo_for_unique_match(repo_path, search_text)
                if path is None:
                    log.warning(
                        "SEARCH/REPLACE block dropped: no ## File: marker "
                        "and search_text did not uniquely match any file "
                        "in %s",
                        repo_path,
                    )
                else:
                    blocks.append((path, search_text, replace_text))
                state = "OUTSIDE"
                pending_path = None
                search_buf = []
                replace_buf = []
                continue
            if stripped == _SR_SEARCH_MARKER or stripped == _SR_SEPARATOR:
                # Nested SEARCH or double separator -> malformed; drop.
                log.warning(
                    "SEARCH/REPLACE block dropped: unexpected '%s' inside "
                    "REPLACE section",
                    stripped,
                )
                state = "OUTSIDE"
                pending_path = None
                search_buf = []
                replace_buf = []
                if stripped == _SR_SEARCH_MARKER:
                    state = "IN_SEARCH"
                    pending_path = current_path
                    current_path = None
                continue
            replace_buf.append(raw_line if raw_line.endswith("\n") else raw_line + "\n")
            continue

    # If we ended mid-block (no closing >>>>>>> REPLACE), drop it - the
    # loop exit already discards search_buf / replace_buf.
    if state != "OUTSIDE":
        log.warning(
            "SEARCH/REPLACE block dropped: response ended before '>>>>>>> REPLACE'"
        )

    return blocks


def _scan_repo_for_unique_match(repo_dir: Path, search_text: str) -> str | None:
    """Scan ``repo_dir`` for files whose UTF-8 contents contain
    ``search_text`` verbatim. Return the posix-relative path on a single
    match, else ``None``.

    Used only as a fallback when the LLM omitted the ``## File:`` marker.

    Defensive perf caps: suffix allow-list (``_SR_SCAN_SUFFIXES``) filters
    to source / config / doc files before any read, and any file whose
    size exceeds ``_SR_MAX_SCAN_BYTES`` is skipped. This avoids slurping
    binary blobs or wheel caches on real SWE-bench repos. Defense in
    depth: the returned path is also confirmed to resolve inside
    ``repo_dir`` so that a malicious symlink can't redirect the scan
    into ``/etc`` or similar.
    """
    repo_root = repo_dir.resolve()
    matches: list[Path] = []
    for p in repo_dir.rglob("*"):
        if not p.is_file():
            continue
        # Skip anything under a .git directory to avoid matching pack files
        # / hooks that happen to contain the search string.
        try:
            if ".git" in p.relative_to(repo_dir).parts:
                continue
        except ValueError:
            continue
        if p.suffix not in _SR_SCAN_SUFFIXES:
            continue
        try:
            size = p.stat().st_size
        except OSError:
            # Broken symlink, permission error, race with rm: skip.
            continue
        if size > _SR_MAX_SCAN_BYTES:
            continue
        try:
            contents = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if search_text in contents:
            matches.append(p)
            if len(matches) > 1:
                return None
    if len(matches) != 1:
        return None
    # Symlink defense in depth: confirm the candidate's resolved location
    # is still inside the repo. rglob() doesn't follow symlinks by
    # default on Path, but a symlink whose target lives outside the tree
    # would silently redirect future reads.
    winner = matches[0]
    try:
        winner.resolve().relative_to(repo_root)
    except ValueError:
        log.warning(
            "SEARCH/REPLACE scan: %s resolves outside repo_dir - dropping",
            winner,
        )
        return None
    return winner.relative_to(repo_dir).as_posix()


def _blocks_to_unified_diff(
    blocks: list[tuple[str, str, str]],
    repo_dir: str | Path,
) -> tuple[str, dict]:
    """Convert SEARCH/REPLACE blocks to a well-formed unified diff.

    For each block:
    1. Read the target file with ``read_text(encoding="utf-8")`` so
       universal-newlines normalises CRLF to LF (matching the LF-only
       search_text that the LLM emitted).
    2. Exact-match first (``find()``); record ``match_kind = "exact"``.
    3. On exact-match failure, slide a window of ``len(search_lines)``
       across the file and keep the window whose
       ``SequenceMatcher.ratio()`` vs search_text is >= 0.95. The minus
       side of the emitted hunk uses the ACTUAL file lines from that
       window so ``git apply --check`` accepts the patch. Record
       ``match_kind = "fuzzy"``.
    4. If neither works, record ``match_kind = "missing"`` and contribute
       no hunk.

    Returns ``(diff_text, {"per_block": [{"file": ..., "match_kind": ...}, ...]})``.
    Caller treats an empty ``diff_text`` as "no patch".
    """
    repo_path = Path(repo_dir)
    repo_root = repo_path.resolve()
    diff_parts: list[str] = []
    per_block: list[dict[str, str]] = []

    for path_str, search_text, replace_text in blocks:
        # Path-traversal guard. The ``## File:`` marker is LLM-controlled and
        # could hold ``../../etc/passwd`` or an absolute Windows path
        # (``C:/Windows/...``). On Windows, ``Path("/repo") / "C:/Windows"``
        # silently resolves to ``C:/Windows`` (the absolute RHS wins). Join,
        # resolve, then require the result to live under repo_root. ``strict=
        # False`` (the default) so non-existent paths still resolve rather
        # than raising — we want the match-kind=missing branch below to be
        # the one that catches "unknown file", not this guard.
        candidate = (repo_path / path_str).resolve()
        try:
            candidate.relative_to(repo_root)
        except ValueError:
            log.warning(
                "SEARCH/REPLACE block: path %r resolves to %s, outside "
                "repo_dir %s - marking missing",
                path_str, candidate, repo_root,
            )
            per_block.append({"file": path_str, "match_kind": "missing"})
            continue
        file_path = candidate
        try:
            file_text = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            log.warning(
                "SEARCH/REPLACE block: could not read %s (%s) - marking missing",
                file_path, exc,
            )
            per_block.append({"file": path_str, "match_kind": "missing"})
            continue

        file_lines = file_text.splitlines(keepends=True)

        hunk: str | None = None
        match_kind: str

        # --- 1. exact match -------------------------------------------------
        idx = file_text.find(search_text)
        if idx != -1:
            start_line = file_text.count("\n", 0, idx) + 1
            minus_lines = search_text.splitlines(keepends=True)
            plus_lines = replace_text.splitlines(keepends=True)
            hunk = _build_hunk(path_str, start_line, minus_lines, plus_lines)
            match_kind = "exact"
        else:
            # --- 2. fuzzy fallback ------------------------------------------
            search_lines = search_text.splitlines(keepends=True)
            n = len(search_lines)
            best_ratio = 0.0
            best_start: int | None = None  # 0-based file-line index
            if n > 0 and n <= len(file_lines):
                for i in range(0, len(file_lines) - n + 1):
                    window = "".join(file_lines[i:i + n])
                    ratio = difflib.SequenceMatcher(
                        None, window, search_text,
                    ).ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_start = i
            if best_start is not None and best_ratio >= _SR_FUZZY_THRESHOLD:
                minus_lines = file_lines[best_start:best_start + n]
                plus_lines = replace_text.splitlines(keepends=True)
                hunk = _build_hunk(path_str, best_start + 1, minus_lines, plus_lines)
                match_kind = "fuzzy"
            else:
                match_kind = "missing"

        per_block.append({"file": path_str, "match_kind": match_kind})
        if hunk is not None:
            diff_parts.append(hunk)

    diff_text = "".join(diff_parts)
    return diff_text, {"per_block": per_block}


def _build_hunk(
    path: str,
    start_line: int,
    minus_lines: list[str],
    plus_lines: list[str],
) -> str:
    """Build a single-file unified-diff chunk with no context lines.

    The chunk starts with ``diff --git a/<path> b/<path>`` and the pair
    of ``---``/``+++`` header lines, followed by one hunk composed of all
    minus lines then all plus lines. ``path`` is used verbatim; callers
    pass a forward-slash posix path.

    All ``minus_lines`` and ``plus_lines`` MUST already end with ``\\n``
    except possibly the very last line if the original file/replacement
    lacked a trailing newline.

    "No newline at end of file" semantics: we emit ``\\ No newline at
    end of file`` ONLY when the hunk covers the file's final line and
    that line lacks a terminating newline. In practice this invariant
    holds because callers derive ``minus_lines`` via
    ``splitlines(keepends=True)`` on the file bytes, which preserves
    the trailing-newline signal — an absent final ``\\n`` on
    ``minus_lines[-1]`` means the hunk really does reach EOF. If a
    future caller feeds an arbitrary mid-file window whose last line
    happens to lack ``\\n``, the marker would be emitted incorrectly;
    don't do that.
    """
    # Detect missing trailing newlines on the last minus/plus line. If the
    # block includes the file's trailing "\n" (the common SWE-bench case)
    # this never fires.
    minus_noeol = bool(minus_lines) and not minus_lines[-1].endswith("\n")
    plus_noeol = bool(plus_lines) and not plus_lines[-1].endswith("\n")

    header = (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
    )
    hunk_header = (
        f"@@ -{start_line},{len(minus_lines)} "
        f"+{start_line},{len(plus_lines)} @@\n"
    )

    body: list[str] = []
    for i, ln in enumerate(minus_lines):
        if i == len(minus_lines) - 1 and minus_noeol:
            body.append("-" + ln + "\n")
            body.append("\\ No newline at end of file\n")
        else:
            body.append("-" + ln)
    for i, ln in enumerate(plus_lines):
        if i == len(plus_lines) - 1 and plus_noeol:
            body.append("+" + ln + "\n")
            body.append("\\ No newline at end of file\n")
        else:
            body.append("+" + ln)

    return header + hunk_header + "".join(body)


# ---------------------------------------------------------------------------
# SWE-Bench Adapter
# ---------------------------------------------------------------------------

@dataclass
class SWEBenchResult:
    """Per-instance evaluation result."""
    instance_id: str
    repo: str
    resolved: bool
    patch_generated: bool
    patch_applied: bool
    error: str = ""
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    system_used: int = 0


class SWEBenchBench:
    """SWE-Bench evaluation adapter for YGN-SAGE.

    Generates patches via AgentSystem, then evaluates them using the
    official swebench Docker harness.

    Args:
        system: AgentSystem instance (from boot.py).
        event_bus: EventBus for progress events.
        dataset: "lite" (300), "verified" (500), or "full" (2294).
        timeout_per_task: Max seconds for agent to generate a patch.
        eval_timeout: Max seconds for Docker evaluation per instance.
        max_workers: Parallel Docker evaluation workers.
        run_id: Identifier for this evaluation run.
    """

    def __init__(
        self,
        system: Any = None,
        event_bus: Any = None,
        dataset: str = "lite",
        # 120s was too tight once topology + tool-use was the default: the
        # 2026-04-17 F1 smoke ran every S3 task over the wall (~127s) and
        # timed out before the synthesizer emitted its patch. 300s buys the
        # agent loop (max_steps=20 for S3) enough time to explore and
        # finalize. Bench adapters can still override per-run.
        timeout_per_task: float = 300.0,
        eval_timeout: int = 300,
        max_workers: int = 4,
        run_id: str | None = None,
    ):
        if dataset not in _DATASET_MAP:
            raise ValueError(
                f"Unknown dataset '{dataset}'. Supported: {list(_DATASET_MAP.keys())}"
            )
        self.system = system
        self.event_bus = event_bus
        self.dataset = dataset
        self.timeout_per_task = timeout_per_task
        self.eval_timeout = eval_timeout
        self.max_workers = max_workers
        self.run_id = run_id or f"sage-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        self.manifest: BenchmarkManifest | None = None

    @staticmethod
    def _setup_repo(instance: dict[str, Any]) -> str | None:
        """Clone the repo at base_commit into a temp directory.

        Returns the path to the repo root, or None if clone fails.
        The caller must chdir into it and clean up after.
        """
        import subprocess, tempfile

        repo = instance.get("repo", "")  # e.g. "astropy/astropy"
        base_commit = instance.get("base_commit", "")
        if not repo or not base_commit:
            return None

        repo_url = f"https://github.com/{repo}.git"
        tmp = tempfile.mkdtemp(prefix="sage_swe_")
        repo_dir = os.path.join(tmp, repo.split("/")[-1])

        try:
            # Clone and checkout the exact base_commit.
            # Strategy: shallow clone, then deepen/fetch if commit not present.
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, repo_dir],
                capture_output=True, timeout=180, check=True,
            )
            # Fetch the specific commit (may not be in shallow clone)
            result = subprocess.run(
                ["git", "checkout", base_commit],
                cwd=repo_dir, capture_output=True, timeout=30,
            )
            if result.returncode != 0:
                # Commit not in shallow clone — fetch it specifically
                subprocess.run(
                    ["git", "fetch", "--depth", "1", "origin", base_commit],
                    cwd=repo_dir, capture_output=True, timeout=120, check=True,
                )
                subprocess.run(
                    ["git", "checkout", base_commit],
                    cwd=repo_dir, capture_output=True, timeout=30, check=True,
                )
            log.info("Repo cloned: %s @ %s -> %s", repo, base_commit[:8], repo_dir)
            return repo_dir
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            log.warning("Repo clone failed for %s: %s", repo, e)
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)
            return None

    # ------------------------------------------------------------------
    # Phase 1: Generate patches
    # ------------------------------------------------------------------

    async def generate_patches(
        self,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Generate patches for SWE-Bench instances via AgentSystem.

        ``offset`` skips the first N instances. Useful to avoid tasks the
        model has memorized (well-known SWE-bench Lite first few are
        astropy/scikit-learn classics that show up in training data). If
        offset >= len(instances), returns an empty list.

        Returns list of prediction dicts in swebench format:
        {instance_id, model_name_or_path, model_patch, ...metadata}
        """
        if self.system is None:
            log.error("No AgentSystem configured")
            return []

        instances = load_swebench_dataset(self.dataset)
        if offset:
            instances = instances[offset:]
        if limit is not None:
            instances = instances[:limit]

        # Detect model info
        model_id = "unknown"
        provider_name = ""
        if hasattr(self.system, "agent_loop"):
            llm = getattr(self.system.agent_loop, "_llm", None)
            if llm:
                model_id = getattr(llm, "model_id", "unknown")
                provider_name = type(llm).__name__

        self.manifest = BenchmarkManifest(
            benchmark=f"swebench_{self.dataset}",
            model=model_id,
            provider=provider_name,
        )

        # Refresh provider exclusion list at batch start: a provider marked
        # DEAD at boot may have recovered (quota reset, Gemini outage over).
        # Re-probes any exclusion older than TTL; syncs new state to the
        # Rust ModelAssigner so routing decisions use current reality.
        try:
            pipeline = getattr(self.system, "pipeline", None)
            pool = getattr(pipeline, "provider_pool", None) if pipeline else None
            assigner = getattr(pipeline, "assigner", None) if pipeline else None
            if pool and hasattr(pool, "refresh_exclusion_list"):
                dead_after = await pool.refresh_exclusion_list(model_assigner=assigner)
                log.info(
                    "[generate_patches] provider exclusion refreshed: still dead=%s",
                    dead_after,
                )
        except Exception as exc:  # noqa: BLE001 - best-effort refresh, don't block the run
            log.warning("[generate_patches] exclusion refresh failed: %s", exc)

        predictions: list[dict[str, Any]] = []

        for i, instance in enumerate(instances):
            instance_id = instance["instance_id"]
            # C4 (2026-04-22): pass TaskInput directly; AgentSystem.run()
            # dispatches to render_swebench_prompt internally. The
            # module-level `_build_task_prompt` wrapper is preserved
            # for the byte-identity regression test in
            # tests/test_input_swebench.py.
            task_input = normalize_swebench(instance)
            t0 = time.perf_counter()
            error = ""
            system_used = 0
            patch = ""
            structured_failure = ""  # D7 audit: track step_budget_exhausted
            tool_call_count = 0
            tool_turn_count = 0
            executed_commands: list[str] = []
            execution_path = ""
            # T2.4: attribute which extraction path produced the patch so
            # post-hoc paired smokes can bucket empty outcomes by cause.
            # Values: "unified" | "search-replace-exact" | "search-replace-fuzzy"
            # | "search-replace-missing" | "empty". Initialised to "empty" so
            # a timeout / exception before the extractor runs records the
            # right bucket.
            extraction_method = "empty"
            repair_stage = ""

            # Clone repo at base_commit so agent tools (execute_bash) can read code
            repo_dir = None
            original_cwd = os.getcwd()
            try:
                repo_dir = self._setup_repo(instance)
                if repo_dir:
                    os.chdir(repo_dir)
            except Exception as e:
                log.warning("[%s] Repo setup failed: %s — agent runs without code access", instance_id, e)

            try:
                # SWE-bench tasks are always S3 (complex multi-step code work).
                # Skip router misclassification by hinting explicitly.
                response = await asyncio.wait_for(
                    self.system.run(task_input, system_hint=3),
                    timeout=self.timeout_per_task,
                )
                # D7 audit (2026-04-18): flag sentinel BEFORE _extract_patch
                # strips it — _extract_patch intentionally returns "" for
                # sentinel, so the prediction dict needs an auxiliary flag
                # to distinguish sentinel-emptied from real-empty.
                if response and _SENTINEL_MARKER in response:
                    structured_failure = "step_budget_exhausted"
                patch = _extract_patch(response)
                if patch:
                    extraction_method = "unified"

                # T2.4: SEARCH/REPLACE fallback. Only runs when the env
                # gate opts in AND the unified extractor returned empty —
                # the advisor-flagged graceful-degradation contract ("the
                # cleanest patch we got", not "the format we asked for")
                # is what the ordering enforces. Step 7 of the Mandatory
                # Workflow still references a ```diff fence in both
                # templates, so models will sometimes emit a diff even
                # under SR mode; that path wins when it yields content.
                if (
                    not patch
                    and response
                    and _get_emission_format() == "search-replace"
                    and repo_dir
                ):
                    raw_blocks = _extract_search_replace_blocks(response, repo_dir)
                    # Normalise backslash paths (spec reviewer flag):
                    # LLMs on Windows sometimes emit ``pkg\mod.py`` even
                    # under a forward-slash prompt. Doing this inside
                    # the bench wiring (not inside the extractor) keeps
                    # the extractor's contract tests byte-stable.
                    normalised_blocks = [
                        (str(PurePath(p)).replace("\\", "/"), s, r)
                        for p, s, r in raw_blocks
                    ]
                    log.info(
                        "[%s] SR fallback: %d raw blocks, %d after normalisation",
                        instance_id, len(raw_blocks), len(normalised_blocks),
                    )
                    if normalised_blocks:
                        sr_diff, sr_meta = _blocks_to_unified_diff(
                            normalised_blocks, repo_dir,
                        )
                        if sr_diff:
                            patch = sr_diff
                            kinds = [
                                b["match_kind"] for b in sr_meta.get("per_block", [])
                            ]
                            # Precedence: any fuzzy wins, else all-exact
                            # counts as exact; any missing without a
                            # concrete exact/fuzzy alongside collapses to
                            # missing. Note the distinction vs `"empty"`:
                            #   - ``"empty"`` = extractor returned ``[]``
                            #     (no SR markers parsed from the response;
                            #     the model emitted prose or a fence we
                            #     did not recognise).
                            #   - ``"search-replace-missing"`` = blocks
                            #     parsed fine, but none of their
                            #     ``search_text`` matched any repo file
                            #     (exact or fuzzy). T2.5 per-bucket
                            #     analysis keys off this split.
                            if "fuzzy" in kinds:
                                extraction_method = "search-replace-fuzzy"
                            elif "exact" in kinds:
                                extraction_method = "search-replace-exact"
                            else:
                                extraction_method = "search-replace-missing"
                        else:
                            # Blocks parsed but none matched — mark missing
                            # so the decision-gate histogram surfaces it.
                            extraction_method = "search-replace-missing"

                # v16 fix (2026-04-21): validate + repair malformed patches
                # before emission. Addresses the astropy-7746 / django-10914
                # ERROR bucket (LLM hallucinates hunk-header counts or
                # stale context lines on large files). Two-stage:
                #   1. programmatic counts-fix (zero-cost)
                #   2. LLM one-shot repair with `git apply --check` stderr
                # See docs/benchmarks/2026-04-21-swebench-v15-eval-results.md
                # and sage.bench.swebench_patch_repair.
                if patch and repo_dir:
                    from sage.bench.swebench_patch_repair import try_repair_patch
                    llm_handle = getattr(
                        getattr(self.system, "agent_loop", None), "_llm", None,
                    )
                    patch, repair_stage = await try_repair_patch(
                        patch=patch,
                        repo_dir=repo_dir,
                        llm=llm_handle,
                        problem_statement=instance.get("problem_statement", ""),
                        instance_id=instance_id,
                        llm_timeout=60.0,
                    )

                pipeline_ctx = getattr(getattr(self.system, "pipeline", None), "last_context", None)
                execution_path = getattr(self.system, "_last_execution_path", "")
                system_used = (
                    getattr(pipeline_ctx, "system", 0)
                    or getattr(getattr(self.system, "agent_loop", None), "_last_routing_system", 0)
                    or 2
                )
                tool_call_count = int(getattr(pipeline_ctx, "tool_call_count", 0) or 0)
                tool_turn_count = int(getattr(pipeline_ctx, "tool_turn_count", 0) or 0)
                executed_commands = list(getattr(pipeline_ctx, "executed_commands", []) or [])
            except asyncio.TimeoutError:
                error = f"generation_timeout_{self.timeout_per_task:.0f}s"
                log.warning("[%s] Generation timed out", instance_id)
            except Exception as e:
                # Include the exception type + traceback so "Generation failed: 0"
                # type of mysteries are debuggable straight from the log.
                import traceback
                error = f"{type(e).__name__}: {str(e) or repr(e)}"[:200]
                log.error(
                    "[%s] Generation failed: %s\n%s",
                    instance_id, error, traceback.format_exc(),
                )
            finally:
                # Restore cwd and cleanup repo
                os.chdir(original_cwd)
                if repo_dir:
                    import shutil
                    try:
                        shutil.rmtree(repo_dir, ignore_errors=True)
                    except Exception:
                        pass

            latency = (time.perf_counter() - t0) * 1000
            cost = getattr(
                getattr(self.system, "agent_loop", None), "total_cost_usd", 0.0
            )

            predictions.append({
                _KEY_INSTANCE_ID: instance_id,
                _KEY_MODEL: f"sage/{model_id}",
                _KEY_PREDICTION: patch,
                # Metadata (prefixed with _ to not interfere with swebench)
                "_latency_ms": round(latency, 1),
                "_cost_usd": round(cost, 6),
                "_system_used": system_used,
                "_tool_call_count": tool_call_count,
                "_tool_turn_count": tool_turn_count,
                "_executed_commands": executed_commands,
                "_execution_path": execution_path,
                "_error": error,
                "_structured_failure": structured_failure,  # D7 audit
                "_repo": instance["repo"],
                "_repair_stage": repair_stage,  # v16: "", unchanged, programmatic_counts, llm_repair, failed
                "_extraction_method": extraction_method,  # T2.4: unified | search-replace-{exact,fuzzy,missing} | empty
            })

            self.manifest.add(TaskTrace(
                task_id=instance_id,
                passed=False,  # Unknown until evaluation
                latency_ms=round(latency, 1),
                cost_usd=round(cost, 6),
                model=model_id,
                routing=f"S{system_used}",
                error=error[:200] if error else "",
                meta={
                    "execution_path": execution_path,
                    "tool_call_count": tool_call_count,
                    "tool_turn_count": tool_turn_count,
                    "executed_commands": executed_commands,
                },
            ))

            if self.event_bus:
                from sage.agent_loop import AgentEvent
                self.event_bus.emit(AgentEvent(
                    type="BENCH_RESULT",
                    step=i + 1,
                    timestamp=time.time(),
                    meta={
                        "benchmark": f"swebench_{self.dataset}",
                        "task_id": instance_id,
                        "system_used": system_used,
                        "tool_call_count": tool_call_count,
                        "tool_turn_count": tool_turn_count,
                        "latency_ms": round(latency, 1),
                        "patch_len": len(patch),
                        "progress": f"{i + 1}/{len(instances)}",
                    },
                ))

            has_patch = "PATCH" if patch else "EMPTY"
            status = has_patch if not error else f"ERR:{error[:30]}"
            print(
                f"  [{i + 1}/{len(instances)}] {instance_id}: "
                f"{status} ({latency:.0f}ms, {len(patch)} chars)",
                flush=True,
            )

        return predictions

    # ------------------------------------------------------------------
    # Phase 2: Write predictions file
    # ------------------------------------------------------------------

    def write_predictions(
        self,
        predictions: list[dict[str, Any]],
        path: str | Path,
    ) -> Path:
        """Write predictions in swebench JSONL format.

        Format per line: {instance_id, model_name_or_path, model_patch}
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for pred in predictions:
                entry = {
                    _KEY_INSTANCE_ID: pred[_KEY_INSTANCE_ID],
                    _KEY_MODEL: pred[_KEY_MODEL],
                    _KEY_PREDICTION: pred[_KEY_PREDICTION],
                }
                # Persist SAGE-specific metadata when the annotator set it
                # (generate_patches always does since T2.4; older callers or
                # hand-built fixtures may not). Guarded on ``in`` rather than
                # ``.get()`` so we never write a ``_extraction_method: None``
                # record — absence stays absence.
                if _KEY_EXTRACTION_METHOD in pred:
                    entry[_KEY_EXTRACTION_METHOD] = pred[_KEY_EXTRACTION_METHOD]
                f.write(json.dumps(entry) + "\n")

        log.info("Wrote %d predictions to %s", len(predictions), path)
        return path

    # ------------------------------------------------------------------
    # Phase 3: Docker-based evaluation (official swebench harness)
    # ------------------------------------------------------------------

    def evaluate_with_harness(
        self,
        predictions_path: str | Path,
        dataset_name: str | None = None,
    ) -> dict[str, Any]:
        """Run official swebench Docker evaluation (swebench >= 4.1.0).

        This builds Docker images for each repo/version, applies the patch,
        runs the test suite, and grades results.

        Returns dict with:
            resolved_ids: list of resolved instance IDs
            completed_ids: list of completed (evaluated) instance IDs
            error_ids: list of instances that errored during evaluation
            resolved_rate: float
        """
        try:
            from swebench.harness.run_evaluation import (
                run_instances,
                get_dataset_from_preds,
                build_env_images,
                clean_images,
                list_images,
                load_swebench_dataset as swe_load_dataset,
            )
            from swebench.harness.utils import get_predictions_from_file
            from swebench.harness.run_evaluation import make_run_report
        except ImportError as e:
            return {
                "error": f"swebench harness not available: {e}",
                "resolved_ids": [],
                "resolved_rate": 0.0,
            }

        # Inject corporate CA bundle into the base-image build context so
        # `wget` inside the container can resolve repo.anaconda.com behind
        # TLS-inspecting proxies. No-op outside corporate networks.
        from sage.bench.swebench_ca_patch import apply_corporate_ca_patch
        apply_corporate_ca_patch()

        import docker

        predictions_path = str(predictions_path)
        hf_name = dataset_name or _DATASET_MAP[self.dataset]

        # Load predictions
        predictions = get_predictions_from_file(
            predictions_path, hf_name, "test"
        )
        predictions_dict = {p[_KEY_INSTANCE_ID]: p for p in predictions}

        # Get filtered dataset (excludes completed instances)
        dataset = get_dataset_from_preds(
            dataset_name=hf_name,
            split="test",
            instance_ids=list(predictions_dict.keys()),
            predictions=predictions_dict,
            run_id=self.run_id,
            rewrite_reports=False,
        )

        # Load full dataset for final report
        full_dataset = swe_load_dataset(
            hf_name, "test", list(predictions_dict.keys())
        )

        client = docker.from_env()

        if not dataset:
            log.info("No instances to run (all completed or no valid predictions)")
        else:
            log.info("Evaluating %d instances", len(dataset))

            print(f"\n  Building Docker images and evaluating {len(dataset)} instances...")
            print(f"  Run ID: {self.run_id}")
            print(f"  Timeout per instance: {self.eval_timeout}s")
            print(f"  Max workers: {self.max_workers}")
            print()

            existing_images = list_images(client)

            # Build environment images first
            build_env_images(
                client,
                dataset,
                False,  # force_rebuild
                self.max_workers,
                None,  # namespace
                "latest",  # instance_image_tag
                "latest",  # env_image_tag
            )

            # Run evaluation
            run_instances(
                predictions=predictions_dict,
                instances=dataset,
                cache_level="env",
                clean=False,
                force_rebuild=False,
                max_workers=self.max_workers,
                run_id=self.run_id,
                timeout=self.eval_timeout,
            )

            # Clean up build-only images
            clean_images(client, existing_images, "env", False)

        # Generate final report
        report_path = make_run_report(
            predictions=predictions_dict,
            full_dataset=full_dataset,
            run_id=self.run_id,
            client=client,
        )

        # Parse report
        if report_path and report_path.exists():
            report = json.loads(report_path.read_text())
        else:
            report = {}

        resolved_ids = report.get("resolved_ids", [])
        completed_ids = report.get("completed_ids", [])
        error_ids = report.get("error_ids", [])
        total = len(predictions)

        return {
            "resolved_ids": resolved_ids,
            "completed_ids": completed_ids,
            "error_ids": error_ids,
            "total": total,
            "resolved": len(resolved_ids),
            "resolved_rate": len(resolved_ids) / max(total, 1),
            "report_path": str(report_path) if report_path else None,
        }

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    async def run(self, limit: int | None = None, offset: int = 0) -> BenchReport:
        """Full pipeline: generate patches, evaluate with Docker harness.

        Returns a BenchReport compatible with the existing benchmark framework.
        """
        # Phase 1: Generate patches
        predictions = await self.generate_patches(limit=limit, offset=offset)
        if not predictions:
            return BenchReport.from_results(
                f"swebench_{self.dataset}", [],
                model_config={"model": self.manifest.model if self.manifest else "unknown"},
            )

        # Phase 2: Write predictions
        import tempfile
        out_dir = Path(tempfile.mkdtemp(prefix="sage_swebench_"))
        preds_path = self.write_predictions(predictions, out_dir / "predictions.jsonl")

        # Phase 3: Evaluate
        print(f"\n  Predictions saved to: {preds_path}")
        print(f"  Starting Docker evaluation...")

        eval_results = self.evaluate_with_harness(preds_path)

        if "error" in eval_results and eval_results.get("resolved_rate", 0) == 0:
            log.error("Docker evaluation failed: %s", eval_results["error"])
            print(f"\n  Docker evaluation failed: {eval_results['error']}")
            print(f"  Predictions are saved at: {preds_path}")
            print(f"  You can evaluate manually with:")
            print(f"    python -m swebench.harness.run_evaluation \\")
            print(f"      --predictions_path {preds_path} \\")
            print(f"      --dataset_name {_DATASET_MAP[self.dataset]} \\")
            print(f"      --run_id {self.run_id}")
            print()

            # Return generation-only results
            task_results = self._predictions_to_task_results(predictions, set())
            return BenchReport.from_results(
                f"swebench_{self.dataset}", task_results,
                model_config={"model": self.manifest.model if self.manifest else "unknown"},
            )

        # Phase 4: Build task results
        resolved_set = set(eval_results.get("resolved_ids", []))
        task_results = self._predictions_to_task_results(predictions, resolved_set)

        # Update manifest traces
        if self.manifest:
            for trace, result in zip(self.manifest.traces, task_results):
                trace.passed = result.passed

        resolved_count = sum(1 for r in task_results if r.passed)
        total = len(task_results)
        print(f"\n  SWE-Bench {self.dataset}: {resolved_count}/{total} resolved "
              f"({resolved_count/max(total,1):.1%})")

        return BenchReport.from_results(
            f"swebench_{self.dataset}", task_results,
            model_config={"model": self.manifest.model if self.manifest else "unknown"},
        )

    async def run_generate_only(self, limit: int | None = None, offset: int = 0) -> Path:
        """Generate patches and save predictions file (skip Docker evaluation).

        Useful when Docker is not available or for deferred evaluation on Linux.
        Returns path to the predictions JSONL file.
        """
        predictions = await self.generate_patches(limit=limit, offset=offset)
        if not predictions:
            raise RuntimeError("No predictions generated")

        import tempfile
        out_dir = Path(tempfile.mkdtemp(prefix="sage_swebench_"))
        preds_path = self.write_predictions(predictions, out_dir / "predictions.jsonl")

        # Also save the full predictions with metadata
        meta_path = out_dir / "predictions_meta.json"
        meta_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

        # Print summary — classify real vs sentinel vs empty so the header
        # can't lie. A "sentinel" patch is the agent_loop fallback string
        # when the LLM produced zero content for N steps; counting it as a
        # real patch (as the pre-fix code did) misreported 6 "patches" on
        # a run that had only 1 real patch.
        # D7 audit: pass the full prediction dict so _classify_prediction can
        # read `_structured_failure` (sentinel was stripped from the patch
        # at extract time, so the patch alone can't distinguish anymore).
        real_count = sum(1 for p in predictions if _classify_prediction(p) == "real")
        sentinel_count = sum(1 for p in predictions if _classify_prediction(p) == "sentinel")
        empty_count = sum(1 for p in predictions if _classify_prediction(p) == "empty")
        errors_count = sum(1 for p in predictions if p.get("_error"))

        print(f"\n  Generation complete:")
        print(f"    Total instances: {len(predictions)}")
        print(f"    Real patches:    {real_count}")
        print(f"    Sentinels:       {sentinel_count}  (agent exited with no content)")
        print(f"    Empty:           {empty_count}  (generation failed)")
        print(f"    Errors:          {errors_count}")
        print(f"\n  Predictions: {preds_path}")
        print(f"  Metadata: {meta_path}")
        print(f"\n  To evaluate (requires Docker with Linux containers):")
        print(f"    python -m swebench.harness.run_evaluation \\")
        print(f"      --predictions_path {preds_path} \\")
        print(f"      --dataset_name {_DATASET_MAP[self.dataset]} \\")
        print(f"      --run_id {self.run_id}")
        print()

        return preds_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _predictions_to_task_results(
        self,
        predictions: list[dict[str, Any]],
        resolved_set: set[str],
    ) -> list[TaskResult]:
        """Convert predictions + evaluation results to TaskResult list."""
        results: list[TaskResult] = []
        for pred in predictions:
            instance_id = pred[_KEY_INSTANCE_ID]
            resolved = instance_id in resolved_set
            has_patch = bool(pred[_KEY_PREDICTION])
            error = pred.get("_error", "")
            if not has_patch and not error:
                error = "empty_patch"

            results.append(TaskResult(
                task_id=instance_id,
                passed=resolved,
                system_used=pred.get("_system_used", 0),
                latency_ms=pred.get("_latency_ms", 0.0),
                cost_usd=pred.get("_cost_usd", 0.0),
                error=error,
            ))
        return results


# ---------------------------------------------------------------------------
# Standalone evaluation (for pre-generated predictions)
# ---------------------------------------------------------------------------

def evaluate_predictions(
    predictions_path: str,
    dataset: str = "lite",
    run_id: str | None = None,
    timeout: int = 300,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Evaluate a pre-generated predictions file with the swebench harness.

    This is useful when predictions were generated on one machine (e.g., Windows)
    and evaluation needs to happen on another (e.g., Linux with Docker).

    Args:
        predictions_path: Path to JSONL predictions file.
        dataset: "lite", "verified", or "full".
        run_id: Optional run identifier.
        timeout: Timeout per instance in seconds.
        max_workers: Number of parallel evaluation workers.

    Returns:
        Evaluation results dict.
    """
    bench = SWEBenchBench(
        dataset=dataset,
        eval_timeout=timeout,
        max_workers=max_workers,
        run_id=run_id,
    )
    return bench.evaluate_with_harness(predictions_path)


# ---------------------------------------------------------------------------
# Quick dataset info
# ---------------------------------------------------------------------------

def dataset_info(dataset: str = "lite") -> dict[str, Any]:
    """Print summary information about a SWE-Bench dataset."""
    instances = load_swebench_dataset(dataset)

    repos: dict[str, int] = {}
    for inst in instances:
        repo = inst["repo"]
        repos[repo] = repos.get(repo, 0) + 1

    difficulties: dict[str, int] = {}
    for inst in instances:
        diff = inst.get("difficulty", "unknown")
        difficulties[diff] = difficulties.get(diff, 0) + 1

    return {
        "dataset": dataset,
        "hf_name": _DATASET_MAP.get(dataset, dataset),
        "total_instances": len(instances),
        "repos": dict(sorted(repos.items(), key=lambda x: -x[1])),
        "repo_count": len(repos),
        "difficulties": difficulties,
        "columns": list(instances[0].keys()) if instances else [],
    }
