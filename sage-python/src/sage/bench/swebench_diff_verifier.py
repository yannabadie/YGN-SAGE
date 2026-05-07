"""Pre-emission diff-context verifier for SWE-bench.

Spec:
    docs/superpowers/specs/2026-04-23-diff-context-verifier-design.md

The canonical API is ``verify_diff_context_with_reasons()``, which
returns both content mismatch records and reason/outcome telemetry.
The legacy ``verify_diff_context()`` wrapper returns only mismatches
for callers that still need the old repair-mode shape.

This verifier still checks unified diffs only. It verifies old-side
context and removed lines against repo bytes, and separately reports
structural/no-op cases such as malformed hunk headers, header/body
count drift, file creation/deletion, and non-unified input so that an
empty mismatch list no longer means both "clean" and "no opinion".

``repair_with_verifier_feedback()`` remains the A3 (2026-04-24)
one-shot LLM repair path fed by real ``HunkMismatch`` records. A22
reason telemetry intentionally does not drive that prompt path.

Match policy
------------
* Exact string equality first. If every context and removed line
  matches the file byte-for-byte: clean.
* Otherwise, compare whitespace-stripped forms:
  * Stripped lines equal + ``difflib.SequenceMatcher.ratio()`` >= 0.95:
    accepted as clean. Whitespace drift within tolerance.
  * Stripped lines equal + ratio < 0.95:
    ``kind="fuzzy_below_threshold"``. Purely whitespace divergence
    heavy enough to drop the raw ratio below the SR-parity threshold.
  * Stripped lines differ: ``kind="content_mismatch"``, regardless of
    ratio. Semantic divergence is never accepted; the 0.95 threshold
    gates whitespace tolerance only.

The spec (2026-04-23-diff-context-verifier-design.md) literal policy
was "exact or ratio >= 0.95", but the verbatim astropy-14182 Arm B
emitted patch has ONE hallucinated line (``Table.__init__`` vs real
``FixedWidth.__init__``) in a 6-line body: SequenceMatcher ratio
0.956 > 0.95. That policy would have let the very hallucination the
feature exists to catch slip through. Spec open question #1 left room
for exactly this data-driven clarification; value 0.95 is unchanged,
its domain of applicability narrowed.

The module intentionally ships no external deps. ``difflib`` is stdlib;
the unified-diff parser here is regex plus line accumulation, matching
the shape emitted by both ``git diff`` and our own
``_blocks_to_unified_diff``.
"""
from __future__ import annotations

import difflib
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


_FUZZY_THRESHOLD = 0.95
"""Mirror of the SR extractor's threshold. Keep aligned; if this ever
forks, move both into ``swebench_emission_constants`` per spec guidance."""


_HUNK_HEADER_RE = re.compile(
    r"^@@\s+-(?P<old_start>\d+)(?:,(?P<old_count>\d+))?\s+"
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))?\s+@@"
)
"""Matches a unified-diff hunk header. Trailing function context after
the second ``@@`` is intentionally unconsumed."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


DiffVerifierReason = Literal[
    "clean",
    "content_mismatch",
    "fuzzy_below_threshold",
    "file_missing",
    "malformed_hunk_header",
    "hunk_body_count_mismatch",
    "file_creation_or_deletion",
    "not_unified_diff",
    "unsupported_no_opinion",
]


_OUTCOME_PRECEDENCE: tuple[DiffVerifierReason, ...] = (
    "malformed_hunk_header",
    "hunk_body_count_mismatch",
    "file_missing",
    "content_mismatch",
    "fuzzy_below_threshold",
    "file_creation_or_deletion",
    "unsupported_no_opinion",
    "not_unified_diff",
    "clean",
)


@dataclass(frozen=True)
class HunkMismatch:
    """Structured mismatch record.

    ``file`` is relative to ``repo_dir``. ``hunk_index`` is zero-indexed
    across all hunks in the diff. ``expected`` is the old-side body from
    the patch; ``actual`` is what the repo contains at that old range.
    ``kind`` stays restricted to the content/file classes consumed by
    repair-mode feedback.
    """

    file: str
    hunk_index: int
    old_start: int
    old_count: int
    expected: list[str]
    actual: list[str]
    kind: Literal["file_missing", "content_mismatch", "fuzzy_below_threshold"]
    match_ratio: float


@dataclass(frozen=True)
class DiffVerifierReasonEvent:
    """One reason event emitted while diagnosing a diff."""

    reason: DiffVerifierReason
    scope: Literal["patch", "file", "hunk"]
    file: str | None = None
    hunk_index: int | None = None
    old_start: int | None = None
    old_count: int | None = None
    new_start: int | None = None
    new_count: int | None = None
    message: str = ""


@dataclass(frozen=True)
class DiffVerifierResult:
    """Full verifier diagnosis for one emitted patch."""

    mismatches: list[HunkMismatch]
    reason_events: list[DiffVerifierReasonEvent]
    outcome: DiffVerifierReason

    @property
    def reasons(self) -> list[DiffVerifierReason]:
        return [event.reason for event in self.reason_events]


@dataclass(frozen=True)
class _ParsedHunk:
    old_file: str | None
    new_file: str | None
    hunk_index: int
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    old_body: list[str]
    new_body: list[str]


@dataclass
class _OpenHunk:
    old_file: str | None
    new_file: str | None
    hunk_index: int
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    old_body: list[str]
    new_body: list[str]


@dataclass(frozen=True)
class _ParsedDiff:
    items: list[DiffVerifierReasonEvent | _ParsedHunk]
    saw_unified_marker: bool


def verify_diff_context(
    diff: str,
    repo_dir: Path,
    fuzzy_threshold: float = _FUZZY_THRESHOLD,
    *,
    run_frame_builder: Any | None = None,
    run_id: str | None = None,
    node_run_id: str | None = None,
    event_seq: int | None = None,
    source_id: str = "swebench:diff_verifier",
) -> list[HunkMismatch]:
    """Return one ``HunkMismatch`` per problematic hunk.

    Compatibility wrapper. Use ``verify_diff_context_with_reasons()``
    when callers need to distinguish clean patches from unsupported or
    no-op verifier outcomes.
    """
    return verify_diff_context_with_reasons(
        diff,
        repo_dir,
        fuzzy_threshold,
        run_frame_builder=run_frame_builder,
        run_id=run_id,
        node_run_id=node_run_id,
        event_seq=event_seq,
        source_id=source_id,
    ).mismatches


def verify_diff_context_with_reasons(
    diff: str,
    repo_dir: Path,
    fuzzy_threshold: float = _FUZZY_THRESHOLD,
    *,
    run_frame_builder: Any | None = None,
    run_id: str | None = None,
    node_run_id: str | None = None,
    event_seq: int | None = None,
    source_id: str = "swebench:diff_verifier",
) -> DiffVerifierResult:
    """Return mismatch records plus reason/outcome telemetry for one diff."""
    parsed = _parse_diff_items(diff)

    mismatches: list[HunkMismatch] = []
    reason_events: list[DiffVerifierReasonEvent] = []

    for item in parsed.items:
        if isinstance(item, DiffVerifierReasonEvent):
            reason_events.append(item)
            continue

        if (
            len(item.old_body) != item.old_count
            or len(item.new_body) != item.new_count
        ):
            reason_events.append(
                _reason_event(
                    "hunk_body_count_mismatch",
                    "hunk",
                    file=_event_file(item.old_file, item.new_file),
                    hunk_index=item.hunk_index,
                    old_start=item.old_start,
                    old_count=item.old_count,
                    new_start=item.new_start,
                    new_count=item.new_count,
                    message=(
                        "hunk body line counts do not match old/new ranges"
                    ),
                )
            )
            continue

        # File creations (--- /dev/null) have no old-side bytes to
        # verify. The file-level telemetry was emitted by the parser.
        file_path = item.old_file
        if file_path is None:
            continue

        abs_path = (repo_dir / file_path).resolve()
        if not abs_path.is_file():
            mismatch = HunkMismatch(
                file=file_path,
                hunk_index=item.hunk_index,
                old_start=item.old_start,
                old_count=item.old_count,
                expected=item.old_body,
                actual=[],
                kind="file_missing",
                match_ratio=0.0,
            )
            mismatches.append(mismatch)
            reason_events.append(_event_from_mismatch(mismatch))
            continue

        try:
            file_text = abs_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            reason_events.append(
                _reason_event(
                    "unsupported_no_opinion",
                    "hunk",
                    file=file_path,
                    hunk_index=item.hunk_index,
                    old_start=item.old_start,
                    old_count=item.old_count,
                    new_start=item.new_start,
                    new_count=item.new_count,
                    message="could not read old-side file",
                )
            )
            continue

        file_lines = file_text.splitlines()
        end_idx = item.old_start - 1 + item.old_count
        actual_body = file_lines[item.old_start - 1 : end_idx]

        if actual_body == item.old_body:
            reason_events.append(_clean_event(item, file_path))
            continue

        expected_joined = "\n".join(item.old_body)
        actual_joined = "\n".join(actual_body)
        ratio = difflib.SequenceMatcher(
            None, expected_joined, actual_joined
        ).ratio()

        stripped_expected = [ln.strip() for ln in item.old_body]
        stripped_actual = [ln.strip() for ln in actual_body]
        stripped_equal = (
            len(stripped_expected) == len(stripped_actual)
            and stripped_expected == stripped_actual
        )

        kind: Literal[
            "file_missing", "content_mismatch", "fuzzy_below_threshold"
        ]
        if stripped_equal:
            if ratio >= fuzzy_threshold:
                reason_events.append(_clean_event(item, file_path))
                continue
            kind = "fuzzy_below_threshold"
        else:
            kind = "content_mismatch"

        mismatch = HunkMismatch(
            file=file_path,
            hunk_index=item.hunk_index,
            old_start=item.old_start,
            old_count=item.old_count,
            expected=item.old_body,
            actual=actual_body,
            kind=kind,
            match_ratio=ratio,
        )
        mismatches.append(mismatch)
        reason_events.append(_event_from_mismatch(mismatch))

    if not reason_events:
        reason_events.append(
            _reason_event(
                "unsupported_no_opinion"
                if parsed.saw_unified_marker
                else "not_unified_diff",
                "patch",
                message=(
                    "unified-like input had no verifiable hunks"
                    if parsed.saw_unified_marker
                    else "input contains no unified diff structure"
                ),
            )
        )

    result = DiffVerifierResult(
        mismatches=mismatches,
        reason_events=reason_events,
        outcome=_roll_up_outcome(reason_events),
    )
    _record_diff_delta(
        diff=diff,
        result=result,
        run_frame_builder=run_frame_builder,
        run_id=run_id,
        node_run_id=node_run_id,
        event_seq=event_seq,
        source_id=source_id,
    )
    return result


def _record_diff_delta(
    *,
    diff: str,
    result: DiffVerifierResult,
    run_frame_builder: Any | None,
    run_id: str | None,
    node_run_id: str | None,
    event_seq: int | None,
    source_id: str,
) -> None:
    from sage.runtime.oracle.env import oracle_enabled
    if not oracle_enabled() or run_frame_builder is None:
        return
    from sage.runtime.evidence.producers.diff import produce_diff_verifier_deltas

    patch_hash = hashlib.sha256(diff.encode("utf-8", errors="replace")).hexdigest()
    producer_result = produce_diff_verifier_deltas(
        run_id=run_id or run_frame_builder.run_id,
        node_run_id=node_run_id,
        event_seq=event_seq,
        source_id=source_id,
        verify_result={
            "outcome": result.outcome,
            "mismatches": [{"kind": mismatch.kind} for mismatch in result.mismatches],
            "reasons": list(result.reasons),
            "patch_hash": patch_hash,
        },
    )
    for delta in producer_result.deltas:
        run_frame_builder.record_delta(delta)


# ---------------------------------------------------------------------------
# Parser - minimal stdlib.
# ---------------------------------------------------------------------------


def _parse_diff_items(diff: str) -> _ParsedDiff:
    """Parse unified-diff markers into ordered hunk/event items."""
    items: list[DiffVerifierReasonEvent | _ParsedHunk] = []
    current_old_file: str | None = None
    current_new_file: str | None = None
    saw_old_header = False
    have_file_header = False
    saw_unified_marker = False
    hunk_counter = 0
    open_hunk: _OpenHunk | None = None

    def _flush_hunk() -> None:
        nonlocal open_hunk
        if open_hunk is None:
            return
        items.append(
            _ParsedHunk(
                old_file=open_hunk.old_file,
                new_file=open_hunk.new_file,
                hunk_index=open_hunk.hunk_index,
                old_start=open_hunk.old_start,
                old_count=open_hunk.old_count,
                new_start=open_hunk.new_start,
                new_count=open_hunk.new_count,
                old_body=open_hunk.old_body,
                new_body=open_hunk.new_body,
            )
        )
        open_hunk = None

    for raw_line in diff.splitlines():
        if raw_line.startswith("diff --git "):
            _flush_hunk()
            current_old_file = None
            current_new_file = None
            saw_old_header = False
            have_file_header = False
            saw_unified_marker = True
            continue

        if raw_line.startswith("--- "):
            _flush_hunk()
            current_old_file = _normalise_diff_path(raw_line[4:])
            current_new_file = None
            saw_old_header = True
            have_file_header = False
            saw_unified_marker = True
            continue

        if (
            raw_line.startswith("+++ ")
            and open_hunk is None
            and saw_old_header
        ):
            current_new_file = _normalise_diff_path(raw_line[4:])
            have_file_header = True
            saw_unified_marker = True
            if current_old_file is None or current_new_file is None:
                items.append(
                    _reason_event(
                        "file_creation_or_deletion",
                        "file",
                        file=_event_file(current_old_file, current_new_file),
                        message="/dev/null file section",
                    )
                )
            continue

        if raw_line.startswith("@@"):
            _flush_hunk()
            saw_unified_marker = True
            hunk_index = hunk_counter
            hunk_counter += 1
            m = _HUNK_HEADER_RE.match(raw_line)
            if not m:
                items.append(
                    _reason_event(
                        "malformed_hunk_header",
                        "hunk",
                        file=_event_file(current_old_file, current_new_file),
                        hunk_index=hunk_index,
                        message=f"malformed hunk header: {raw_line}",
                    )
                )
                continue
            old_start = int(m.group("old_start"))
            old_count = _count_or_one(m.group("old_count"))
            new_start = int(m.group("new_start"))
            new_count = _count_or_one(m.group("new_count"))
            if not have_file_header:
                items.append(
                    _reason_event(
                        "unsupported_no_opinion",
                        "hunk",
                        hunk_index=hunk_index,
                        old_start=old_start,
                        old_count=old_count,
                        new_start=new_start,
                        new_count=new_count,
                        message="hunk has no complete file header",
                    )
                )
                continue
            open_hunk = _OpenHunk(
                old_file=current_old_file,
                new_file=current_new_file,
                hunk_index=hunk_index,
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
                old_body=[],
                new_body=[],
            )
            continue

        if open_hunk is None:
            continue

        if raw_line.startswith("\\"):
            # "\ No newline at end of file" is not a body line.
            continue
        if raw_line.startswith(" "):
            open_hunk.old_body.append(raw_line[1:])
            open_hunk.new_body.append(raw_line[1:])
        elif raw_line.startswith("-"):
            open_hunk.old_body.append(raw_line[1:])
        elif raw_line.startswith("+"):
            open_hunk.new_body.append(raw_line[1:])
        else:
            # Defensive compatibility with the old parser: treat an
            # unprefixed blank/body line as context instead of crashing.
            open_hunk.old_body.append(raw_line)
            open_hunk.new_body.append(raw_line)

    _flush_hunk()
    return _ParsedDiff(items=items, saw_unified_marker=saw_unified_marker)


def _count_or_one(value: str | None) -> int:
    return int(value) if value is not None else 1


def _normalise_diff_path(raw_path: str) -> str | None:
    path = raw_path.strip()
    if "\t" in path:
        path = path.split("\t", 1)[0]
    if path == "/dev/null":
        return None
    if path.startswith(("a/", "b/")):
        return path[2:]
    return path


def _event_file(old_file: str | None, new_file: str | None) -> str | None:
    return old_file if old_file is not None else new_file


def _reason_event(
    reason: DiffVerifierReason,
    scope: Literal["patch", "file", "hunk"],
    *,
    file: str | None = None,
    hunk_index: int | None = None,
    old_start: int | None = None,
    old_count: int | None = None,
    new_start: int | None = None,
    new_count: int | None = None,
    message: str = "",
) -> DiffVerifierReasonEvent:
    return DiffVerifierReasonEvent(
        reason=reason,
        scope=scope,
        file=file,
        hunk_index=hunk_index,
        old_start=old_start,
        old_count=old_count,
        new_start=new_start,
        new_count=new_count,
        message=message,
    )


def _clean_event(item: _ParsedHunk, file_path: str) -> DiffVerifierReasonEvent:
    return _reason_event(
        "clean",
        "hunk",
        file=file_path,
        hunk_index=item.hunk_index,
        old_start=item.old_start,
        old_count=item.old_count,
        new_start=item.new_start,
        new_count=item.new_count,
    )


def _event_from_mismatch(mismatch: HunkMismatch) -> DiffVerifierReasonEvent:
    return _reason_event(
        mismatch.kind,
        "hunk",
        file=mismatch.file,
        hunk_index=mismatch.hunk_index,
        old_start=mismatch.old_start,
        old_count=mismatch.old_count,
    )


def _roll_up_outcome(
    reason_events: list[DiffVerifierReasonEvent],
) -> DiffVerifierReason:
    reasons = {event.reason for event in reason_events}
    for reason in _OUTCOME_PRECEDENCE:
        if reason in reasons:
            return reason
    return "unsupported_no_opinion"


# ---------------------------------------------------------------------------
# A3 (2026-04-24) - LLM one-shot repair with verifier-mismatch feedback
# ---------------------------------------------------------------------------


_VERIFIER_REPAIR_PROMPT = """\
The unified-diff patch below had hunks whose context/removed lines \
don't match the actual file contents at the claimed positions. \
Realign the patch so every context line matches the real bytes in \
the repository. Preserve the semantic change the patch was trying \
to make.

## Original issue

{problem_statement}

## Patch that has content mismatches

```diff
{broken_patch}
```

## Per-hunk mismatch diagnostic

The verifier read the repo file at each hunk's claimed `old_start` \
position and compared the context+removed lines. Lines below labelled \
EXPECTED are what the patch claims is in the file; ACTUAL is what is \
really there. Fix the patch so the two match.

{mismatch_report}

Output ONLY a corrected unified diff inside a single ```diff fenced \
block. Use the real file contents you were just shown. No prose, no \
explanation - just the diff.\
"""


def _format_mismatch_report(mismatches: list[HunkMismatch]) -> str:
    """Render mismatch diagnostics for the repair prompt."""
    _MAX_LINES_PER_SECTION = 20
    _MAX_LINE_CHARS = 200

    def _trim(lines: list[str]) -> list[str]:
        out = [ln[:_MAX_LINE_CHARS] for ln in lines[:_MAX_LINES_PER_SECTION]]
        if len(lines) > _MAX_LINES_PER_SECTION:
            out.append(
                f"  ... ({len(lines) - _MAX_LINES_PER_SECTION} more "
                "lines omitted)"
            )
        return out

    sections: list[str] = []
    for m in mismatches:
        header = (
            f"### Hunk {m.hunk_index} - {m.file} "
            f"@@ -{m.old_start},{m.old_count} (kind={m.kind}, "
            f"ratio={m.match_ratio:.2f})"
        )
        if m.kind == "file_missing":
            sections.append(
                header
                + "\n  File does not exist at the claimed path. "
                "Either the path is wrong, or the repository layout "
                "differs from what the agent assumed."
            )
            continue

        exp_block = "\n".join(f"    {ln}" for ln in _trim(m.expected))
        act_block = "\n".join(f"    {ln}" for ln in _trim(m.actual))
        sections.append(
            f"{header}\n"
            f"  EXPECTED (from the patch):\n{exp_block}\n"
            f"  ACTUAL (from the repo):\n{act_block}"
        )

    return "\n\n".join(sections) if sections else "(no mismatches)"


async def repair_with_verifier_feedback(
    llm: Any,
    problem_statement: str,
    broken_patch: str,
    mismatches: list[HunkMismatch],
    instance_id: str = "",
    timeout: float = 60.0,
    repair_budget_usd: float | None = None,
) -> tuple[str, str]:
    """One-shot LLM repair using mismatch diagnostic as feedback.

    Returns ``(new_patch, stage)`` where ``stage`` is one of:

    * ``"verifier_repair"`` - LLM returned a non-empty corrected diff.
    * ``"verifier_repair_empty"`` - LLM returned nothing extractable.
    * ``"verifier_repair_skipped"`` - no mismatches, no llm handle,
      or repair budget exhausted (timeout=0 or repair_budget_usd=0).

    ``repair_budget_usd`` is an explicit budget cap for the repair call.
    If 0 (or timeout=0, which is the caller's signal for "no budget"),
    repair is skipped entirely and ``"verifier_repair_skipped"`` is
    returned without any LLM call.  This ensures budget-exhausted skips
    are distinguishable from timeout failures in post-hoc telemetry.
    """
    if not mismatches or llm is None:
        return broken_patch, "verifier_repair_skipped"

    # Budget-exhausted guard: caller signals "don't spend on repair"
    # by passing repair_budget_usd=0 or timeout=0.
    if repair_budget_usd == 0 or timeout == 0:
        return broken_patch, "verifier_repair_skipped"

    try:
        from sage.bench.swebench_bench import _extract_patch
    except ImportError:

        def _extract_patch(s: str) -> str:  # type: ignore[misc]
            return s

    from sage.llm.base import Message, Role

    prompt = _VERIFIER_REPAIR_PROMPT.format(
        problem_statement=(problem_statement or "")[:2000],
        broken_patch=broken_patch,
        mismatch_report=_format_mismatch_report(mismatches),
    )

    import asyncio
    import logging

    log = logging.getLogger(__name__)

    try:
        response = await asyncio.wait_for(
            llm.generate(messages=[Message(role=Role.USER, content=prompt)]),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        log.warning(
            "[%s] verifier-repair LLM timed out after %.0fs",
            instance_id or "?",
            timeout,
        )
        return broken_patch, "verifier_repair_empty"
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[%s] verifier-repair LLM call failed: %s",
            instance_id or "?",
            exc,
        )
        return broken_patch, "verifier_repair_empty"

    extracted = _extract_patch(getattr(response, "content", None) or "")
    if not extracted.strip():
        return broken_patch, "verifier_repair_empty"

    return extracted, "verifier_repair"
