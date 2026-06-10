"""Normalize SWE-bench Pro grading outputs into machine-readable verdicts.

RESOLUTION_UNBLOCKERS criteria 5-6 (cgpro post-run 2026-06-10, conv
``cgpro_b2_unblockers_verify``). The official grader's per-instance
artifacts are lossy in exactly the places a repair loop needs:

- ``output.json`` can collapse a build break to the opaque
  ``NO_TESTS_FOUND_OR_PARSING_ERROR`` bucket while the stdout log carries
  the causal line (teleport ``[build failed]``, 2026-06-10 graded N=5);
- the first compiler/test diagnostic (e.g. ``error TS2551 ... Did you
  mean '_message'?``) is repair-exploitable but lives only in stdout;
- a local result-write crash (NodeBB cp1252/emoji) leaves NO machine
  artifact at all.

This parser walks the grading dir and emits ONE normalized verdict per
prediction:

    {
      "<instance_id>": {
        "resolved": bool,
        "verdict": RESOLVED | EMPTY_PATCH | PATCH_APPLY_FAILED |
                   BUILD_FAILED | TEST_FAILED |
                   NO_TESTS_FOUND_OR_PARSING_ERROR |
                   GRADER_OUTPUT_WRITE_FAILED,
        "first_compiler_error": str | None,   # ADVISORY (repair feedback)
        "first_test_error": str | None,       # ADVISORY (repair feedback)
        "output_json_present": bool,
        "stdout_log_present": bool,
      }, ...
    }

The two ``first_*`` fields are ADVISORY: they feed repair prompts and
human triage, never gates. Verdict precedence (most causal wins):

    EMPTY_PATCH > PATCH_APPLY_FAILED > BUILD_FAILED >
    TEST_FAILED (tests ran: output.json has results) >
    NO_TESTS_FOUND_OR_PARSING_ERROR (opaque bucket, nothing better known) >
    GRADER_OUTPUT_WRITE_FAILED (no machine artifact at all)

Run the grader itself with ``PYTHONIOENCODING=utf-8`` on Windows — the
upstream eval script writes logs without an explicit encoding and dies
on emoji under cp1252 (the NodeBB incident this taxonomy's last bucket
exists for).

Usage:
    python sage-python/scripts/swebench_pro_post_grader_parse.py \
        --grading-dir docs/benchmarks/<bundle>/grading \
        --predictions docs/benchmarks/<bundle>/run/predictions.json \
        --eval-results docs/benchmarks/<bundle>/grading/eval_results.json \
        --output docs/benchmarks/<bundle>/grading/graded_verdicts.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("sage.bench.swebench_pro_post_grader_parse")

# First-diagnostic patterns, multi-language, ordered by specificity.
# ADVISORY extraction for repair feedback — heuristic by design (cgpro
# flagged the fragility; missing languages degrade to None, never crash).
_COMPILER_ERROR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"error TS\d+:.*"),                       # TypeScript
    re.compile(r"error\[E\d+\]:.*"),                     # Rust
    re.compile(r"FAIL\s+\S+\s+\[build failed\]"),        # Go test build
    re.compile(r"^.+\.go:\d+:\d+:\s+.*", re.MULTILINE),  # Go compiler line
    re.compile(r"SyntaxError:.*"),                       # Python
    re.compile(r"ImportError:.*|ModuleNotFoundError:.*"),
    re.compile(r"make.*\*\*\*.*Error \d+"),              # make
    re.compile(r"`make` failed with exit code: \d+"),    # node-gyp style
    re.compile(r"error: cannot find symbol.*"),          # Java
)

_TEST_ERROR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"Test failed:.*"),
    re.compile(r"AssertionError.*"),
    re.compile(r"^\s*\d+ failing", re.MULTILINE),
    re.compile(r"^FAILED .*", re.MULTILINE),
    re.compile(r"^\s*✗ .*", re.MULTILINE),
    re.compile(r"FAIL[: ].*"),
)

_APPLY_FAILURE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"error: patch failed:.*"),
    re.compile(r"error: .*: patch does not apply"),
    re.compile(r"Hunk #\d+ FAILED.*"),
    re.compile(r"can't find file to patch.*"),
)

_BUILD_FAILURE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\[build failed\]"),
    re.compile(r"error TS\d+:"),
    re.compile(r"error\[E\d+\]:"),
    re.compile(r"`make` failed with exit code: \d+"),
    re.compile(r"make.*\*\*\*.*Error \d+"),
    re.compile(r"Compilation failed"),
    re.compile(r"error: cannot find symbol"),
)

_NO_TESTS_BUCKET = "NO_TESTS_FOUND_OR_PARSING_ERROR"


def _first_match(text: str, patterns: tuple[re.Pattern[str], ...]) -> str | None:
    """Earliest match in the TEXT (by position), trying every pattern."""
    best: tuple[int, str] | None = None
    for pattern in patterns:
        m = pattern.search(text)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), m.group(0).strip())
    return best[1] if best else None


def _read_text_lossy(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _load_output_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _classify(
    *,
    patch: str,
    stdout_text: str,
    output_data: dict[str, Any] | None,
    resolved: bool,
) -> str:
    if resolved:
        return "RESOLVED"
    if not patch:
        return "EMPTY_PATCH"
    if _first_match(stdout_text, _APPLY_FAILURE_PATTERNS):
        return "PATCH_APPLY_FAILED"

    tests = (output_data or {}).get("tests")
    tests_list = tests if isinstance(tests, list) else []
    test_names = {
        str(t.get("name", "")) for t in tests_list if isinstance(t, dict)
    }
    # Opaque-bucket markers as OBSERVED in real grader output (2026-06-10
    # Phase 2.a bundle) — exact phrases, NOT substrings: a legitimate test
    # literally named 'error_handling_test' must count as a real result
    # (review MAJOR 2026-06-10).
    opaque_only = bool(test_names) and all(
        "test/unknown" in name
        or name == _NO_TESTS_BUCKET
        or name.startswith("Build/Runtime Error")
        for name in test_names
    )
    real_results = bool(test_names) and not opaque_only

    if _first_match(stdout_text, _BUILD_FAILURE_PATTERNS):
        # The causal build break beats the opaque bucket (teleport class)
        # — UNLESS tests demonstrably ran (real per-test results), in
        # which case the break happened in the test phase: TEST_FAILED.
        return "TEST_FAILED" if real_results else "BUILD_FAILED"
    if real_results:
        return "TEST_FAILED"
    if output_data is not None:
        return _NO_TESTS_BUCKET
    return "GRADER_OUTPUT_WRITE_FAILED"


def _instance_artifacts(instance_dir: Path) -> tuple[Path | None, Path | None]:
    """Locate the ``*_stdout.log`` and ``*_output.json`` regardless of the
    run prefix the grader used."""
    stdout_path = next(iter(sorted(instance_dir.glob("*_stdout.log"))), None)
    output_path = next(iter(sorted(instance_dir.glob("*_output.json"))), None)
    return stdout_path, output_path


def build_verdicts(
    *,
    grading_dir: Path,
    predictions: list[dict[str, Any]],
    eval_results: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    verdicts: dict[str, dict[str, Any]] = {}
    for record in predictions:
        instance_id = str(record.get("instance_id") or "")
        if not instance_id:
            continue
        patch = str(record.get("patch") or "")
        resolved = bool(eval_results.get(instance_id, False))
        instance_dir = grading_dir / instance_id
        stdout_path, output_path = (
            _instance_artifacts(instance_dir)
            if instance_dir.is_dir()
            else (None, None)
        )
        stdout_text = _read_text_lossy(stdout_path) if stdout_path else ""
        output_data = _load_output_json(output_path) if output_path else None

        verdict = _classify(
            patch=patch,
            stdout_text=stdout_text,
            output_data=output_data,
            resolved=resolved,
        )
        verdicts[instance_id] = {
            "resolved": resolved,
            "verdict": verdict,
            "first_compiler_error": _first_match(
                stdout_text, _COMPILER_ERROR_PATTERNS
            ),
            "first_test_error": _first_match(stdout_text, _TEST_ERROR_PATTERNS),
            "output_json_present": output_path is not None,
            "stdout_log_present": stdout_path is not None,
        }
    return verdicts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grading-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--eval-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        log.error("predictions must be a JSON list (Pro shape)")
        return 2
    try:
        eval_results = json.loads(args.eval_results.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.warning("eval_results unreadable — treating all as unresolved")
        eval_results = {}

    verdicts = build_verdicts(
        grading_dir=args.grading_dir,
        predictions=predictions,
        eval_results=eval_results if isinstance(eval_results, dict) else {},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(verdicts, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    by_verdict: dict[str, int] = {}
    for v in verdicts.values():
        by_verdict[v["verdict"]] = by_verdict.get(v["verdict"], 0) + 1
    log.info(
        "Wrote %d verdicts to %s (%s)",
        len(verdicts),
        args.output,
        ", ".join(f"{k}={n}" for k, n in sorted(by_verdict.items())),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
