from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Any

from sage.runtime.evidence.delta import (
    RUNTIME_DELTA_SCHEMA_VERSION,
    DeltaPolarity,
    DeltaProducerResult,
    RuntimeDelta,
    _DELTA_KIND_TABLE,
    _POLARITY_RULES,
)
from sage.runtime.evidence.errors import EvidenceError
from sage.runtime.evidence.payloads import PAYLOAD_ALLOWED_KEYS


ALLOWED_DELTA_KINDS = _DELTA_KIND_TABLE["test_parser"]
ALLOWED_PAYLOAD_KEYS = PAYLOAD_ALLOWED_KEYS["test_parser"]
POLARITY_RULES = _POLARITY_RULES["test_parser"]

_COUNT_RE = re.compile(
    r"(?P<count>\d+)\s+"
    r"(?P<kind>passed|pass|failed|failures?|errors?|skipped|skip)",
    re.IGNORECASE,
)
_PYTEST_SUMMARY_RE = re.compile(
    r"^=+\s+"
    r"(?P<body>.*?)"
    r"\s+in\s+(?P<seconds>\d+(?:\.\d+)?)s"
    r"\s+=+$",
    re.IGNORECASE,
)
_UNITTEST_RAN_RE = re.compile(
    r"\bRan\s+(?P<tests>\d+)\s+tests?\s+in\s+(?P<seconds>\d+(?:\.\d+)?)s\b"
)


def parse_pytest_output(stdout: str, stderr: str, exit_code: int) -> dict[str, Any]:
    """Parse pytest's delimited summary tail line without retaining raw output."""
    del exit_code
    for line in _tail_lines(stdout, stderr):
        summary = _PYTEST_SUMMARY_RE.fullmatch(line)
        if summary is None:
            continue
        counts = _parse_count_tokens(summary.group("body"))
        if not counts:
            continue
        duration_ms = float(summary.group("seconds")) * 1000.0
        return {**counts, "duration_ms": duration_ms}
    return {"parse_failed": True}


def parse_unittest_output(stdout: str, stderr: str, exit_code: int) -> dict[str, Any]:
    """Parse stdlib unittest's structured tail summary."""
    del exit_code
    tail = "\n".join(_tail_lines(stdout, stderr))
    ran = _UNITTEST_RAN_RE.search(tail)
    if ran is None:
        return {"parse_failed": True}
    total = int(ran.group("tests"))
    duration_ms = float(ran.group("seconds")) * 1000.0
    failed = 0
    errors = 0
    skipped = 0
    failed_match = re.search(r"FAILED\s+\((?P<body>[^)]*)\)", tail)
    if failed_match is not None:
        body = failed_match.group("body")
        for key, value in re.findall(r"(failures|errors|skipped)=(\d+)", body):
            if key == "failures":
                failed = int(value)
            elif key == "errors":
                errors = int(value)
            elif key == "skipped":
                skipped = int(value)
    passed = max(0, total - failed - errors - skipped)
    return {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "errors": errors,
        "duration_ms": duration_ms,
    }


def parse_junit_output(stdout: str, stderr: str, exit_code: int) -> dict[str, Any]:
    """Parse a JUnit XML block or file content string."""
    del exit_code
    xml_text = (stdout or stderr or "").strip()
    if not xml_text:
        return {"parse_failed": True}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {"parse_failed": True}

    suites = [root] if root.tag == "testsuite" else list(root.findall(".//testsuite"))
    if not suites:
        return {"parse_failed": True}

    tests = failures = errors = skipped = 0
    duration_sec = 0.0
    for suite in suites:
        tests += _int_attr(suite, "tests")
        failures += _int_attr(suite, "failures")
        errors += _int_attr(suite, "errors")
        skipped += _int_attr(suite, "skipped")
        duration_sec += _float_attr(suite, "time")
    passed = max(0, tests - failures - errors - skipped)
    return {
        "passed": passed,
        "failed": failures,
        "skipped": skipped,
        "errors": errors,
        "duration_ms": duration_sec * 1000.0,
    }


def produce_test_parser_deltas(
    *,
    run_id: str,
    node_run_id: str | None,
    event_seq: int | None,
    source_id: str,
    framework: str,
    stdout: str = "",
    stderr: str = "",
    exit_code: int = 0,
    suite_id: str = "default",
    junit_xml: str | None = None,
) -> DeltaProducerResult:
    try:
        framework_name = framework.lower().strip()
        if framework_name == "pytest":
            parser_id = "pytest_summary_v1"
            parsed = parse_pytest_output(stdout, stderr, exit_code)
        elif framework_name == "junit":
            parser_id = "junit_xml_v0"
            parsed = parse_junit_output(junit_xml or stdout, stderr, exit_code)
        elif framework_name == "unittest":
            parser_id = "unittest_summary_v0"
            parsed = parse_unittest_output(stdout, stderr, exit_code)
        else:
            return DeltaProducerResult(
                deltas=(),
                rejected_reason=f"unknown test parser framework: {framework!r}",
            )

        payload: dict[str, Any] = {
            "framework": framework_name,
            "parser_id": parser_id,
            "suite_id": suite_id,
        }
        if parsed.get("parse_failed"):
            delta_kind = "parse_failed"
            polarity: DeltaPolarity = "unknown"
        else:
            passed = int(parsed.get("passed", 0))
            failed = int(parsed.get("failed", 0))
            skipped = int(parsed.get("skipped", 0))
            errors = int(parsed.get("errors", 0))
            payload.update(
                {
                    "passed_count": passed,
                    "failed_count": failed,
                    "skipped_count": skipped,
                    "error_count": errors,
                    "duration_ms": float(parsed.get("duration_ms", 0.0)),
                }
            )
            if failed == 0 and errors == 0 and passed > 0:
                delta_kind = "tests_passed"
                polarity = "positive"
            elif passed > 0 and (failed > 0 or errors > 0):
                delta_kind = "tests_partial"
                polarity = "negative"
            elif failed > 0 or errors > 0:
                delta_kind = "tests_failed"
                polarity = "negative"
            else:
                delta_kind = "parse_failed"
                polarity = "unknown"
                payload = {
                    "framework": framework_name,
                    "parser_id": parser_id,
                    "suite_id": suite_id,
                }

        return DeltaProducerResult(
            deltas=(
                RuntimeDelta(
                    schema_version=RUNTIME_DELTA_SCHEMA_VERSION,
                    producer="test_parser",
                    delta_kind=delta_kind,
                    polarity=polarity,
                    confidence=1.0,
                    run_id=run_id,
                    node_run_id=node_run_id,
                    event_seq=event_seq,
                    source_id=source_id,
                    payload=payload,
                ),
            )
        )
    except (EvidenceError, TypeError, ValueError) as exc:
        return DeltaProducerResult(deltas=(), rejected_reason=str(exc))


def _tail_lines(stdout: str, stderr: str) -> list[str]:
    lines = (stdout + "\n" + stderr).splitlines()
    return [line.strip() for line in lines[-20:] if line.strip()][::-1]


def _parse_count_tokens(line: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for match in _COUNT_RE.finditer(line):
        count = int(match.group("count"))
        kind = match.group("kind").lower()
        if kind.startswith("pass"):
            counts["passed"] += count
        elif kind.startswith("fail"):
            counts["failed"] += count
        elif kind.startswith("error"):
            counts["errors"] += count
        elif kind.startswith("skip"):
            counts["skipped"] += count
    return counts if any(counts.values()) else {}


def _int_attr(node: ET.Element, key: str) -> int:
    try:
        return int(node.attrib.get(key, "0"))
    except ValueError:
        return 0


def _float_attr(node: ET.Element, key: str) -> float:
    try:
        return float(node.attrib.get(key, "0"))
    except ValueError:
        return 0.0
