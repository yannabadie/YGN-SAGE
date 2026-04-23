"""Tests for the SWE-bench generation-phase log-to-file default.

Covers `_setup_bench_file_log` — the helper that attaches a FileHandler to
the root logger when `args.type == "swebench"` so the gen-phase log is
captured to disk next to the bench artifacts by default.

Motivated by the 2026-04-22 parity smoke + 2026-04-23 Track 3.1b
investigation where the gen log was unavailable for post-hoc semantic-miss
tracers because only stderr captured it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import sage.bench.__main__ as bench_main


@pytest.fixture(autouse=True)
def _reset_root_logger_file_handlers():
    """Remove handlers the bench helper added during the test.

    The bench helper attaches a FileHandler to `logging.getLogger()` and
    marks the logger with a sentinel attribute `_sage_bench_file_handler`
    to avoid double-attachment. The root logger is a singleton shared
    across tests in the same process, so we clean up per-test or tests
    interfere — test 3/4 would be short-circuited by a sentinel set in
    test 1.

    We also snapshot the pre-test handler list so `_find_file_handlers`
    can ignore unrelated FileHandlers attached by plugins (e.g. logfire
    attaches a `\\.\nul` FileHandler on Windows at import time).
    """
    root = logging.getLogger()
    before = list(root.handlers)
    before_snapshot = set(id(h) for h in before)
    # Expose the baseline to the test via the root logger itself so the
    # helper function can filter it out.
    root._test_baseline_handlers = before_snapshot  # type: ignore[attr-defined]
    try:
        yield
    finally:
        for h in list(root.handlers):
            if id(h) not in before_snapshot and isinstance(h, logging.FileHandler):
                try:
                    h.close()
                finally:
                    root.removeHandler(h)
        if hasattr(root, "_sage_bench_file_handler"):
            delattr(root, "_sage_bench_file_handler")
        if hasattr(root, "_test_baseline_handlers"):
            delattr(root, "_test_baseline_handlers")


def _find_file_handlers() -> list[logging.FileHandler]:
    """Return FileHandlers added DURING the current test.

    Excludes any FileHandler that was on the root logger when the
    autouse fixture started — this filters out plugin-attached handlers
    (e.g. logfire's `\\.\nul` on Windows) that we shouldn't be asserting
    about.
    """
    root = logging.getLogger()
    baseline = getattr(root, "_test_baseline_handlers", set())
    return [
        h
        for h in root.handlers
        if isinstance(h, logging.FileHandler) and id(h) not in baseline
    ]


def test_handler_added_for_swebench(tmp_path, monkeypatch):
    monkeypatch.delenv("SAGE_BENCH_LOG_FILE", raising=False)
    out_json = tmp_path / "x.json"
    args = SimpleNamespace(type="swebench", output=str(out_json))

    log_path = bench_main._setup_bench_file_log(args)

    assert log_path is not None
    expected = tmp_path / "x-gen.log"
    assert Path(log_path).resolve() == expected.resolve()

    handlers = _find_file_handlers()
    assert len(handlers) == 1
    assert Path(handlers[0].baseFilename).resolve() == expected.resolve()


def test_handler_not_added_for_other_types(tmp_path, monkeypatch):
    monkeypatch.delenv("SAGE_BENCH_LOG_FILE", raising=False)
    out_json = tmp_path / "x.json"
    args = SimpleNamespace(type="bigcodebench", output=str(out_json))

    log_path = bench_main._setup_bench_file_log(args)

    assert log_path is None
    assert _find_file_handlers() == []


def test_opt_out_via_env_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("SAGE_BENCH_LOG_FILE", "")
    args = SimpleNamespace(type="swebench", output=str(tmp_path / "x.json"))

    log_path = bench_main._setup_bench_file_log(args)

    assert log_path is None
    assert _find_file_handlers() == []


def test_opt_out_via_env_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("SAGE_BENCH_LOG_FILE", "0")
    args = SimpleNamespace(type="swebench", output=str(tmp_path / "x.json"))

    log_path = bench_main._setup_bench_file_log(args)

    assert log_path is None
    assert _find_file_handlers() == []


def test_custom_path_via_env(tmp_path, monkeypatch):
    custom = tmp_path / "subdir" / "custom.log"
    monkeypatch.setenv("SAGE_BENCH_LOG_FILE", str(custom))
    args = SimpleNamespace(type="swebench", output=str(tmp_path / "x.json"))

    log_path = bench_main._setup_bench_file_log(args)

    assert log_path is not None
    assert Path(log_path).resolve() == custom.resolve()
    handlers = _find_file_handlers()
    assert len(handlers) == 1
    assert Path(handlers[0].baseFilename).resolve() == custom.resolve()
    # Parent directory should be created by the helper.
    assert custom.parent.is_dir()


def test_fallback_path_when_output_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("SAGE_BENCH_LOG_FILE", raising=False)
    args = SimpleNamespace(type="swebench", output=None)

    log_path = bench_main._setup_bench_file_log(args)

    assert log_path is not None
    p = Path(log_path)
    # Fallback goes under sage-python/logs/ with a UTC timestamp.
    assert p.parent.name == "logs"
    assert p.parent.parent.name == "sage-python"
    assert p.name.startswith("swebench-")
    assert p.suffix == ".log"
    assert p.parent.is_dir()


def test_no_duplicate_handler_on_reentry(tmp_path, monkeypatch):
    monkeypatch.delenv("SAGE_BENCH_LOG_FILE", raising=False)
    args = SimpleNamespace(type="swebench", output=str(tmp_path / "x.json"))

    first = bench_main._setup_bench_file_log(args)
    second = bench_main._setup_bench_file_log(args)

    assert first == second
    assert len(_find_file_handlers()) == 1


def test_formatter_uses_message_only(tmp_path, monkeypatch):
    monkeypatch.delenv("SAGE_BENCH_LOG_FILE", raising=False)
    args = SimpleNamespace(type="swebench", output=str(tmp_path / "x.json"))
    bench_main._setup_bench_file_log(args)

    handlers = _find_file_handlers()
    assert len(handlers) == 1
    fmt = handlers[0].formatter
    assert fmt is not None
    # basicConfig uses "%(message)s" — the file handler mirrors that so
    # operators see the same lines on disk as on stderr.
    assert fmt._fmt == "%(message)s"
