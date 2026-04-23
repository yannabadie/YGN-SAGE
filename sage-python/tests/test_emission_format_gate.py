"""TDD red-phase tests for Track 2 task 2.3 — emission-format env-var gate.

Covers:

* ``_get_emission_format()`` in ``sage.bench.swebench_bench`` — reads
  ``SAGE_EMISSION_FORMAT`` and returns ``"unified"`` (default) or
  ``"search-replace"``. Unknown values WARN + fall back to unified.
* ``SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE`` in
  ``sage.input.swebench`` — parallel template with SEARCH/REPLACE-style
  patch format instructions.
* ``get_swebench_template()`` in ``sage.input.swebench`` — dispatcher
  returning the right template for the current env var.

Non-goals: no wiring into ``generate_patches`` (that's T2.4); no flip
of the default (that's T2.5). This file only tests the gate plumbing.
"""
from __future__ import annotations

import logging

import pytest


# ---------------------------------------------------------------------------
# _get_emission_format — env-var reader
# ---------------------------------------------------------------------------


def test_get_emission_format_defaults_to_unified_when_unset(monkeypatch):
    monkeypatch.delenv("SAGE_EMISSION_FORMAT", raising=False)
    from sage.bench.swebench_bench import _get_emission_format

    assert _get_emission_format() == "unified"


def test_get_emission_format_returns_search_replace_when_set(monkeypatch):
    monkeypatch.setenv("SAGE_EMISSION_FORMAT", "search-replace")
    from sage.bench.swebench_bench import _get_emission_format

    assert _get_emission_format() == "search-replace"


def test_get_emission_format_returns_unified_when_set_explicitly(monkeypatch):
    monkeypatch.setenv("SAGE_EMISSION_FORMAT", "unified")
    from sage.bench.swebench_bench import _get_emission_format

    assert _get_emission_format() == "unified"


def test_get_emission_format_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("SAGE_EMISSION_FORMAT", "SEARCH-REPLACE")
    from sage.bench.swebench_bench import _get_emission_format

    assert _get_emission_format() == "search-replace"


def test_get_emission_format_strips_surrounding_whitespace(monkeypatch):
    monkeypatch.setenv("SAGE_EMISSION_FORMAT", "  search-replace  ")
    from sage.bench.swebench_bench import _get_emission_format

    assert _get_emission_format() == "search-replace"


def test_get_emission_format_falls_back_and_warns_on_unknown(monkeypatch, caplog):
    monkeypatch.setenv("SAGE_EMISSION_FORMAT", "yolo")
    from sage.bench.swebench_bench import _get_emission_format

    caplog.set_level(logging.WARNING, logger="sage.bench.swebench_bench")
    result = _get_emission_format()

    assert result == "unified"
    assert any(
        record.levelno == logging.WARNING
        and "SAGE_EMISSION_FORMAT" in record.getMessage()
        and "yolo" in record.getMessage()
        for record in caplog.records
    ), f"expected WARN about yolo, got records: {caplog.records!r}"


def test_get_emission_format_falls_back_on_empty_string(monkeypatch):
    """Empty string is not in the whitelist -> WARN + fall back to unified."""
    monkeypatch.setenv("SAGE_EMISSION_FORMAT", "")
    from sage.bench.swebench_bench import _get_emission_format

    assert _get_emission_format() == "unified"


# ---------------------------------------------------------------------------
# SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE — prompt variant smoke test
# ---------------------------------------------------------------------------


def test_search_replace_template_contains_block_markers():
    from sage.input.swebench import SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE

    assert isinstance(SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE, str)
    assert "<<<<<<< SEARCH" in SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE
    assert "=======\n" in SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE
    assert ">>>>>>> REPLACE" in SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE
    assert "## File:" in SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE


def test_both_templates_have_problem_statement_placeholder():
    from sage.input.swebench import (
        SWEBENCH_SYSTEM_TEMPLATE,
        SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE,
    )

    assert "{problem_statement}" in SWEBENCH_SYSTEM_TEMPLATE
    assert "{problem_statement}" in SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE


def test_both_templates_keep_exploration_rule():
    """The "at least THREE distinct tool calls" guard must survive in
    both variants — that's part of the "Mandatory Workflow" section
    the spec says to keep identical."""
    from sage.input.swebench import (
        SWEBENCH_SYSTEM_TEMPLATE,
        SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE,
    )

    assert "at least THREE distinct tool calls" in SWEBENCH_SYSTEM_TEMPLATE
    assert "at least THREE distinct tool calls" in SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE


def test_repository_and_issue_description_sections_are_byte_identical():
    """The ``## Repository`` and ``## Issue Description`` sections MUST
    be byte-identical between the two templates — those identify the
    task and are the load-bearing discriminator for paired-smoke
    attribution. If the two templates' descriptions of the task drift,
    any pass-rate delta is confounded with prompt-phrasing drift
    rather than emission format.

    Narrower than the original T2.3 invariant (which locked the prefix
    all the way through ``## Mandatory Workflow``). The 2026-04-23
    prompt-hygiene pass found two byte-identity carryovers that made
    no sense under SR emission: the opening line ("minimal unified
    diff patch") and step 7 ("emit it directly in a ```diff fence").
    Those are ALLOWED to differ now; the contract narrows to the
    two sections that identify the task (the paired-smoke variable).

    Enforced byte-for-byte: hyphen-vs-em-dash, trailing whitespace,
    backslash line-continuations all count.
    """
    from sage.input.swebench import (
        SWEBENCH_SYSTEM_TEMPLATE,
        SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE,
    )

    start_marker = "## Repository"
    end_marker = "## Mandatory Workflow"

    def _extract_task_id_block(template: str) -> str:
        assert template.count(start_marker) == 1, (
            f"{start_marker!r} must appear exactly once"
        )
        assert template.count(end_marker) == 1, (
            f"{end_marker!r} must appear exactly once"
        )
        return template[
            template.index(start_marker) : template.index(end_marker)
        ]

    unified_block = _extract_task_id_block(SWEBENCH_SYSTEM_TEMPLATE)
    sr_block = _extract_task_id_block(SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE)

    assert unified_block == sr_block, (
        "Repository + Issue Description drift detected. These sections "
        "identify the task and must stay byte-identical so paired-smoke "
        "deltas attribute to the emission format, not prompt phrasing."
    )


# ---------------------------------------------------------------------------
# get_swebench_template — dispatcher
# ---------------------------------------------------------------------------


def test_get_swebench_template_defaults_to_unified(monkeypatch):
    monkeypatch.delenv("SAGE_EMISSION_FORMAT", raising=False)
    from sage.input.swebench import SWEBENCH_SYSTEM_TEMPLATE, get_swebench_template

    assert get_swebench_template() is SWEBENCH_SYSTEM_TEMPLATE


def test_get_swebench_template_returns_search_replace_when_flagged(monkeypatch):
    monkeypatch.setenv("SAGE_EMISSION_FORMAT", "search-replace")
    from sage.input.swebench import (
        SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE,
        get_swebench_template,
    )

    assert get_swebench_template() is SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE


def test_get_swebench_template_returns_unified_on_unknown(monkeypatch):
    monkeypatch.setenv("SAGE_EMISSION_FORMAT", "yolo")
    from sage.input.swebench import SWEBENCH_SYSTEM_TEMPLATE, get_swebench_template

    assert get_swebench_template() is SWEBENCH_SYSTEM_TEMPLATE


# ---------------------------------------------------------------------------
# Public API exports
# ---------------------------------------------------------------------------


def test_exports_from_sage_input():
    """T2.4 will call these from the ``sage.input`` package; they must
    be importable at the package root."""
    from sage.input import (
        SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE,
        get_swebench_template,
    )

    assert callable(get_swebench_template)
    assert isinstance(SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE, str)
