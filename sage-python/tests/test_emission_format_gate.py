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


def test_prefix_before_patch_format_is_byte_identical():
    """T2.3 spec: "Keeps the whole Repository, Issue Description,
    Mandatory Workflow sections IDENTICAL (line-for-line)."

    This invariant is load-bearing for the T2.5 paired smoke — if the
    unified and search-replace templates drift in wording above the
    "## Patch Format" seam, any pass-rate delta between the two arms
    is confounded with prompt drift rather than emission format.

    Enforced byte-for-byte: hyphen-vs-em-dash, trailing whitespace,
    backslash line-continuations all count.
    """
    from sage.input.swebench import (
        SWEBENCH_SYSTEM_TEMPLATE,
        SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE,
    )

    seam = "## Patch Format"
    assert SWEBENCH_SYSTEM_TEMPLATE.count(seam) == 1, (
        "seam must appear exactly once in unified template"
    )
    assert SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE.count(seam) == 1, (
        "seam must appear exactly once in search-replace template"
    )
    pre_unified = SWEBENCH_SYSTEM_TEMPLATE[: SWEBENCH_SYSTEM_TEMPLATE.index(seam)]
    pre_search_replace = SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE[
        : SWEBENCH_SYSTEM_TEMPLATE_SEARCH_REPLACE.index(seam)
    ]

    assert pre_unified == pre_search_replace, (
        "Prefix drift detected above the Patch Format seam. The spec "
        "requires byte-identity so T2.5 can attribute smoke deltas to "
        "the emission format alone."
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
