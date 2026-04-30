"""Regression tests for ``sage.runtime.oracle.env.oracle_enabled``.

cgpro 2026-04-29 cycle-7 default-on flip approval requires:

- Unset ``SAGE_ORACLE`` ⇒ ON (default-on)
- ``SAGE_ORACLE=1`` ⇒ ON (explicit)
- ``SAGE_ORACLE=0`` ⇒ OFF (kill-switch)
- ``SAGE_ORACLE=false`` / ``off`` / ``no`` ⇒ OFF (case-insensitive aliases)
- whitespace-tolerance + case-insensitive

Pin the contract here so future predicate edits don't drift.
"""
from __future__ import annotations

import pytest

from sage.runtime.oracle.env import oracle_enabled


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SAGE_ORACLE", raising=False)


def test_unset_default_is_on() -> None:
    assert oracle_enabled() is True


def test_explicit_one_is_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAGE_ORACLE", "1")
    assert oracle_enabled() is True


def test_explicit_zero_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAGE_ORACLE", "0")
    assert oracle_enabled() is False


@pytest.mark.parametrize(
    "value",
    [
        "false", "False", "FALSE",
        "off", "OFF",
        "no", "NO",
        # cgpro 2026-04-30 cycle-7 VERIFY round-1: operators do type these.
        "disable", "Disable", "DISABLE",
        "disabled", "Disabled", "DISABLED",
    ],
)
def test_alias_false_values_are_off(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("SAGE_ORACLE", value)
    assert oracle_enabled() is False


@pytest.mark.parametrize("value", ["true", "TRUE", "on", "ON", "yes", "YES", "anything", "2", ""])
def test_unrecognized_value_is_on(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Anything not in the false-aliases set is treated as ON (post-flip
    default). Empty string is treated as ON because the env var is set
    (just blank) — caller should use ``=0`` for the kill-switch.
    """
    monkeypatch.setenv("SAGE_ORACLE", value)
    assert oracle_enabled() is True


def test_whitespace_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAGE_ORACLE", "  0  ")
    assert oracle_enabled() is False


def test_killswitch_takes_precedence_over_other_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even with other oracle-related env vars set, =0 always disables."""
    monkeypatch.setenv("SAGE_ORACLE", "0")
    monkeypatch.setenv("SAGE_RUN_FRAME", "1")
    monkeypatch.setenv("SAGE_BENCH_ORACLE_SEAM", "1")
    assert oracle_enabled() is False
