"""Tests for ExoCortex store-name resolution — no silent default.

Before 2026-04-18 the module silently fell back to
``fileSearchStores/ygnsageresearch-wii7kwkqozrd`` — the project
maintainer's personal store — whenever neither an explicit param nor the
``SAGE_EXOCORTEX_STORE`` env var was present. Any external
``pip install ygn-sage`` user hit that path and unintentionally wrote
into the project's shared research store. Flagged by Codex 2026-04-18 as
a v1.0 shipping blocker (P1.3 of mega-plan).

The contract pinned here:

* Explicit param wins over anything.
* ``SAGE_EXOCORTEX_STORE`` env var is the only silent source.
* No hardcoded default. When neither is provided, ``store_name`` is
  ``None``, ``is_available`` is ``False``, and ExoCortex features
  silently no-op (with a one-shot WARNING log — not asserted here, the
  behaviour is what matters).
"""
import logging
import sys
import types as _types

if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = _types.ModuleType("sage_core")

import pytest

from sage.memory.remote_rag import ExoCortex


def test_unset_store_returns_none(monkeypatch):
    """Absence of both param and env yields store_name=None — the shipping blocker."""
    monkeypatch.delenv("SAGE_EXOCORTEX_STORE", raising=False)
    exo = ExoCortex()
    assert exo.store_name is None


def test_unset_store_disables_availability(monkeypatch):
    """Without a store, ExoCortex is unavailable even with a valid API key."""
    monkeypatch.delenv("SAGE_EXOCORTEX_STORE", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    exo = ExoCortex()
    assert exo.is_available is False


def test_env_var_provides_store(monkeypatch):
    """SAGE_EXOCORTEX_STORE env var is the only silent source."""
    monkeypatch.setenv("SAGE_EXOCORTEX_STORE", "custom/store")
    exo = ExoCortex()
    assert exo.store_name == "custom/store"


def test_explicit_param_overrides_env(monkeypatch):
    """Explicit store_name always wins."""
    monkeypatch.setenv("SAGE_EXOCORTEX_STORE", "env/store")
    exo = ExoCortex(store_name="explicit/store")
    assert exo.store_name == "explicit/store"


def test_explicit_empty_string_disables(monkeypatch):
    """Passing store_name='' explicitly disables — stronger signal than None."""
    monkeypatch.setenv("SAGE_EXOCORTEX_STORE", "env/store")
    exo = ExoCortex(store_name="")
    # An explicit "" is treated as "caller explicitly wants disabled",
    # bypassing the env var.
    assert exo.store_name == ""
    assert exo.is_available is False


def test_none_param_falls_through_to_env(monkeypatch):
    """store_name=None is the conventional 'use defaults' signal."""
    monkeypatch.setenv("SAGE_EXOCORTEX_STORE", "env/store")
    exo = ExoCortex(store_name=None)
    assert exo.store_name == "env/store"


def test_no_module_level_default_store_constant():
    """The pre-April-18 DEFAULT_STORE constant must not reappear by accident."""
    import sage.memory.remote_rag as mod
    assert not hasattr(mod, "DEFAULT_STORE"), (
        "remote_rag.DEFAULT_STORE reintroduces the silent fallback that leaked the "
        "project's research store into every pip-install user. Keep the private "
        "_PROJECT_LEGACY_STORE marker instead, or gate behind an explicit opt-in."
    )


def test_warning_emitted_when_no_store(monkeypatch, caplog):
    """Operator must see a one-shot warning — silent breakage is the bug we fixed."""
    # The warning is module-global one-shot; reset the flag so the test is deterministic.
    import sage.memory.remote_rag as mod
    mod._WARNED_NO_STORE = False
    monkeypatch.delenv("SAGE_EXOCORTEX_STORE", raising=False)
    with caplog.at_level(logging.WARNING, logger="sage.memory.remote_rag"):
        ExoCortex()
    assert any("no store configured" in rec.message.lower() for rec in caplog.records)
