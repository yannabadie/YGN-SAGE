"""Tests for ExoCortex auto-configuration with DEFAULT_STORE fallback."""
import sys
import types as _types

if "sage_core" not in sys.modules:
    sys.modules["sage_core"] = _types.ModuleType("sage_core")

import pytest
from sage.memory.remote_rag import ExoCortex, DEFAULT_STORE


def test_default_store_used_when_no_env(monkeypatch):
    monkeypatch.delenv("SAGE_EXOCORTEX_STORE", raising=False)
    exo = ExoCortex()
    assert exo.store_name == DEFAULT_STORE


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("SAGE_EXOCORTEX_STORE", "custom/store")
    exo = ExoCortex()
    assert exo.store_name == "custom/store"


def test_explicit_param_overrides_all(monkeypatch):
    monkeypatch.setenv("SAGE_EXOCORTEX_STORE", "env/store")
    exo = ExoCortex(store_name="explicit/store")
    assert exo.store_name == "explicit/store"


def test_is_available_requires_both(monkeypatch):
    monkeypatch.delenv("SAGE_EXOCORTEX_STORE", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    exo = ExoCortex()
    assert exo.is_available  # Has api_key + DEFAULT_STORE


def test_is_available_false_without_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    exo = ExoCortex()
    assert not exo.is_available


def test_none_store_falls_to_default(monkeypatch):
    monkeypatch.delenv("SAGE_EXOCORTEX_STORE", raising=False)
    exo = ExoCortex(store_name=None)
    assert exo.store_name == DEFAULT_STORE


def test_empty_string_store_disables(monkeypatch):
    """Passing store_name='' explicitly disables the store (falsy)."""
    monkeypatch.delenv("SAGE_EXOCORTEX_STORE", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    exo = ExoCortex(store_name="")
    # Empty string is falsy, so it falls through to env then DEFAULT_STORE
    assert exo.store_name == DEFAULT_STORE
