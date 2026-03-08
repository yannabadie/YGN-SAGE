"""Tests for ExoCortex model configurability."""
from sage.memory.remote_rag import ExoCortex, _DEFAULT_MODEL


def test_exocortex_accepts_model_parameter():
    """ExoCortex constructor must accept a model_id parameter."""
    exo = ExoCortex(store_name="test", model_id="custom-model")
    assert exo._model_id == "custom-model"


def test_exocortex_default_model():
    """ExoCortex should default to the known default model."""
    exo = ExoCortex(store_name="test")
    assert exo._model_id == _DEFAULT_MODEL


def test_exocortex_model_from_env(monkeypatch):
    """SAGE_EXOCORTEX_MODEL env var should override default."""
    monkeypatch.setenv("SAGE_EXOCORTEX_MODEL", "gpt-5-nano")
    exo = ExoCortex(store_name="test")
    assert exo._model_id == "gpt-5-nano"


def test_exocortex_explicit_overrides_env(monkeypatch):
    """Explicit model_id should override env var."""
    monkeypatch.setenv("SAGE_EXOCORTEX_MODEL", "env-model")
    exo = ExoCortex(store_name="test", model_id="explicit-model")
    assert exo._model_id == "explicit-model"
