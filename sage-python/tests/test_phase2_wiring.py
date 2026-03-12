"""Tests for Phase 2 wiring -- topology templates + verifier in boot."""
import pytest


def test_boot_phase2_removed():
    """AgentSystem must NOT have template_store/verifier (audit P10).

    DynamicTopologyEngine owns its verifier internally.
    """
    from sage.boot import boot_agent_system
    system = boot_agent_system(use_mock_llm=True)
    assert not hasattr(system, "template_store")
    assert not hasattr(system, "verifier")


def test_phase2_sage_core_direct():
    """PyTemplateStore and PyHybridVerifier are available directly from sage_core."""
    try:
        from sage_core import PyTemplateStore, PyHybridVerifier
    except ImportError:
        import pytest; pytest.skip("sage_core not compiled")
    store = PyTemplateStore()
    verifier = PyHybridVerifier()
    assert store is not None
    assert verifier is not None


def test_template_store_create_sequential():
    """If sage_core available, TemplateStore.create should return a TopologyGraph."""
    try:
        from sage_core import PyTemplateStore
    except ImportError:
        pytest.skip("sage_core not compiled")

    store = PyTemplateStore()
    graph = store.create("sequential", "test-model")
    assert graph.node_count() == 3
    assert graph.is_acyclic()


def test_verifier_validates_template():
    """If sage_core available, HybridVerifier should validate a template topology."""
    try:
        from sage_core import PyHybridVerifier, PyTemplateStore
    except ImportError:
        pytest.skip("sage_core not compiled")

    store = PyTemplateStore()
    verifier = PyHybridVerifier()
    graph = store.create("parallel", "test-model")
    result = verifier.verify(graph)
    assert result.valid
    assert len(result.errors) == 0


def test_template_store_available():
    """TemplateStore.available() should return 8 template names."""
    try:
        from sage_core import PyTemplateStore
    except ImportError:
        pytest.skip("sage_core not compiled")

    store = PyTemplateStore()
    names = store.available()
    assert len(names) == 8
    assert "sequential" in names
    assert "avr" in names


def test_verifier_custom_fan_limits():
    """HybridVerifier should respect custom fan-in/fan-out limits."""
    try:
        from sage_core import PyHybridVerifier, PyTemplateStore
    except ImportError:
        pytest.skip("sage_core not compiled")

    store = PyTemplateStore()
    # parallel template with 3 workers: aggregator has fan-in of 3 (message edges)
    graph = store.create("parallel", "test-model")

    # Tight fan-in limit should trigger error
    verifier = PyHybridVerifier(max_fan_in=1, max_fan_out=10)
    result = verifier.verify(graph)
    assert not result.valid
    assert any("fan-in" in e for e in result.errors)


def test_template_store_unknown_template():
    """TemplateStore.create should raise ValueError for unknown templates."""
    try:
        from sage_core import PyTemplateStore
    except ImportError:
        pytest.skip("sage_core not compiled")

    store = PyTemplateStore()
    with pytest.raises(ValueError, match="Unknown template"):
        store.create("nonexistent", "test-model")


def test_template_store_repr():
    """TemplateStore repr should show template count."""
    try:
        from sage_core import PyTemplateStore
    except ImportError:
        pytest.skip("sage_core not compiled")

    store = PyTemplateStore()
    assert "TemplateStore(templates=8)" == repr(store)


def test_verifier_repr():
    """HybridVerifier repr should show fan limits."""
    try:
        from sage_core import PyHybridVerifier
    except ImportError:
        pytest.skip("sage_core not compiled")

    verifier = PyHybridVerifier(max_fan_in=5, max_fan_out=7)
    assert "HybridVerifier(fan_in=5, fan_out=7)" == repr(verifier)


def test_verification_result_repr():
    """VerificationResult repr should show valid/invalid status."""
    try:
        from sage_core import PyHybridVerifier, PyTemplateStore
    except ImportError:
        pytest.skip("sage_core not compiled")

    store = PyTemplateStore()
    verifier = PyHybridVerifier()
    graph = store.create("sequential", "test-model")
    result = verifier.verify(graph)
    assert "VALID" in repr(result)


def test_all_templates_pass_verification():
    """All 8 built-in templates should pass default verification."""
    try:
        from sage_core import PyHybridVerifier, PyTemplateStore
    except ImportError:
        pytest.skip("sage_core not compiled")

    store = PyTemplateStore()
    verifier = PyHybridVerifier()
    for name in store.available():
        graph = store.create(name, "test-model")
        result = verifier.verify(graph)
        assert result.valid, f"Template '{name}' failed: {result.errors}"
