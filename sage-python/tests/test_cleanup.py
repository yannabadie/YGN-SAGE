"""Regression tests for cleanup: naming, dead code removal."""
from __future__ import annotations


def test_model_card_catalog_importable():
    """ModelCardCatalog is the renamed ModelRegistry from sage.llm.model_registry."""
    from sage.llm.model_registry import ModelCardCatalog
    assert ModelCardCatalog is not None


def test_dynamic_router_removed():
    """DynamicRouter was dead code — import must fail."""
    import importlib
    try:
        importlib.import_module("sage.routing.dynamic")
        assert False, "sage.routing.dynamic should not exist"
    except (ImportError, ModuleNotFoundError):
        pass


def test_topology_planner_removed():
    """TopologyPlanner was dead code — import must fail."""
    import importlib
    try:
        importlib.import_module("sage.topology.planner")
        assert False, "sage.topology.planner should not exist"
    except (ImportError, ModuleNotFoundError):
        pass
