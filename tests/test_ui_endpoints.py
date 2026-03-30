"""Tests for new UI endpoints added in Task 3.

Uses unittest.mock to avoid requiring sage-python installation.
"""
from __future__ import annotations

import sys
import types
import json
import unittest.mock as mock

import pytest


# ---------------------------------------------------------------------------
# Mock sage imports before importing app
# ---------------------------------------------------------------------------
def _setup_sage_mocks():
    """Install lightweight mocks for sage-python modules."""
    # sage.events.bus
    bus_mod = types.ModuleType("sage.events.bus")

    class MockEventBus:
        def __init__(self):
            self._events = []

        def emit(self, event):
            self._events.append(event)

        def query(self, last_n=100):
            return self._events[-last_n:]

        def clear(self):
            self._events.clear()

        async def stream(self):
            # Yield nothing — allows the websocket handler to block without errors
            return
            yield  # make it an async generator

    bus_mod.EventBus = MockEventBus

    # sage.agent_loop
    loop_mod = types.ModuleType("sage.agent_loop")

    class MockAgentEvent:
        def __init__(self, type="", step=0, timestamp=0.0, meta=None,
                     cost_usd=None, system=None, model=None, latency_ms=None):
            self.type = type
            self.step = step
            self.timestamp = timestamp
            self.meta = meta or {}
            self.cost_usd = cost_usd
            self.system = system
            self.model = model
            self.latency_ms = latency_ms

    loop_mod.AgentEvent = MockAgentEvent

    # Register mock modules
    sage_mod = types.ModuleType("sage")
    sage_events_mod = types.ModuleType("sage.events")
    sys.modules.setdefault("sage", sage_mod)
    sys.modules.setdefault("sage.events", sage_events_mod)
    sys.modules.setdefault("sage.events.bus", bus_mod)
    sys.modules.setdefault("sage.agent_loop", loop_mod)


_setup_sage_mocks()

# Now import the app (sage mocks are in place)
from ui.app import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _auth_headers():
    """Return empty auth headers (dev mode — no token configured)."""
    return {}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChatStreamRequiresMessage:
    """POST /api/chat/stream — 400 when message is empty."""

    def test_missing_message_key_returns_400(self):
        resp = client.post("/api/chat/stream", json={}, headers=_auth_headers())
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data

    def test_empty_string_message_returns_400(self):
        resp = client.post("/api/chat/stream", json={"message": ""}, headers=_auth_headers())
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data

    def test_whitespace_only_message_returns_400(self):
        resp = client.post("/api/chat/stream", json={"message": "   "}, headers=_auth_headers())
        assert resp.status_code == 400
        data = resp.json()
        assert "error" in data


class TestTopologyGraphReturnsStructure:
    """GET /api/topology/graph — returns valid Cytoscape.js-compatible structure."""

    def test_returns_expected_keys(self):
        resp = client.get("/api/topology/graph", headers=_auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data
        assert "source" in data
        assert "confidence" in data
        assert "template_library" in data

    def test_nodes_and_edges_are_lists(self):
        resp = client.get("/api/topology/graph", headers=_auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["nodes"], list)
        assert isinstance(data["edges"], list)

    def test_template_library_has_8_items(self):
        resp = client.get("/api/topology/graph", headers=_auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        library = data["template_library"]
        assert len(library) == 8
        assert "sequential" in library
        assert "debate" in library
        assert "self_moa" in library

    def test_confidence_is_numeric(self):
        resp = client.get("/api/topology/graph", headers=_auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["confidence"], (int, float))


class TestProvidersHealthReturnsList:
    """GET /api/providers/health — returns a list with required fields."""

    def test_returns_list(self):
        resp = client.get("/api/providers/health", headers=_auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_each_item_has_required_fields(self):
        resp = client.get("/api/providers/health", headers=_auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) > 0
        required = {
            "id", "provider", "available", "code_score", "reasoning_score",
            "cost_input", "cost_output", "circuit_state",
            "latency_p50_ms", "latency_p99_ms", "error_rate_1h",
        }
        for item in data:
            missing = required - set(item.keys())
            assert not missing, f"Provider entry missing fields: {missing}"

    def test_available_is_boolean(self):
        resp = client.get("/api/providers/health", headers=_auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        for item in data:
            assert isinstance(item["available"], bool)


class TestRoutingPipelineReturnsStages:
    """GET /api/routing/pipeline — returns pipeline stages and last_decision."""

    def test_returns_expected_keys(self):
        resp = client.get("/api/routing/pipeline", headers=_auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert "stages" in data
        assert "last_decision" in data

    def test_stages_has_4_items(self):
        resp = client.get("/api/routing/pipeline", headers=_auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["stages"]) == 4

    def test_stages_have_required_fields(self):
        resp = client.get("/api/routing/pipeline", headers=_auth_headers())
        assert resp.status_code == 200
        stages = resp.json()["stages"]
        required = {"name", "order", "method", "description"}
        for stage in stages:
            missing = required - set(stage.keys())
            assert not missing, f"Stage missing fields: {missing}"

    def test_knn_stage_references_arxiv(self):
        """Verify kNN stage cites arXiv 2505.12601 as per spec."""
        resp = client.get("/api/routing/pipeline", headers=_auth_headers())
        assert resp.status_code == 200
        stages = resp.json()["stages"]
        knn_stages = [s for s in stages if s.get("method") == "knn"]
        assert len(knn_stages) == 1
        assert "2505.12601" in knn_stages[0]["description"]

    def test_last_decision_is_none_when_no_system(self):
        resp = client.get("/api/routing/pipeline", headers=_auth_headers())
        assert resp.status_code == 200
        data = resp.json()
        # No system booted in tests — last_decision must be null
        assert data["last_decision"] is None

    def test_stage_orders_are_sequential(self):
        resp = client.get("/api/routing/pipeline", headers=_auth_headers())
        assert resp.status_code == 200
        stages = resp.json()["stages"]
        orders = [s["order"] for s in stages]
        assert sorted(orders) == list(range(1, len(stages) + 1))
