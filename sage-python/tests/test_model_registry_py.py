"""Test Python ModelRegistry (migrated from Rust model_registry.rs)."""
import pytest
from sage.llm.model_card import ModelCard, CognitiveSystem
from sage.llm.model_registry import ModelRegistry, TelemetryRecord


class TestTelemetryRecord:
    def test_empty(self):
        tr = TelemetryRecord()
        assert tr.avg_quality() == 0.0
        assert tr.latency_p95() == 0.0

    def test_avg_quality(self):
        tr = TelemetryRecord()
        tr.record_full(0.8, 0.01, 100.0)
        tr.record_full(0.6, 0.02, 200.0)
        assert abs(tr.avg_quality() - 0.7) < 0.01

    def test_p95_latency(self):
        tr = TelemetryRecord()
        for i in range(20):
            tr.record_full(0.8, 0.01, 100.0 + i * 10.0)
        assert tr.latency_p95() > 200.0

    def test_zero_latency_not_recorded(self):
        tr = TelemetryRecord()
        tr.record(0.8, 0.01)
        tr.record(0.9, 0.02)
        tr.record_full(0.7, 0.01, 200.0)
        assert abs(tr.latency_p95() - 200.0) < 0.001

    def test_ring_buffer_bounded(self):
        tr = TelemetryRecord()
        for i in range(150):
            tr.record_full(0.5, 0.01, float(i))
        assert len(tr._latencies) <= 100


class TestModelRegistry:
    def _make_card(self, id="m1", **kw):
        defaults = dict(provider="test", family="test")
        defaults.update(kw)
        return ModelCard(id=id, **defaults)

    def test_register_and_get(self):
        reg = ModelRegistry()
        reg.register(self._make_card())
        assert reg.get("m1") is not None
        assert reg.get("m1").id == "m1"

    def test_len(self):
        reg = ModelRegistry()
        assert len(reg) == 0
        reg.register(self._make_card("a"))
        reg.register(self._make_card("b"))
        assert len(reg) == 2

    def test_list_ids(self):
        reg = ModelRegistry()
        reg.register(self._make_card("a"))
        reg.register(self._make_card("b"))
        assert set(reg.list_ids()) == {"a", "b"}

    def test_select_for_system_returns_sorted_list(self):
        reg = ModelRegistry()
        reg.register(self._make_card("fast", s1_affinity=0.9, s2_affinity=0.2))
        reg.register(self._make_card("smart", s1_affinity=0.2, s2_affinity=0.9))
        results = reg.select_for_system(CognitiveSystem.S2)
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0].id == "smart"

    def test_calibrated_affinity_no_telemetry(self):
        reg = ModelRegistry()
        reg.register(self._make_card("m1", s1_affinity=0.9))
        aff = reg.calibrated_affinity("m1", CognitiveSystem.S1)
        assert abs(aff - 0.9) < 0.001

    def test_calibrated_affinity_blends(self):
        reg = ModelRegistry()
        reg.register(self._make_card("m1", s1_affinity=0.9))
        for _ in range(25):
            reg.record_telemetry("m1", 0.5, 0.01)
        aff = reg.calibrated_affinity("m1", CognitiveSystem.S1)
        assert abs(aff - 0.70) < 0.01

    def test_calibrated_affinity_caps_at_80_percent(self):
        reg = ModelRegistry()
        reg.register(self._make_card("m1", s1_affinity=0.9))
        for _ in range(100):
            reg.record_telemetry("m1", 1.0, 0.01)
        aff = reg.calibrated_affinity("m1", CognitiveSystem.S1)
        assert abs(aff - 0.98) < 0.01

    def test_select_best_for_domain(self):
        reg = ModelRegistry()
        reg.register(self._make_card("math", domain_scores={"math": 0.94}, s3_affinity=0.9))
        reg.register(self._make_card("general", s1_affinity=0.9))
        best = reg.select_best_for_domain("math", 10.0)
        assert best is not None
        assert best.id == "math"
