"""Tests for the ablation study framework."""
import pytest
from sage.bench.ablation import AblationConfig, ABLATION_CONFIGS


class TestAblationConfigDefaults:
    def test_defaults_all_true(self):
        cfg = AblationConfig()
        assert cfg.memory is True
        assert cfg.avr is True
        assert cfg.routing is True
        assert cfg.guardrails is True

    def test_default_label(self):
        cfg = AblationConfig()
        assert cfg.label == "full"


class TestAblationConfigCustom:
    def test_memory_false(self):
        cfg = AblationConfig(memory=False)
        assert cfg.memory is False
        assert cfg.avr is True
        assert cfg.routing is True
        assert cfg.guardrails is True

    def test_custom_label(self):
        cfg = AblationConfig(memory=False, label="no-memory")
        assert cfg.label == "no-memory"

    def test_all_false(self):
        cfg = AblationConfig(memory=False, avr=False, routing=False, guardrails=False)
        assert cfg.memory is False
        assert cfg.avr is False
        assert cfg.routing is False
        assert cfg.guardrails is False


class TestAblationConfigs:
    def test_has_six_entries(self):
        assert len(ABLATION_CONFIGS) == 6

    def test_correct_labels(self):
        labels = [c.label for c in ABLATION_CONFIGS]
        assert labels == ["full", "baseline", "no-memory", "no-avr", "no-routing", "no-guardrails"]

    def test_full_config(self):
        full = ABLATION_CONFIGS[0]
        assert full.memory is True
        assert full.avr is True
        assert full.routing is True
        assert full.guardrails is True

    def test_baseline_config(self):
        baseline = ABLATION_CONFIGS[1]
        assert baseline.memory is False
        assert baseline.avr is False
        assert baseline.routing is False
        assert baseline.guardrails is False

    def test_no_memory_config(self):
        cfg = ABLATION_CONFIGS[2]
        assert cfg.memory is False
        assert cfg.avr is True

    def test_no_avr_config(self):
        cfg = ABLATION_CONFIGS[3]
        assert cfg.avr is False
        assert cfg.memory is True

    def test_no_routing_config(self):
        cfg = ABLATION_CONFIGS[4]
        assert cfg.routing is False
        assert cfg.memory is True

    def test_no_guardrails_config(self):
        cfg = ABLATION_CONFIGS[5]
        assert cfg.guardrails is False
        assert cfg.memory is True


class TestAblationApply:
    def test_apply_sets_skip_flags(self):
        """apply() sets _skip_* flags on mock system's agent loop."""
        class MockLoop:
            _skip_memory = False
            _skip_avr = False
            _skip_routing = False
            _skip_guardrails = False

        class MockSystem:
            agent_loop = MockLoop()

        system = MockSystem()
        cfg = AblationConfig(memory=False, avr=True, routing=False, guardrails=True, label="custom")
        cfg.apply(system)

        assert system.agent_loop._skip_memory is True
        assert system.agent_loop._skip_avr is False
        assert system.agent_loop._skip_routing is True
        assert system.agent_loop._skip_guardrails is False

    def test_apply_full_disables_nothing(self):
        class MockLoop:
            _skip_memory = True
            _skip_avr = True
            _skip_routing = True
            _skip_guardrails = True

        class MockSystem:
            agent_loop = MockLoop()

        system = MockSystem()
        AblationConfig(label="full").apply(system)

        assert system.agent_loop._skip_memory is False
        assert system.agent_loop._skip_avr is False
        assert system.agent_loop._skip_routing is False
        assert system.agent_loop._skip_guardrails is False

    def test_apply_baseline_disables_all(self):
        class MockLoop:
            _skip_memory = False
            _skip_avr = False
            _skip_routing = False
            _skip_guardrails = False

        class MockSystem:
            agent_loop = MockLoop()

        system = MockSystem()
        AblationConfig(memory=False, avr=False, routing=False, guardrails=False, label="baseline").apply(system)

        assert system.agent_loop._skip_memory is True
        assert system.agent_loop._skip_avr is True
        assert system.agent_loop._skip_routing is True
        assert system.agent_loop._skip_guardrails is True
