"""Test Python ModelCard (migrated from Rust model_card.rs).

Field names, defaults, and method signatures match Rust PyO3 class exactly.
"""
import pytest
from sage.llm.model_card import ModelCard, CognitiveSystem


class TestCognitiveSystem:
    def test_values(self):
        assert CognitiveSystem.S1 == 1
        assert CognitiveSystem.S2 == 2
        assert CognitiveSystem.S3 == 3


def _make_card(s1=0.5, s2=0.5, s3=0.5, **kw):
    """Helper matching Rust make_test_card()."""
    defaults = dict(
        id="test", provider="test", family="test",
        code_score=0.5, reasoning_score=0.5, tool_use_score=0.5,
        math_score=0.5, formal_z3_strength=0.5,
        cost_input_per_m=1.0, cost_output_per_m=2.0,
        latency_ttft_ms=500.0, tokens_per_sec=100.0,
        s1_affinity=s1, s2_affinity=s2, s3_affinity=s3,
        context_window=128000,
    )
    defaults.update(kw)
    return ModelCard(**defaults)


class TestModelCard:
    def test_best_system_s1(self):
        card = _make_card(s1=0.9, s2=0.5, s3=0.3)
        assert card.best_system() == CognitiveSystem.S1

    def test_best_system_s2(self):
        card = _make_card(s1=0.3, s2=0.9, s3=0.5)
        assert card.best_system() == CognitiveSystem.S2

    def test_best_system_tie_favors_s1(self):
        card = _make_card(s1=0.7, s2=0.7, s3=0.7)
        assert card.best_system() == CognitiveSystem.S1

    def test_affinity_for(self):
        card = _make_card(s1=0.1, s2=0.5, s3=0.9)
        assert abs(card.affinity_for(CognitiveSystem.S1) - 0.1) < 0.001
        assert abs(card.affinity_for(CognitiveSystem.S2) - 0.5) < 0.001
        assert abs(card.affinity_for(CognitiveSystem.S3) - 0.9) < 0.001

    def test_estimate_cost(self):
        card = _make_card()
        assert abs(card.estimate_cost(1000, 500) - 0.002) < 0.0001

    def test_domain_score_known(self):
        card = _make_card(domain_scores={"math": 0.94, "code": 0.87})
        assert abs(card.domain_score("math") - 0.94) < 0.001

    def test_domain_score_unknown_returns_05(self):
        """Rust default is 0.5 (neutral), NOT 0.0."""
        card = _make_card()
        assert abs(card.domain_score("unknown") - 0.5) < 0.001

    def test_parse_toml(self):
        toml_str = '''
[[models]]
id = "gemini-2.5-flash"
provider = "google"
family = "gemini-2.5"
code_score = 0.85
reasoning_score = 0.80
tool_use_score = 0.90
math_score = 0.75
formal_z3_strength = 0.60
cost_input_per_m = 0.075
cost_output_per_m = 0.30
latency_ttft_ms = 200.0
tokens_per_sec = 200.0
s1_affinity = 0.70
s2_affinity = 0.85
s3_affinity = 0.40
recommended_topologies = ["sequential", "avr"]
supports_tools = true
supports_json_mode = true
supports_vision = true
context_window = 1048576
'''
        cards = ModelCard.parse_toml(toml_str)
        assert len(cards) == 1
        assert cards[0].id == "gemini-2.5-flash"
        assert abs(cards[0].s2_affinity - 0.85) < 0.001
        assert cards[0].context_window == 1048576
        assert cards[0].supports_tools is True

    def test_parse_toml_with_domain_scores(self):
        toml_str = '''
[[models]]
id = "test"
provider = "test"
family = "test"
code_score = 0.5
reasoning_score = 0.5
tool_use_score = 0.5
math_score = 0.5
formal_z3_strength = 0.5
cost_input_per_m = 1.0
cost_output_per_m = 2.0
latency_ttft_ms = 500.0
tokens_per_sec = 100.0
s1_affinity = 0.5
s2_affinity = 0.5
s3_affinity = 0.5
recommended_topologies = []
supports_tools = false
supports_json_mode = false
supports_vision = false
context_window = 128000
safety_rating = 0.85

[models.domain_scores]
math = 0.94
code = 0.87
'''
        cards = ModelCard.parse_toml(toml_str)
        assert abs(cards[0].safety_rating - 0.85) < 0.001
        assert abs(cards[0].domain_score("math") - 0.94) < 0.001
