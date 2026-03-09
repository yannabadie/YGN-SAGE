"""Integration test fixtures -- NO mocks. Real implementations only."""
import pytest
from sage.memory.semantic import SemanticMemory
from sage.memory.episodic import EpisodicMemory
from sage.memory.embedder import Embedder
from sage.guardrails.builtin import CostGuardrail, OutputGuardrail
from sage.guardrails.base import GuardrailPipeline
from sage.events.bus import EventBus


@pytest.fixture
def semantic_memory():
    return SemanticMemory(max_relations=100)


@pytest.fixture
def episodic_memory(tmp_path):
    return EpisodicMemory(db_path=str(tmp_path / "test_episodic.db"))


@pytest.fixture
def embedder():
    return Embedder()  # Will use best available backend


@pytest.fixture
def guardrail_pipeline():
    return GuardrailPipeline([
        CostGuardrail(max_usd=1.0),
        OutputGuardrail(),
    ])


@pytest.fixture
def event_bus():
    return EventBus()
