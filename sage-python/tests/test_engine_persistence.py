"""Tests for P1: Persist MAP-Elites + bandit posteriors across restarts."""
import os
import shutil
import tempfile

import pytest

# Skip if sage_core is not compiled with cognitive feature
try:
    from sage_core import TopologyEngine, ContextualBandit
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

pytestmark = pytest.mark.skipif(not _HAS_RUST, reason="sage_core not available")


@pytest.fixture
def state_dir():
    """Create a temporary directory for state persistence tests."""
    d = tempfile.mkdtemp(prefix="sage_persist_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestEngineStatePersistence:
    """Round-trip save/load tests for TopologyEngine state."""

    def test_save_load_creates_files(self, state_dir):
        """save_state should create bandit_state.db and archive_state.db."""
        engine = TopologyEngine()
        if not hasattr(engine, "save_state"):
            pytest.skip("save_state not available (cognitive feature missing)")

        engine.save_state(state_dir)

        assert os.path.exists(os.path.join(state_dir, "bandit_state.db"))
        assert os.path.exists(os.path.join(state_dir, "archive_state.db"))

    def test_save_load_round_trip(self, state_dir):
        """State should survive a save/load round trip."""
        if not hasattr(TopologyEngine(), "save_state"):
            pytest.skip("save_state not available (cognitive feature missing)")

        engine = TopologyEngine()

        # Generate + record some outcomes to populate archive and bandit
        result = engine.generate("Write a sorting function", None, 2, 0.5)
        engine.cache_topology(result.topology)
        engine.record_outcome(
            result.topology.id,
            "sorting task",
            ["sort", "algorithm"],
            None,
            0.9,
            0.01,
            150.0,
        )

        cells_before = engine.archive_cell_count()
        assert cells_before > 0, "Archive should have cells after record_outcome"

        # Save state
        engine.save_state(state_dir)

        # Create fresh engine and load
        engine2 = TopologyEngine()
        assert engine2.archive_cell_count() == 0
        arms, cells = engine2.load_state(state_dir)

        assert cells == cells_before, (
            f"Archive cells mismatch: loaded {cells}, expected {cells_before}"
        )

    def test_cold_start_load_returns_zeros(self, state_dir):
        """Loading from empty directory should return (0, 0) without error."""
        engine = TopologyEngine()
        if not hasattr(engine, "load_state"):
            pytest.skip("load_state not available (cognitive feature missing)")

        arms, cells = engine.load_state(state_dir)
        assert arms == 0
        assert cells == 0

    def test_loaded_engine_is_functional(self, state_dir):
        """Engine should work normally after loading persisted state."""
        if not hasattr(TopologyEngine(), "save_state"):
            pytest.skip("save_state not available (cognitive feature missing)")

        # Populate and save
        engine = TopologyEngine()
        result = engine.generate("Write quicksort", None, 2, 0.0)
        engine.cache_topology(result.topology)
        engine.record_outcome(
            result.topology.id, "quicksort", ["sort"], None, 0.85, 0.005, 200.0,
        )
        engine.save_state(state_dir)

        # Load into fresh engine
        engine2 = TopologyEngine()
        engine2.load_state(state_dir)

        # Should be able to generate without errors
        result2 = engine2.generate("Write merge sort", None, 2, 0.1)
        assert result2.topology.node_count() > 0
        assert result2.confidence >= 0.0


class TestBanditSqlitePersistence:
    """Tests for standalone bandit SQLite persistence (PyO3 wrappers)."""

    def test_bandit_save_load_round_trip(self, state_dir):
        """Bandit posteriors should survive a save/load round trip."""
        bandit = ContextualBandit(0.98, 0.15)
        if not hasattr(bandit, "save_to_sqlite"):
            pytest.skip("save_to_sqlite not available (cognitive feature missing)")

        bandit.register_arm("model-a", "sequential")
        bandit.register_arm("model-b", "avr")

        # Record some observations
        for _ in range(5):
            d = bandit.select(1.0)
            q = 0.9 if d.model_id == "model-a" else 0.3
            bandit.record(d.decision_id, q, 0.02, 150.0)

        db_path = os.path.join(state_dir, "bandit_test.db")
        bandit.save_to_sqlite(db_path)

        loaded = ContextualBandit.load_from_sqlite(db_path)
        assert loaded.arm_count() == bandit.arm_count()
        assert loaded.total_observations() == bandit.total_observations()

    def test_loaded_bandit_is_functional(self, state_dir):
        """Loaded bandit should be able to choose and record."""
        bandit = ContextualBandit(0.99, 0.1)
        if not hasattr(bandit, "save_to_sqlite"):
            pytest.skip("save_to_sqlite not available (cognitive feature missing)")

        bandit.register_arm("model-x", "avr")
        db_path = os.path.join(state_dir, "bandit_functional.db")
        bandit.save_to_sqlite(db_path)

        loaded = ContextualBandit.load_from_sqlite(db_path)
        decision = loaded.select(0.0)
        assert decision.model_id == "model-x"
        loaded.record(decision.decision_id, 0.8, 0.01, 100.0)
        assert loaded.total_observations() == 1
