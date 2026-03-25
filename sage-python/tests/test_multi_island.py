"""Tests for multi-island evolutionary topology search."""
from sage.verl.multi_island import MultiIslandEvolver, IslandEntry, Island


class TestIsland:
    def test_insert_empty_cell(self):
        island = Island(island_id=0)
        entry = IslandEntry(yaml="test", score=0.5)
        assert island.insert((3, 0.7, 0.3, 1), entry)
        assert island.cell_count() == 1

    def test_insert_better_replaces(self):
        island = Island(island_id=0)
        island.insert((3, 0.7, 0.3, 1), IslandEntry(yaml="old", score=0.5))
        island.insert((3, 0.7, 0.3, 1), IslandEntry(yaml="new", score=0.8))
        assert island.best_entry().yaml == "new"

    def test_insert_worse_rejected(self):
        island = Island(island_id=0)
        island.insert((3, 0.7, 0.3, 1), IslandEntry(yaml="good", score=0.8))
        assert not island.insert((3, 0.7, 0.3, 1), IslandEntry(yaml="bad", score=0.3))
        assert island.best_entry().yaml == "good"


class TestMultiIslandEvolver:
    def test_two_islands(self):
        ev = MultiIslandEvolver(k=2)
        assert len(ev.islands) == 2

    def test_insert_and_select(self):
        ev = MultiIslandEvolver(k=2)
        ev.insert(0, (3, 1.0, 0.0, 0), IslandEntry(yaml="a", score=0.5))
        ev.insert(1, (4, 0.5, 0.5, 2), IslandEntry(yaml="b", score=0.8))
        assert ev.global_best().yaml == "b"

    def test_migration_transfers(self):
        ev = MultiIslandEvolver(k=2, migration_interval=1)
        ev.insert(0, (3, 1.0, 0.0, 0), IslandEntry(yaml="elite", score=0.9))
        # Island 1 is empty
        assert ev.islands[1].cell_count() == 0
        migrated = ev.maybe_migrate()
        assert migrated >= 1
        # Island 1 should now have the elite
        assert ev.islands[1].cell_count() >= 1

    def test_stats(self):
        ev = MultiIslandEvolver(k=2)
        ev.insert(0, (2, 1.0, 0.0, 0), IslandEntry(yaml="x", score=0.5))
        stats = ev.stats()
        assert stats["k"] == 2
        assert stats["total_cells"] >= 1
        assert len(stats["per_island"]) == 2
