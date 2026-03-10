use sage_core::topology::map_elites::{BehaviorDescriptor, MapElitesArchive};
use sage_core::topology::templates;
use sage_core::topology::topology_graph::*;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn make_valid_graph(template: &str, model: &str) -> TopologyGraph {
    templates::TemplateStore::create(template, model).unwrap()
}

fn make_multi_model_graph() -> TopologyGraph {
    let mut g = TopologyGraph::try_new("sequential").unwrap();
    let n0 = TopologyNode::new(
        "coder".into(),
        "gemini-flash".into(),
        1,
        vec!["text".into()],
        0,
        1.0,
        60.0,
    );
    let n1 = TopologyNode::new(
        "reviewer".into(),
        "gpt-5".into(),
        2,
        vec!["reasoning".into()],
        0,
        2.0,
        120.0,
    );
    let n2 = TopologyNode::new(
        "formatter".into(),
        "claude-4".into(),
        1,
        vec!["text".into()],
        0,
        1.0,
        60.0,
    );
    let i0 = g.add_node(n0);
    let i1 = g.add_node(n1);
    let i2 = g.add_node(n2);
    g.try_add_edge(i0, i1, TopologyEdge::control()).unwrap();
    g.try_add_edge(i1, i2, TopologyEdge::control()).unwrap();
    g
}

// ── Test 1: Empty archive has zero cells ─────────────────────────────────────

#[test]
fn test_empty_archive_has_zero_cells() {
    let archive = MapElitesArchive::new();
    assert_eq!(archive.cell_count(), 0);
    assert!(archive.best_by_quality().is_none());
    assert!(archive.all_entries().is_empty());
    assert!((archive.coverage() - 0.0).abs() < f32::EPSILON);
}

// ── Test 2: Insert into empty cell succeeds ──────────────────────────────────

#[test]
fn test_insert_into_empty_cell_succeeds() {
    let mut archive = MapElitesArchive::new();
    let graph = make_valid_graph("sequential", "model-a");
    let desc = BehaviorDescriptor::from_raw(3, 2, 0.05, 0.5);

    let inserted = archive.insert(&desc, graph, 0.9, 0.05, 100.0);
    assert!(inserted);
    assert_eq!(archive.cell_count(), 1);

    let entry = archive.get(&desc).unwrap();
    assert!((entry.quality - 0.9).abs() < f32::EPSILON);
    assert!((entry.cost - 0.05).abs() < f32::EPSILON);
    assert!((entry.latency_ms - 100.0).abs() < f32::EPSILON);
    assert_eq!(entry.evaluation_count, 1);
}

// ── Test 3: Insert invalid topology rejected ─────────────────────────────────

#[test]
fn test_insert_invalid_topology_rejected() {
    let mut archive = MapElitesArchive::new();

    // Node with empty model_id and empty capabilities => fails verifier.
    let mut graph = TopologyGraph::try_new("sequential").unwrap();
    let node = TopologyNode::new(
        "broken".into(),
        "".into(),
        1,
        vec![],
        0,
        1.0,
        60.0,
    );
    graph.add_node(node);

    let desc = BehaviorDescriptor::from_raw(1, 1, 0.005, 0.0);
    let inserted = archive.insert(&desc, graph, 0.8, 0.005, 50.0);
    assert!(!inserted, "Invalid topology should be rejected");
    assert_eq!(archive.cell_count(), 0);
}

// ── Test 4: Pareto domination replaces inferior entry ────────────────────────

#[test]
fn test_pareto_domination_replaces_inferior() {
    let mut archive = MapElitesArchive::new();
    let desc = BehaviorDescriptor::from_raw(3, 2, 0.05, 0.5);

    // Insert initial entry.
    let graph1 = make_valid_graph("sequential", "model-a");
    assert!(archive.insert(&desc, graph1, 0.7, 0.08, 200.0));

    // Insert Pareto-dominating entry (strictly higher quality AND strictly lower cost).
    let graph2 = make_valid_graph("sequential", "model-b");
    assert!(archive.insert(&desc, graph2, 0.9, 0.03, 100.0));

    // Verify replacement.
    assert_eq!(archive.cell_count(), 1);
    let entry = archive.get(&desc).unwrap();
    assert!((entry.quality - 0.9).abs() < f32::EPSILON);
    assert!((entry.cost - 0.03).abs() < f32::EPSILON);
}

// ── Test 5: Non-dominating insert rejected ───────────────────────────────────

#[test]
fn test_non_dominating_insert_rejected() {
    let mut archive = MapElitesArchive::new();
    let desc = BehaviorDescriptor::from_raw(3, 2, 0.05, 0.5);

    // Insert initial entry with good quality and low cost.
    let graph1 = make_valid_graph("sequential", "model-a");
    assert!(archive.insert(&desc, graph1, 0.9, 0.03, 100.0));

    // Attempt: lower quality AND higher cost => rejected.
    let graph2 = make_valid_graph("sequential", "model-b");
    assert!(!archive.insert(&desc, graph2, 0.7, 0.08, 200.0));

    // Attempt: higher quality but also higher cost => rejected.
    let graph3 = make_valid_graph("sequential", "model-c");
    assert!(!archive.insert(&desc, graph3, 0.95, 0.05, 150.0));

    // Attempt: lower cost but also lower quality => rejected.
    let graph4 = make_valid_graph("sequential", "model-d");
    assert!(!archive.insert(&desc, graph4, 0.8, 0.01, 80.0));

    // Attempt: same quality, same cost => rejected (must be strictly better).
    let graph5 = make_valid_graph("sequential", "model-e");
    assert!(!archive.insert(&desc, graph5, 0.9, 0.03, 90.0));

    // Verify original entry unchanged.
    let entry = archive.get(&desc).unwrap();
    assert!((entry.quality - 0.9).abs() < f32::EPSILON);
    assert!((entry.cost - 0.03).abs() < f32::EPSILON);
}

// ── Test 6: best_by_quality finds highest ────────────────────────────────────

#[test]
fn test_best_by_quality_finds_highest() {
    let mut archive = MapElitesArchive::new();

    // Insert three entries in different cells with different qualities.
    let desc1 = BehaviorDescriptor::from_raw(1, 1, 0.005, 0.1);
    let desc2 = BehaviorDescriptor::from_raw(3, 2, 0.05, 0.5);
    let desc3 = BehaviorDescriptor::from_raw(6, 5, 0.20, 0.9);

    archive.insert(&desc1, make_valid_graph("sequential", "m"), 0.70, 0.005, 50.0);
    archive.insert(&desc2, make_valid_graph("sequential", "m"), 0.95, 0.05, 100.0);
    archive.insert(&desc3, make_valid_graph("sequential", "m"), 0.80, 0.20, 200.0);

    let best = archive.best_by_quality().unwrap();
    assert!((best.quality - 0.95).abs() < f32::EPSILON);
}

// ── Test 7: all_entries returns correct count ────────────────────────────────

#[test]
fn test_all_entries_returns_correct_count() {
    let mut archive = MapElitesArchive::new();

    let descriptors = [
        BehaviorDescriptor::from_raw(1, 1, 0.005, 0.1),
        BehaviorDescriptor::from_raw(2, 3, 0.05, 0.5),
        BehaviorDescriptor::from_raw(4, 5, 0.20, 0.9),
    ];

    for (i, desc) in descriptors.iter().enumerate() {
        archive.insert(
            desc,
            make_valid_graph("sequential", "m"),
            0.5 + i as f32 * 0.1,
            0.005 + i as f32 * 0.05,
            50.0 + i as f32 * 50.0,
        );
    }

    let entries = archive.all_entries();
    assert_eq!(entries.len(), 3);
    assert_eq!(archive.cell_count(), 3);

    // Each entry should have the correct key.
    for (key, _entry) in &entries {
        assert!(key.0 >= 1 && key.0 <= 4); // valid agent count bucket
        assert!(key.1 >= 1 && key.1 <= 3); // valid depth bucket
        assert!(key.2 >= 1 && key.2 <= 3); // valid cost bucket
        assert!(key.3 >= 1 && key.3 <= 3); // valid diversity bucket
    }
}

// ── Test 8: Coverage calculation ─────────────────────────────────────────────

#[test]
fn test_coverage_calculation() {
    let mut archive = MapElitesArchive::new();
    assert!((archive.coverage() - 0.0).abs() < f32::EPSILON);

    // Insert one entry.
    let desc = BehaviorDescriptor::from_raw(1, 1, 0.005, 0.1);
    archive.insert(&desc, make_valid_graph("sequential", "m"), 0.8, 0.005, 50.0);

    let expected = 1.0 / 108.0;
    assert!(
        (archive.coverage() - expected).abs() < 1e-5,
        "Expected coverage ~{}, got {}",
        expected,
        archive.coverage()
    );

    // Insert second entry in different cell.
    let desc2 = BehaviorDescriptor::from_raw(6, 5, 0.20, 0.9);
    archive.insert(&desc2, make_valid_graph("sequential", "m"), 0.7, 0.20, 200.0);

    let expected2 = 2.0 / 108.0;
    assert!(
        (archive.coverage() - expected2).abs() < 1e-5,
        "Expected coverage ~{}, got {}",
        expected2,
        archive.coverage()
    );

    // Total possible cells.
    assert_eq!(BehaviorDescriptor::TOTAL_CELLS, 108);
}

// ── Test 9: BehaviorDescriptor bucketing logic ───────────────────────────────

#[test]
fn test_behavior_descriptor_bucketing() {
    // Agent count: 1=solo, 2=pair, 3=small team (3-5), 4=large (6+).
    let solo = BehaviorDescriptor::from_raw(1, 1, 0.005, 0.1);
    assert_eq!(solo.agent_count_bucket, 1);

    let pair = BehaviorDescriptor::from_raw(2, 1, 0.005, 0.1);
    assert_eq!(pair.agent_count_bucket, 2);

    let small_team = BehaviorDescriptor::from_raw(4, 1, 0.005, 0.1);
    assert_eq!(small_team.agent_count_bucket, 3);

    let large = BehaviorDescriptor::from_raw(10, 1, 0.005, 0.1);
    assert_eq!(large.agent_count_bucket, 4);

    // Max depth: 1=shallow (1-2), 2=medium (3-4), 3=deep (5+).
    let shallow = BehaviorDescriptor::from_raw(1, 2, 0.005, 0.1);
    assert_eq!(shallow.max_depth_bucket, 1);

    let medium = BehaviorDescriptor::from_raw(1, 4, 0.005, 0.1);
    assert_eq!(medium.max_depth_bucket, 2);

    let deep = BehaviorDescriptor::from_raw(1, 7, 0.005, 0.1);
    assert_eq!(deep.max_depth_bucket, 3);

    // Cost: 1=cheap (<$0.01), 2=moderate ($0.01-$0.10), 3=expensive (>$0.10).
    let cheap = BehaviorDescriptor::from_raw(1, 1, 0.005, 0.1);
    assert_eq!(cheap.cost_bucket, 1);

    let moderate = BehaviorDescriptor::from_raw(1, 1, 0.05, 0.1);
    assert_eq!(moderate.cost_bucket, 2);

    let expensive = BehaviorDescriptor::from_raw(1, 1, 0.50, 0.1);
    assert_eq!(expensive.cost_bucket, 3);

    // Model diversity: 1=homogeneous (<0.3), 2=mixed (0.3-0.7), 3=diverse (>0.7).
    let homo = BehaviorDescriptor::from_raw(1, 1, 0.005, 0.1);
    assert_eq!(homo.model_diversity_bucket, 1);

    let mixed = BehaviorDescriptor::from_raw(1, 1, 0.005, 0.5);
    assert_eq!(mixed.model_diversity_bucket, 2);

    let diverse = BehaviorDescriptor::from_raw(1, 1, 0.005, 0.9);
    assert_eq!(diverse.model_diversity_bucket, 3);

    // Verify composite key.
    let desc = BehaviorDescriptor::from_raw(4, 3, 0.05, 0.5);
    assert_eq!(desc.key(), (3, 2, 2, 2));
}

// ── Test 10: from_topology extracts correct features ─────────────────────────

#[test]
fn test_from_topology_extracts_features() {
    // Sequential template: 3 nodes, depth 2, single model.
    let graph = make_valid_graph("sequential", "same-model");
    let desc = BehaviorDescriptor::from_topology(&graph, 0.005);

    assert_eq!(graph.node_count(), 3);
    assert_eq!(desc.agent_count_bucket, 3);     // 3 agents => small team
    assert_eq!(desc.max_depth_bucket, 1);        // depth 2 => shallow
    assert_eq!(desc.cost_bucket, 1);             // $0.005 => cheap

    // 1 unique model / 3 nodes = 0.333 => mixed (0.3-0.7)
    assert_eq!(desc.model_diversity_bucket, 2);

    // Multi-model: 3 nodes, 3 different models => diversity = 1.0.
    let multi = make_multi_model_graph();
    let desc2 = BehaviorDescriptor::from_topology(&multi, 0.20);
    assert_eq!(desc2.model_diversity_bucket, 3); // 1.0 => diverse

    // Parallel template with 3 workers: 5 nodes.
    let parallel = make_valid_graph("parallel", "model-x");
    let desc3 = BehaviorDescriptor::from_topology(&parallel, 0.01);
    assert_eq!(parallel.node_count(), 5);
    assert_eq!(desc3.agent_count_bucket, 3); // 5 agents => small team (3-5)
}

// ── Test 11: Save/load round-trip (cognitive feature) ────────────────────────

#[cfg(feature = "cognitive")]
mod cognitive_tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_db() -> PathBuf {
        let dir = std::env::temp_dir();
        dir.join(format!("sage_map_elites_test_{}.db", ulid::Ulid::new()))
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let db = temp_db();
        let mut archive = MapElitesArchive::new();

        // Populate with several entries.
        let desc1 = BehaviorDescriptor::from_raw(1, 1, 0.005, 0.1);
        let desc2 = BehaviorDescriptor::from_raw(3, 2, 0.05, 0.5);
        let desc3 = BehaviorDescriptor::from_raw(6, 5, 0.20, 0.9);

        archive.insert(&desc1, make_valid_graph("sequential", "model-a"), 0.70, 0.005, 50.0);
        archive.insert(&desc2, make_valid_graph("parallel", "model-b"), 0.85, 0.05, 100.0);
        archive.insert(&desc3, make_valid_graph("sequential", "model-c"), 0.92, 0.20, 200.0);

        assert_eq!(archive.cell_count(), 3);

        // Save.
        archive.save_to_sqlite(db.to_str().unwrap()).unwrap();

        // Load.
        let loaded = MapElitesArchive::load_from_sqlite(db.to_str().unwrap()).unwrap();
        assert_eq!(loaded.cell_count(), 3);

        // Verify entries were preserved.
        let entry1 = loaded.get(&desc1).unwrap();
        assert!((entry1.quality - 0.70).abs() < 1e-5);
        assert!((entry1.cost - 0.005).abs() < 1e-5);
        assert_eq!(entry1.evaluation_count, 1);

        let entry2 = loaded.get(&desc2).unwrap();
        assert!((entry2.quality - 0.85).abs() < 1e-5);

        let entry3 = loaded.get(&desc3).unwrap();
        assert!((entry3.quality - 0.92).abs() < 1e-5);

        // Best by quality should be entry3.
        let best = loaded.best_by_quality().unwrap();
        assert!((best.quality - 0.92).abs() < 1e-5);

        // Topology graphs should have correct structure.
        assert_eq!(entry1.graph.node_count(), 3); // sequential = 3 nodes
        assert_eq!(entry2.graph.node_count(), 5); // parallel(3 workers) = 5 nodes

        // Cleanup.
        let _ = std::fs::remove_file(&db);
    }

    #[test]
    fn test_save_empty_archive() {
        let db = temp_db();
        let archive = MapElitesArchive::new();

        archive.save_to_sqlite(db.to_str().unwrap()).unwrap();

        let loaded = MapElitesArchive::load_from_sqlite(db.to_str().unwrap()).unwrap();
        assert_eq!(loaded.cell_count(), 0);

        let _ = std::fs::remove_file(&db);
    }

    #[test]
    fn test_save_overwrites_previous() {
        let db = temp_db();

        // First save: 2 entries.
        let mut archive1 = MapElitesArchive::new();
        let desc1 = BehaviorDescriptor::from_raw(1, 1, 0.005, 0.1);
        let desc2 = BehaviorDescriptor::from_raw(3, 2, 0.05, 0.5);
        archive1.insert(&desc1, make_valid_graph("sequential", "m"), 0.7, 0.005, 50.0);
        archive1.insert(&desc2, make_valid_graph("sequential", "m"), 0.8, 0.05, 100.0);
        archive1.save_to_sqlite(db.to_str().unwrap()).unwrap();

        // Second save: 1 entry only.
        let mut archive2 = MapElitesArchive::new();
        let desc3 = BehaviorDescriptor::from_raw(6, 5, 0.20, 0.9);
        archive2.insert(&desc3, make_valid_graph("sequential", "m"), 0.9, 0.20, 200.0);
        archive2.save_to_sqlite(db.to_str().unwrap()).unwrap();

        // Load should have only 1 entry (second save cleared the first).
        let loaded = MapElitesArchive::load_from_sqlite(db.to_str().unwrap()).unwrap();
        assert_eq!(loaded.cell_count(), 1);
        assert!(loaded.get(&desc3).is_some());

        let _ = std::fs::remove_file(&db);
    }
}
