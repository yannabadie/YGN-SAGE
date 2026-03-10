//! Integration tests for DynamicTopologyEngine.

use sage_core::memory::smmu::MultiViewMMU;
use sage_core::topology::engine::{TopologyEngine, TopologySource};
use sage_core::topology::templates;

// ── Test 1: New engine has empty cache ───────────────────────────────────

#[test]
fn test_new_engine_has_empty_cache() {
    let engine = TopologyEngine::new();
    assert_eq!(engine.topology_count(), 0);
}

// ── Test 2: New engine has empty archive ─────────────────────────────────

#[test]
fn test_new_engine_has_empty_archive() {
    let engine = TopologyEngine::new();
    assert_eq!(engine.archive().cell_count(), 0);
}

// ── Test 3: Template fallback for S1 → sequential ───────────────────────

#[test]
fn test_template_fallback_s1_sequential() {
    let mut engine = TopologyEngine::new();
    let smmu = MultiViewMMU::new();

    let result = engine.generate(&smmu, "simple greeting", None, 1, 0.0);
    assert_eq!(result.source, TopologySource::TemplateFallback);
    assert_eq!(result.topology.template_type, "sequential");
    assert_eq!(result.topology.node_count(), 3);
}

// ── Test 4: Template fallback for S2 → avr ──────────────────────────────

#[test]
fn test_template_fallback_s2_avr() {
    let mut engine = TopologyEngine::new();
    let smmu = MultiViewMMU::new();

    let result = engine.generate(&smmu, "write sorting code", None, 2, 0.0);
    assert_eq!(result.source, TopologySource::TemplateFallback);
    assert_eq!(result.topology.template_type, "avr");
    assert_eq!(result.topology.node_count(), 3);
}

// ── Test 5: Template fallback for S3 → debate ───────────────────────────

#[test]
fn test_template_fallback_s3_debate() {
    let mut engine = TopologyEngine::new();
    let smmu = MultiViewMMU::new();

    let result = engine.generate(&smmu, "prove theorem", None, 3, 0.0);
    assert_eq!(result.source, TopologySource::TemplateFallback);
    assert_eq!(result.topology.template_type, "debate");
    assert_eq!(result.topology.node_count(), 4);
}

// ── Test 6: cache_topology stores and get_topology retrieves ─────────────

#[test]
fn test_cache_topology_stores_and_retrieves() {
    let mut engine = TopologyEngine::new();

    let graph1 = templates::sequential("model-a");
    let id1 = graph1.id.clone();
    engine.cache_topology(graph1);

    let graph2 = templates::avr("model-b", "model-c");
    let id2 = graph2.id.clone();
    engine.cache_topology(graph2);

    assert_eq!(engine.topology_count(), 2);

    let retrieved1 = engine.get_topology(&id1).unwrap();
    assert_eq!(retrieved1.template_type, "sequential");

    let retrieved2 = engine.get_topology(&id2).unwrap();
    assert_eq!(retrieved2.template_type, "avr");

    // Missing ID returns None
    assert!(engine.get_topology("nonexistent-id").is_none());
}

// ── Test 7: record_outcome feeds archive ─────────────────────────────────

#[test]
fn test_record_outcome_feeds_archive() {
    let mut engine = TopologyEngine::new();
    let mut smmu = MultiViewMMU::new();

    // Cache a topology first
    let graph = templates::sequential("model-a");
    let topo_id = graph.id.clone();
    engine.cache_topology(graph);

    assert_eq!(engine.archive().cell_count(), 0);

    engine.record_outcome(
        &mut smmu,
        &topo_id,
        "Sort an array efficiently",
        vec!["sort".into(), "array".into()],
        None,
        0.9,
        0.005,
        500.0,
    );

    // Archive should have gained an entry
    assert!(
        engine.archive().cell_count() > 0,
        "Archive should have at least one cell after record_outcome"
    );
}

// ── Test 8: record_outcome feeds S-MMU bridge ────────────────────────────

#[test]
fn test_record_outcome_feeds_smmu_bridge() {
    let mut engine = TopologyEngine::new();
    let mut smmu = MultiViewMMU::new();

    let graph = templates::avr("model-a", "model-b");
    let topo_id = graph.id.clone();
    engine.cache_topology(graph);

    assert_eq!(engine.bridge().chunk_count(), 0);

    engine.record_outcome(
        &mut smmu,
        &topo_id,
        "Write a binary search",
        vec!["binary_search".into()],
        Some(vec![1.0; 384]),
        0.85,
        0.01,
        300.0,
    );

    assert_eq!(
        engine.bridge().chunk_count(),
        1,
        "Bridge should have one chunk after record_outcome"
    );
    assert_eq!(smmu.chunk_count(), 1, "S-MMU should have one chunk");
}

// ── Test 9: generate with empty state falls back to template ─────────────

#[test]
fn test_generate_empty_state_falls_back() {
    let mut engine = TopologyEngine::new();
    let smmu = MultiViewMMU::new();

    // All three system tiers should fall back to templates
    for (system, expected_template) in [(1, "sequential"), (2, "avr"), (3, "debate")] {
        let result = engine.generate(&smmu, "any task", None, system, 0.5);
        assert_eq!(
            result.source,
            TopologySource::TemplateFallback,
            "System {} should fall back to template",
            system
        );
        assert_eq!(
            result.topology.template_type, expected_template,
            "System {} should produce {} template",
            system, expected_template
        );
    }
}

// ── Test 10: generate after record_outcome may hit archive or mutation ───

#[test]
fn test_generate_after_record_may_hit_archive_or_mutation() {
    let mut engine = TopologyEngine::new();
    let mut smmu = MultiViewMMU::new();

    // Record several outcomes to populate archive
    for i in 0..5 {
        let graph = templates::sequential(&format!("model-{}", i));
        let topo_id = graph.id.clone();
        engine.cache_topology(graph);

        engine.record_outcome(
            &mut smmu,
            &topo_id,
            &format!("Task {}", i),
            vec!["code".into()],
            None,
            0.9,
            0.005,
            200.0,
        );
    }

    // Archive should be populated
    assert!(engine.archive().cell_count() > 0);

    // Generate should now potentially hit archive or mutation instead of just template
    let result = engine.generate(&smmu, "Write quicksort", None, 2, 0.1);

    // Could be ArchiveHit, Mutation, or TemplateFallback — all are valid outcomes
    // The key assertion is that it doesn't panic and returns a valid topology
    assert!(result.topology.node_count() > 0);
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
}

// ── Test 11: evolve runs without error on populated archive ──────────────

#[test]
fn test_evolve_on_populated_archive() {
    let mut engine = TopologyEngine::new();
    let mut smmu = MultiViewMMU::new();

    // Populate archive with a few entries
    for i in 0..3 {
        let graph = templates::sequential(&format!("model-{}", i));
        let topo_id = graph.id.clone();
        engine.cache_topology(graph);

        engine.record_outcome(
            &mut smmu,
            &topo_id,
            &format!("Task {}", i),
            vec!["test".into()],
            None,
            0.8,
            0.005 + i as f32 * 0.01,
            100.0 + i as f32 * 50.0,
        );
    }

    let cells_before = engine.archive().cell_count();
    assert!(cells_before > 0);

    // Run evolution — should not panic
    engine.evolve(4, 3);

    // Archive should have at least as many cells (evolution can only add, not remove)
    assert!(
        engine.archive().cell_count() >= cells_before,
        "Evolution should not reduce archive cells"
    );
}

// ── Test 12: evolve on empty archive is a no-op ─────────────────────────

#[test]
fn test_evolve_empty_archive_noop() {
    let mut engine = TopologyEngine::new();
    assert_eq!(engine.archive().cell_count(), 0);

    // Should not panic or change anything
    engine.evolve(10, 5);

    assert_eq!(engine.archive().cell_count(), 0);
}

// ── Test 13: topology_count tracks cache size ────────────────────────────

#[test]
fn test_topology_count_tracks_cache_size() {
    let mut engine = TopologyEngine::new();
    assert_eq!(engine.topology_count(), 0);

    engine.cache_topology(templates::sequential("m1"));
    assert_eq!(engine.topology_count(), 1);

    engine.cache_topology(templates::avr("m2", "m3"));
    assert_eq!(engine.topology_count(), 2);

    engine.cache_topology(templates::debate("m4", "m5"));
    assert_eq!(engine.topology_count(), 3);

    engine.cache_topology(templates::parallel("m6", 3));
    assert_eq!(engine.topology_count(), 4);
}

// ── Bonus Test 14: record_outcome with uncached topology uses defaults ───

#[test]
fn test_record_outcome_uncached_topology() {
    let mut engine = TopologyEngine::new();
    let mut smmu = MultiViewMMU::new();

    // Record outcome for a topology not in cache — should not panic
    engine.record_outcome(
        &mut smmu,
        "01JTEST_UNCACHED",
        "Task with uncached topology",
        vec!["test".into()],
        None,
        0.7,
        0.01,
        200.0,
    );

    // Bridge should still have recorded it
    assert_eq!(engine.bridge().chunk_count(), 1);
}

// ── Bonus Test 15: multiple evolve rounds are cumulative ─────────────────

#[test]
fn test_multiple_evolve_rounds() {
    let mut engine = TopologyEngine::new();
    let mut smmu = MultiViewMMU::new();

    // Seed archive
    let graph = templates::sequential("gemini-2.5-flash");
    let topo_id = graph.id.clone();
    engine.cache_topology(graph);
    engine.record_outcome(
        &mut smmu,
        &topo_id,
        "Seed task",
        vec!["seed".into()],
        None,
        0.85,
        0.005,
        150.0,
    );

    // First evolution
    engine.evolve(3, 2);
    let cells_after_first = engine.archive().cell_count();

    // Second evolution
    engine.evolve(3, 2);
    let cells_after_second = engine.archive().cell_count();

    // Should not decrease
    assert!(cells_after_second >= cells_after_first);
}

// ── Bonus Test 16: bandit gets arms from record_outcome ──────────────────

#[test]
fn test_record_outcome_feeds_bandit() {
    let mut engine = TopologyEngine::new();
    let mut smmu = MultiViewMMU::new();

    assert_eq!(engine.bandit().arm_count(), 0);

    let graph = templates::sequential("model-a");
    let topo_id = graph.id.clone();
    engine.cache_topology(graph);

    engine.record_outcome(
        &mut smmu,
        &topo_id,
        "Test task",
        vec!["test".into()],
        None,
        0.9,
        0.01,
        100.0,
    );

    assert!(
        engine.bandit().arm_count() > 0,
        "Bandit should have at least one arm after record_outcome"
    );
}
