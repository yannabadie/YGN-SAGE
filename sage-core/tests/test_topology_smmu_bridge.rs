use sage_core::memory::smmu::MultiViewMMU;
use sage_core::routing::bandit::ContextualBandit;
use sage_core::topology::smmu_bridge::{TopologyOutcome, TopologySmmuBridge, TopologySuggestion};

// ── Helper ──────────────────────────────────────────────────────────────────

fn make_outcome(
    topology_id: &str,
    summary: &str,
    template: &str,
    quality: f32,
    embedding: Option<Vec<f32>>,
) -> TopologyOutcome {
    TopologyOutcome {
        topology_id: topology_id.to_string(),
        task_summary: summary.to_string(),
        keywords: vec!["code".into(), "test".into()],
        task_embedding: embedding,
        template: template.to_string(),
        quality,
        cost: 0.01,
        latency_ms: 500.0,
        agent_count: 3,
        max_depth: 2,
        model_diversity: 0.67,
    }
}

// ── Test 1: New bridge is empty ─────────────────────────────────────────────

#[test]
fn test_new_bridge_is_empty() {
    let bridge = TopologySmmuBridge::new();
    assert_eq!(bridge.chunk_count(), 0);

    let bridge_default = TopologySmmuBridge::default();
    assert_eq!(bridge_default.chunk_count(), 0);
}

// ── Test 2: Record outcome increments count ─────────────────────────────────

#[test]
fn test_record_outcome_increments_count() {
    let mut smmu = MultiViewMMU::new();
    let mut bridge = TopologySmmuBridge::new();

    let outcome = make_outcome("01JTEST0001", "Sort an array", "avr", 0.9, None);
    bridge.record_outcome(&mut smmu, outcome);

    assert_eq!(bridge.chunk_count(), 1);
    assert_eq!(smmu.chunk_count(), 1);
}

// ── Test 3: Record returns valid chunk ID ───────────────────────────────────

#[test]
fn test_record_returns_valid_chunk_id() {
    let mut smmu = MultiViewMMU::new();
    let mut bridge = TopologySmmuBridge::new();

    let outcome = make_outcome(
        "01JTEST0001",
        "Write quicksort",
        "sequential",
        0.8,
        Some(vec![1.0; 384]),
    );
    let id = bridge.record_outcome(&mut smmu, outcome);

    // First chunk should be ID 0
    assert_eq!(id, 0);
}

// ── Test 4: Multiple records ────────────────────────────────────────────────

#[test]
fn test_multiple_records() {
    let mut smmu = MultiViewMMU::new();
    let mut bridge = TopologySmmuBridge::new();

    for i in 0..5 {
        let outcome = make_outcome(
            &format!("01JTEST{:04}", i),
            &format!("Task {}", i),
            if i % 2 == 0 { "sequential" } else { "parallel" },
            0.5 + i as f32 * 0.1,
            None,
        );
        let id = bridge.record_outcome(&mut smmu, outcome);
        assert_eq!(id, i);
    }

    assert_eq!(bridge.chunk_count(), 5);
    assert_eq!(smmu.chunk_count(), 5);
}

// ── Test 5: Retrieve on empty bridge returns empty ──────────────────────────

#[test]
fn test_retrieve_on_empty_bridge() {
    let smmu = MultiViewMMU::new();
    let bridge = TopologySmmuBridge::new();

    let results = bridge.retrieve_similar(&smmu, 0, 5);
    assert!(results.is_empty());
}

// ── Test 6: Retrieve with similar embeddings finds results ──────────────────

#[test]
fn test_retrieve_with_similar_embeddings() {
    let mut smmu = MultiViewMMU::new();
    let mut bridge = TopologySmmuBridge::new();

    let emb1 = vec![1.0; 384];
    let emb2 = vec![1.0; 384]; // Identical => cosine similarity = 1.0

    let outcome = make_outcome(
        "01JTEST_SORT",
        "Write a sorting function",
        "avr",
        0.9,
        Some(emb1),
    );
    bridge.record_outcome(&mut smmu, outcome);

    // Register a query chunk directly in S-MMU
    let query_id = smmu.register_chunk(
        0,
        0,
        "Sort an array",
        vec!["sort".into()],
        Some(emb2),
        None,
    );

    let results = bridge.retrieve_similar(&smmu, query_id, 5);
    assert!(!results.is_empty(), "Should find the similar task");

    let first = &results[0];
    assert_eq!(first.topology_id, "01JTEST_SORT");
    assert_eq!(first.template, "avr");
    assert!((first.quality - 0.9).abs() < 1e-5);
    assert!(first.similarity_score > 0.0);
    assert_eq!(first.agent_count, 3);
    assert_eq!(first.max_depth, 2);
}

// ── Test 7: Structural features stored correctly ────────────────────────────

#[test]
fn test_structural_features_stored() {
    let mut smmu = MultiViewMMU::new();
    let mut bridge = TopologySmmuBridge::new();

    let outcome = TopologyOutcome {
        topology_id: "01JTEST_STRUCT".to_string(),
        task_summary: "Complex multi-agent pipeline".to_string(),
        keywords: vec!["pipeline".into()],
        task_embedding: None,
        template: "parallel".to_string(),
        quality: 0.85,
        cost: 0.05,
        latency_ms: 2000.0,
        agent_count: 5,
        max_depth: 3,
        model_diversity: 0.8,
    };
    let chunk_id = bridge.record_outcome(&mut smmu, outcome);

    let meta = bridge.get_meta(chunk_id).expect("meta should exist");
    assert_eq!(meta.topology_id, "01JTEST_STRUCT");
    assert_eq!(meta.template, "parallel");
    assert_eq!(meta.agent_count, 5);
    assert_eq!(meta.max_depth, 3);
    assert!((meta.model_diversity - 0.8).abs() < 1e-5);
    assert!((meta.quality - 0.85).abs() < 1e-5);
    assert!((meta.cost - 0.05).abs() < 1e-5);
    assert!((meta.latency_ms - 2000.0).abs() < 1e-1);
}

// ── Test 8: Bandit prior injection adds arms ────────────────────────────────

#[test]
fn test_inject_priors_adds_arms() {
    let mut bandit = ContextualBandit::create(0.995, 0.1);
    let bridge = TopologySmmuBridge::new();

    assert_eq!(bandit.arm_count(), 0);

    let suggestions = vec![
        TopologySuggestion {
            topology_id: "01JTEST0001".to_string(),
            template: "avr".to_string(),
            quality: 0.9,
            cost: 0.01,
            latency_ms: 500.0,
            similarity_score: 0.95,
            agent_count: 3,
            max_depth: 2,
            model_diversity: 0.67,
        },
        TopologySuggestion {
            topology_id: "01JTEST0002".to_string(),
            template: "parallel".to_string(),
            quality: 0.8,
            cost: 0.02,
            latency_ms: 800.0,
            similarity_score: 0.85,
            agent_count: 4,
            max_depth: 2,
            model_diversity: 0.75,
        },
    ];

    bridge.inject_priors(&mut bandit, &suggestions);

    // Both suggestions have quality >= 0.3, so both templates should be added
    assert_eq!(bandit.arm_count(), 2);
}

// ── Test 9: Prior injection filters low-quality suggestions ─────────────────

#[test]
fn test_inject_priors_filters_low_quality() {
    let mut bandit = ContextualBandit::create(0.995, 0.1);
    let bridge = TopologySmmuBridge::new();

    let suggestions = vec![
        TopologySuggestion {
            topology_id: "01JTEST_GOOD".to_string(),
            template: "avr".to_string(),
            quality: 0.9,
            cost: 0.01,
            latency_ms: 500.0,
            similarity_score: 0.95,
            agent_count: 3,
            max_depth: 2,
            model_diversity: 0.67,
        },
        TopologySuggestion {
            topology_id: "01JTEST_BAD".to_string(),
            template: "sequential".to_string(),
            quality: 0.1, // Below 0.3 threshold
            cost: 0.05,
            latency_ms: 3000.0,
            similarity_score: 0.7,
            agent_count: 1,
            max_depth: 1,
            model_diversity: 0.0,
        },
    ];

    bridge.inject_priors(&mut bandit, &suggestions);

    // Only the good suggestion should be injected (quality >= 0.3)
    assert_eq!(bandit.arm_count(), 1);
}

// ── Test 10: End-to-end record -> retrieve -> inject ────────────────────────

#[test]
fn test_end_to_end_flow() {
    let mut smmu = MultiViewMMU::new();
    let mut bridge = TopologySmmuBridge::new();

    // Step 1: Record an outcome with embedding
    let emb = vec![1.0; 384];
    let outcome = TopologyOutcome {
        topology_id: "01JTEST_E2E".to_string(),
        task_summary: "Implement quicksort with edge cases".to_string(),
        keywords: vec!["code".into(), "sort".into(), "algorithm".into()],
        task_embedding: Some(emb.clone()),
        template: "avr".to_string(),
        quality: 0.92,
        cost: 0.03,
        latency_ms: 1200.0,
        agent_count: 4,
        max_depth: 3,
        model_diversity: 0.75,
    };
    bridge.record_outcome(&mut smmu, outcome);

    // Step 2: Query with a similar embedding
    let query_id = smmu.register_chunk(
        0,
        0,
        "Sort a list of numbers",
        vec!["sort".into(), "code".into()],
        Some(emb),
        None,
    );

    let suggestions = bridge.retrieve_similar(&smmu, query_id, 5);
    assert!(!suggestions.is_empty(), "Should find similar task");

    // Verify suggestion fields
    let s = &suggestions[0];
    assert_eq!(s.topology_id, "01JTEST_E2E");
    assert_eq!(s.template, "avr");
    assert!((s.quality - 0.92).abs() < 1e-5);
    assert_eq!(s.agent_count, 4);
    assert_eq!(s.max_depth, 3);

    // Step 3: Inject priors into bandit
    let mut bandit = ContextualBandit::create(0.995, 0.1);
    bridge.inject_priors(&mut bandit, &suggestions);
    assert!(
        bandit.arm_count() >= 1,
        "Bandit should have at least 1 arm after injection"
    );
}
