use sage_core::memory::smmu::MultiViewMMU;
use sage_core::routing::smmu_bridge::{TopologyBridge, TopologyChunk};

// ── Test 1: Record and basic properties ─────────────────────────────────────

#[test]
fn test_record_and_retrieve() {
    let mut smmu = MultiViewMMU::new();
    let mut bridge = TopologyBridge::new();

    // Record an outcome with a 384-dim embedding
    let emb = vec![1.0; 384];
    let chunk_id = bridge.record_outcome(
        &mut smmu,
        TopologyChunk {
            task_summary: "Write a quicksort function".into(),
            keywords: vec!["code".into(), "sort".into(), "algorithm".into()],
            embedding: Some(emb),
            template: "avr".into(),
            model_id: "gemini-2.5-flash".into(),
            quality: 0.9,
            cost: 0.005,
            latency_ms: 1500.0,
        },
    );

    // Valid chunk ID (first chunk is 0)
    assert_eq!(chunk_id, 0);
    assert_eq!(bridge.chunk_count(), 1);
    assert_eq!(smmu.chunk_count(), 1);
}

// ── Test 2: Retrieve similar tasks via semantic edges ────────────────────────

#[test]
fn test_retrieve_similar_tasks() {
    let mut smmu = MultiViewMMU::new();
    let mut bridge = TopologyBridge::new();

    let emb1 = vec![1.0; 384];
    let emb2 = vec![1.0; 384]; // Identical => cosine similarity = 1.0

    bridge.record_outcome(
        &mut smmu,
        TopologyChunk {
            task_summary: "Write a sorting function".into(),
            keywords: vec!["code".into(), "sort".into()],
            embedding: Some(emb1),
            template: "avr".into(),
            model_id: "model-a".into(),
            quality: 0.9,
            cost: 0.01,
            latency_ms: 1000.0,
        },
    );

    // Register a query task directly in S-MMU (simulating a new task arrival)
    let query_id =
        smmu.register_chunk(0, 0, "Sort an array", vec!["sort".into()], Some(emb2), None);

    let results = bridge.retrieve_similar(&smmu, query_id, 5);
    // Should find the similar task via semantic edge
    assert!(
        !results.is_empty(),
        "Should find the similar sorting task via semantic similarity"
    );
    // Verify the returned metadata
    let (template, model_id, quality, score) = &results[0];
    assert_eq!(template, "avr");
    assert_eq!(model_id, "model-a");
    assert!((quality - 0.9).abs() < 1e-5);
    assert!(*score > 0.0, "Similarity score should be positive");
}

// ── Test 3: Empty bridge returns no results ──────────────────────────────────

#[test]
fn test_empty_bridge() {
    let bridge = TopologyBridge::new();
    assert_eq!(bridge.chunk_count(), 0);

    let smmu = MultiViewMMU::new();
    let results = bridge.retrieve_similar(&smmu, 0, 5);
    assert!(results.is_empty());
}

// ── Test 4: Multiple chunks with different topologies ────────────────────────

#[test]
fn test_multiple_chunks() {
    let mut smmu = MultiViewMMU::new();
    let mut bridge = TopologyBridge::new();

    for i in 0..5 {
        bridge.record_outcome(
            &mut smmu,
            TopologyChunk {
                task_summary: format!("Task {}", i),
                keywords: vec![format!("keyword_{}", i)],
                embedding: None,
                template: if i % 2 == 0 {
                    "sequential".into()
                } else {
                    "parallel".into()
                },
                model_id: "model".into(),
                quality: 0.5 + i as f32 * 0.1,
                cost: 0.01,
                latency_ms: 500.0,
            },
        );
    }

    assert_eq!(bridge.chunk_count(), 5);
    assert_eq!(smmu.chunk_count(), 5);
}

// ── Test 5: Retrieve respects max_results ────────────────────────────────────

#[test]
fn test_retrieve_max_results() {
    let mut smmu = MultiViewMMU::new();
    let mut bridge = TopologyBridge::new();

    // Record 10 chunks with identical embeddings (all will be similar)
    let emb = vec![1.0; 384];
    for i in 0..10 {
        bridge.record_outcome(
            &mut smmu,
            TopologyChunk {
                task_summary: format!("Sorting task {}", i),
                keywords: vec!["sort".into()],
                embedding: Some(emb.clone()),
                template: "avr".into(),
                model_id: format!("model-{}", i),
                quality: 0.5 + i as f32 * 0.05,
                cost: 0.01,
                latency_ms: 500.0,
            },
        );
    }

    // Query with max_results=3
    let query_id = smmu.register_chunk(
        0,
        0,
        "Another sort task",
        vec!["sort".into()],
        Some(emb),
        None,
    );

    let results = bridge.retrieve_similar(&smmu, query_id, 3);
    assert!(
        results.len() <= 3,
        "Should return at most 3 results, got {}",
        results.len()
    );
}

// ── Test 6: Tracing compiles and doesn't break SystemRouter ──────────────────

#[test]
fn test_tracing_compiles_system_router() {
    use sage_core::routing::model_registry::ModelRegistry;
    use sage_core::routing::system_router::SystemRouter;

    let registry = ModelRegistry::from_toml_str(
        r#"
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
        s1_affinity = 0.9
        s2_affinity = 0.5
        s3_affinity = 0.3
        recommended_topologies = ["sequential"]
        supports_tools = true
        supports_json_mode = false
        supports_vision = false
        context_window = 128000
    "#,
    )
    .unwrap();

    let router = SystemRouter::new(registry);
    let decision = router.route("Hello", 10.0);
    assert!(!decision.model_id.is_empty());
    assert!(decision.confidence > 0.0);
}

// ── Test 7: Tracing compiles and doesn't break ContextualBandit ──────────────

#[test]
fn test_tracing_compiles_bandit() {
    use sage_core::routing::bandit::ContextualBandit;

    let mut bandit = ContextualBandit::create(0.995, 0.1);
    bandit.add_arm("model-a", "sequential");
    bandit.add_arm("model-b", "avr");

    // choose() has tracing spans
    let decision = bandit.choose(0.0).unwrap();
    assert!(!decision.model_id.is_empty());

    // record_outcome() has tracing spans
    bandit
        .record_outcome(&decision.decision_id, 0.8, 0.01, 200.0)
        .unwrap();

    assert_eq!(bandit.total_observations(), 1);
}

// ── Test 8: Entity-based retrieval via shared keywords ───────────────────────

#[test]
fn test_retrieve_via_shared_keywords() {
    let mut smmu = MultiViewMMU::new();
    let mut bridge = TopologyBridge::new();

    // Record a chunk with keywords
    bridge.record_outcome(
        &mut smmu,
        TopologyChunk {
            task_summary: "Implement a binary search tree".into(),
            keywords: vec!["code".into(), "tree".into(), "search".into()],
            embedding: None,
            template: "avr".into(),
            model_id: "model-a".into(),
            quality: 0.85,
            cost: 0.02,
            latency_ms: 800.0,
        },
    );

    // Register a query chunk with overlapping keywords
    let query_id = smmu.register_chunk(
        0,
        0,
        "Search through a tree structure",
        vec!["tree".into(), "search".into(), "algorithm".into()],
        None,
        None,
    );

    // Retrieve using entity weights
    let weights = [0.0, 0.0, 0.0, 1.0]; // entity-only
    let raw_results = smmu.retrieve_relevant(query_id, 2, weights);

    // The first chunk (ID 0) should be reachable via shared keywords
    let found = raw_results.iter().any(|(id, _)| *id == 0);
    assert!(
        found,
        "Should find the BST task via shared keywords 'tree' and 'search'"
    );
}
