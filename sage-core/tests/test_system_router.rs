use sage_core::routing::model_card::CognitiveSystem;
use sage_core::routing::model_registry::ModelRegistry;
#[allow(unused_imports)]
use sage_core::routing::system_router::RoutingDecision;
use sage_core::routing::system_router::{RoutingConstraints, SystemRouter};

fn test_registry() -> ModelRegistry {
    ModelRegistry::from_toml_str(
        r#"
    [[models]]
    id = "fast"
    provider = "google"
    family = "gemini"
    code_score = 0.5
    reasoning_score = 0.4
    tool_use_score = 0.6
    math_score = 0.3
    formal_z3_strength = 0.1
    cost_input_per_m = 0.01
    cost_output_per_m = 0.05
    latency_ttft_ms = 100.0
    tokens_per_sec = 400.0
    s1_affinity = 0.95
    s2_affinity = 0.20
    s3_affinity = 0.05
    recommended_topologies = ["sequential"]
    supports_tools = true
    supports_json_mode = true
    supports_vision = false
    context_window = 128000

    [[models]]
    id = "coder"
    provider = "openai"
    family = "gpt-5"
    code_score = 0.9
    reasoning_score = 0.8
    tool_use_score = 0.9
    math_score = 0.7
    formal_z3_strength = 0.5
    cost_input_per_m = 1.75
    cost_output_per_m = 14.0
    latency_ttft_ms = 3000.0
    tokens_per_sec = 100.0
    s1_affinity = 0.10
    s2_affinity = 0.95
    s3_affinity = 0.40
    recommended_topologies = ["avr", "self-moa"]
    supports_tools = true
    supports_json_mode = true
    supports_vision = true
    context_window = 1000000

    [[models]]
    id = "reasoner"
    provider = "google"
    family = "gemini-3.1"
    code_score = 0.8
    reasoning_score = 0.95
    tool_use_score = 0.85
    math_score = 0.9
    formal_z3_strength = 0.9
    cost_input_per_m = 1.25
    cost_output_per_m = 10.0
    latency_ttft_ms = 3000.0
    tokens_per_sec = 150.0
    s1_affinity = 0.05
    s2_affinity = 0.40
    s3_affinity = 0.95
    recommended_topologies = ["avr", "loop"]
    supports_tools = true
    supports_json_mode = true
    supports_vision = false
    context_window = 2000000
    "#,
    )
    .unwrap()
}

// ── Legacy route() tests (backward compat) ───────────────────────────────────

#[test]
fn route_simple_task_to_s1() {
    let reg = test_registry();
    let router = SystemRouter::new(reg);
    let decision = router.route("What is the capital of France?", 10.0);
    assert_eq!(decision.system, CognitiveSystem::S1);
    assert_eq!(decision.model_id, "fast");
}

#[test]
fn route_code_task_to_s2() {
    let reg = test_registry();
    let router = SystemRouter::new(reg);
    let decision = router.route(
        "Write a Python function to sort a list using quicksort",
        10.0,
    );
    assert_eq!(decision.system, CognitiveSystem::S2);
    assert_eq!(decision.model_id, "coder");
}

#[test]
fn route_formal_task_to_s3() {
    let reg = test_registry();
    let router = SystemRouter::new(reg);
    let decision = router.route(
        "Prove by induction that the sum of first n natural numbers is n(n+1)/2",
        10.0,
    );
    assert_eq!(decision.system, CognitiveSystem::S3);
    assert_eq!(decision.model_id, "reasoner");
}

#[test]
fn budget_constraint_downgrades() {
    let reg = test_registry();
    let router = SystemRouter::new(reg);
    let decision = router.route("Write a complex distributed system", 0.001);
    assert_eq!(
        decision.model_id, "fast",
        "Should pick cheapest model under tight budget"
    );
}

#[test]
fn routing_decision_has_confidence() {
    let reg = test_registry();
    let router = SystemRouter::new(reg);
    let decision = router.route("Hello world", 10.0);
    assert!(decision.confidence > 0.0 && decision.confidence <= 1.0);
}

// ── Legacy route() now includes decision_id ──────────────────────────────────

#[test]
fn route_legacy_still_works() {
    let reg = test_registry();
    let router = SystemRouter::new(reg);
    let decision = router.route("What is the capital of France?", 10.0);

    // All existing fields still present
    assert_eq!(decision.system, CognitiveSystem::S1);
    assert_eq!(decision.model_id, "fast");
    assert!(decision.confidence > 0.0);
    assert!(decision.estimated_cost >= 0.0);

    // New decision_id field is a valid ULID (26 chars, Crockford base32)
    assert_eq!(decision.decision_id.len(), 26);
    assert!(decision
        .decision_id
        .chars()
        .all(|c| "0123456789ABCDEFGHJKMNPQRSTVWXYZ".contains(c)));
}

// ── route_constrained() tests ────────────────────────────────────────────────

#[test]
fn route_constrained_decision_has_id() {
    let reg = test_registry();
    let router = SystemRouter::new(reg);
    let constraints = RoutingConstraints::new(0.0, 0.0, 0.0, vec![], String::new(), 0.0);
    let decision = router.route_constrained("Hello world", &constraints);

    // ULID: 26 characters of Crockford base32
    assert_eq!(decision.decision_id.len(), 26);
    assert!(decision
        .decision_id
        .chars()
        .all(|c| "0123456789ABCDEFGHJKMNPQRSTVWXYZ".contains(c)));

    // Two decisions should have distinct IDs
    let decision2 = router.route_constrained("Hello again", &constraints);
    assert_ne!(decision.decision_id, decision2.decision_id);
}

#[test]
fn route_constrained_filters_by_capability() {
    let reg = test_registry();
    let router = SystemRouter::new(reg);

    // Require vision — only "coder" has supports_vision=true
    let constraints = RoutingConstraints::new(
        0.0,
        0.0,
        0.0,
        vec!["vision".into()],
        String::new(),
        0.0,
    );
    let decision = router.route_constrained("What is the capital of France?", &constraints);
    assert_eq!(
        decision.model_id, "coder",
        "Only 'coder' supports vision"
    );
}

#[test]
fn route_constrained_filters_by_latency() {
    let reg = test_registry();
    let router = SystemRouter::new(reg);

    // max_latency_ms = 500 — only "fast" (100ms) passes
    let constraints = RoutingConstraints::new(0.0, 500.0, 0.0, vec![], String::new(), 0.0);
    let decision = router.route_constrained(
        "Write a Python function to sort a list using quicksort",
        &constraints,
    );
    assert_eq!(
        decision.model_id, "fast",
        "Only 'fast' has latency < 500ms"
    );
}

#[test]
fn route_constrained_filters_by_cost() {
    let reg = test_registry();
    let router = SystemRouter::new(reg);

    // Very tight cost: 0.001 USD — only "fast" survives
    // fast cost: (1000 * 0.01 + 2000 * 0.05) / 1_000_000 = 0.00011
    // coder cost: (1000 * 1.75 + 2000 * 14.0) / 1_000_000 = 0.02975
    // reasoner cost: (1000 * 1.25 + 2000 * 10.0) / 1_000_000 = 0.02125
    let constraints = RoutingConstraints::new(0.001, 0.0, 0.0, vec![], String::new(), 0.0);
    let decision = router.route_constrained(
        "Write a Python function to sort a list using quicksort",
        &constraints,
    );
    assert_eq!(
        decision.model_id, "fast",
        "Only 'fast' is cheap enough under $0.001"
    );
}

#[test]
fn route_constrained_filters_by_quality() {
    let reg = test_registry();
    let router = SystemRouter::new(reg);

    // min_quality = 0.85 — "fast" max(code=0.5, reasoning=0.4) = 0.5 fails
    //                       "coder" max(0.9, 0.8) = 0.9 passes
    //                       "reasoner" max(0.8, 0.95) = 0.95 passes
    let constraints = RoutingConstraints::new(0.0, 0.0, 0.85, vec![], String::new(), 0.0);
    let decision = router.route_constrained("What is the capital of France?", &constraints);
    // S1 task but "fast" filtered out; widens to all models, picks from coder/reasoner
    assert!(
        decision.model_id == "coder" || decision.model_id == "reasoner",
        "Should pick a model with quality >= 0.85, got '{}'",
        decision.model_id
    );
}

#[test]
fn route_constrained_widens_on_empty() {
    let reg = test_registry();
    let router = SystemRouter::new(reg);

    // S1 task with vision requirement: "fast" is best S1 but has no vision.
    // Should widen to all models and pick "coder" (only one with vision).
    let constraints = RoutingConstraints::new(
        0.0,
        0.0,
        0.0,
        vec!["vision".into()],
        String::new(),
        0.0,
    );
    let decision = router.route_constrained("What is 2+2?", &constraints);
    assert_eq!(
        decision.model_id, "coder",
        "Should widen beyond S1 candidates and find 'coder' with vision"
    );
}

#[test]
fn route_constrained_unsatisfiable_falls_back() {
    let reg = test_registry();
    let router = SystemRouter::new(reg);

    // Impossible: require vision + latency < 50ms (no model satisfies both)
    // "coder" has vision but latency=3000ms
    // "fast" has latency=100ms but no vision
    let constraints = RoutingConstraints::new(
        0.0,
        50.0,
        0.0,
        vec!["vision".into()],
        String::new(),
        0.0,
    );
    let decision = router.route_constrained("Describe this image", &constraints);

    // When no constraints can be satisfied, falls back to all models.
    // Decision should still return a valid model (not panic).
    assert!(
        !decision.model_id.is_empty(),
        "Should still return a model even when constraints are unsatisfiable"
    );
}

#[test]
fn route_constrained_unconstrained_matches_route() {
    let reg = test_registry();
    let router = SystemRouter::new(reg);

    // With all-zero constraints (unconstrained), route_constrained should
    // pick the same model as route() with unlimited budget.
    let unconstrained = RoutingConstraints::new(0.0, 0.0, 0.0, vec![], String::new(), 0.0);

    let task = "What is the capital of France?";
    let legacy = router.route(task, f32::MAX);
    let constrained = router.route_constrained(task, &unconstrained);

    // Both should route to the same cognitive system
    assert_eq!(legacy.system, CognitiveSystem::S1);
    // Both should pick "fast" for S1 tasks
    assert_eq!(legacy.model_id, "fast");
    assert_eq!(constrained.model_id, "fast");
}

// ── RoutingConstraints construction / repr ───────────────────────────────────

#[test]
fn routing_constraints_defaults() {
    let c = RoutingConstraints::new(0.0, 0.0, 0.0, vec![], String::new(), 0.0);
    assert_eq!(c.max_cost_usd, 0.0);
    assert_eq!(c.max_latency_ms, 0.0);
    assert_eq!(c.min_quality, 0.0);
    assert!(c.required_capabilities.is_empty());
    assert!(c.security_label.is_empty());
    assert_eq!(c.exploration_budget, 0.0);
}

#[test]
fn routing_constraints_debug() {
    let c = RoutingConstraints::new(
        0.5,
        200.0,
        0.8,
        vec!["tools".into(), "vision".into()],
        "confidential".into(),
        0.1,
    );
    let debug = format!("{:?}", c);
    assert!(debug.contains("RoutingConstraints"));
    assert!(debug.contains("max_cost_usd: 0.5"));
    assert!(debug.contains("max_latency_ms: 200.0"));
    assert!(debug.contains("min_quality: 0.8"));
    assert!(debug.contains("tools"));
    assert!(debug.contains("vision"));
    assert!(debug.contains("confidential"));
    assert!(debug.contains("exploration_budget: 0.1"));
}
