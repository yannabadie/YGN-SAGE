use sage_core::routing::model_card::CognitiveSystem;
use sage_core::routing::model_registry::ModelRegistry;
#[allow(unused_imports)]
use sage_core::routing::system_router::RoutingDecision;
use sage_core::routing::system_router::SystemRouter;

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
