use sage_core::routing::model_card::CognitiveSystem;
use sage_core::routing::model_registry::ModelRegistry;

fn sample_toml() -> &'static str {
    r#"
    [[models]]
    id = "fast-model"
    provider = "google"
    family = "gemini"
    code_score = 0.6
    reasoning_score = 0.5
    tool_use_score = 0.7
    math_score = 0.4
    formal_z3_strength = 0.2
    cost_input_per_m = 0.01
    cost_output_per_m = 0.05
    latency_ttft_ms = 100.0
    tokens_per_sec = 400.0
    s1_affinity = 0.95
    s2_affinity = 0.30
    s3_affinity = 0.10
    recommended_topologies = ["sequential"]
    supports_tools = true
    supports_json_mode = true
    supports_vision = false
    context_window = 128000

    [[models]]
    id = "code-model"
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
    s1_affinity = 0.20
    s2_affinity = 0.90
    s3_affinity = 0.60
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
    formal_z3_strength = 0.85
    cost_input_per_m = 1.25
    cost_output_per_m = 10.0
    latency_ttft_ms = 3000.0
    tokens_per_sec = 150.0
    s1_affinity = 0.10
    s2_affinity = 0.50
    s3_affinity = 0.95
    recommended_topologies = ["avr", "loop"]
    supports_tools = true
    supports_json_mode = true
    supports_vision = false
    context_window = 2000000
    "#
}

#[test]
fn load_registry_from_toml() {
    let reg = ModelRegistry::from_toml_str(sample_toml()).unwrap();
    assert_eq!(reg.len(), 3);
}

#[test]
fn get_card_by_id() {
    let reg = ModelRegistry::from_toml_str(sample_toml()).unwrap();
    let card = reg.get("code-model").unwrap();
    assert_eq!(card.provider, "openai");
}

#[test]
fn select_best_for_s1() {
    let reg = ModelRegistry::from_toml_str(sample_toml()).unwrap();
    let candidates = reg.select_for_system(CognitiveSystem::S1);
    assert!(!candidates.is_empty());
    assert_eq!(candidates[0].id, "fast-model");
}

#[test]
fn select_best_for_s2() {
    let reg = ModelRegistry::from_toml_str(sample_toml()).unwrap();
    let candidates = reg.select_for_system(CognitiveSystem::S2);
    assert_eq!(candidates[0].id, "code-model");
}

#[test]
fn select_best_for_s3() {
    let reg = ModelRegistry::from_toml_str(sample_toml()).unwrap();
    let candidates = reg.select_for_system(CognitiveSystem::S3);
    assert_eq!(candidates[0].id, "reasoner");
}

#[test]
fn register_and_unregister() {
    let mut reg = ModelRegistry::from_toml_str(sample_toml()).unwrap();
    assert_eq!(reg.len(), 3);
    reg.unregister("fast-model");
    assert_eq!(reg.len(), 2);
    assert!(reg.get("fast-model").is_none());
}
