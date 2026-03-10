use sage_core::routing::model_card::{CognitiveSystem, ModelCard};

#[test]
fn parse_model_card_from_toml_str() {
    let toml_str = r#"
        [[models]]
        id = "gemini-2.5-flash"
        provider = "google"
        family = "gemini-2.5"
        code_score = 0.85
        reasoning_score = 0.80
        tool_use_score = 0.90
        math_score = 0.75
        formal_z3_strength = 0.60
        cost_input_per_m = 0.075
        cost_output_per_m = 0.30
        latency_ttft_ms = 200.0
        tokens_per_sec = 200.0
        s1_affinity = 0.70
        s2_affinity = 0.85
        s3_affinity = 0.40
        recommended_topologies = ["sequential", "avr"]
        supports_tools = true
        supports_json_mode = true
        supports_vision = true
        context_window = 1048576
    "#;
    let cards = ModelCard::parse_toml(toml_str).unwrap();
    assert_eq!(cards.len(), 1);
    assert_eq!(cards[0].id, "gemini-2.5-flash");
    assert_eq!(cards[0].provider, "google");
    assert!((cards[0].s2_affinity - 0.85).abs() < 0.001);
    assert_eq!(cards[0].context_window, 1048576);
}

#[test]
fn best_system_affinity() {
    let card = ModelCard {
        id: "test".into(),
        provider: "test".into(),
        family: "test".into(),
        code_score: 0.5,
        reasoning_score: 0.5,
        tool_use_score: 0.5,
        math_score: 0.5,
        formal_z3_strength: 0.5,
        cost_input_per_m: 1.0,
        cost_output_per_m: 1.0,
        latency_ttft_ms: 1000.0,
        tokens_per_sec: 100.0,
        s1_affinity: 0.3,
        s2_affinity: 0.9,
        s3_affinity: 0.5,
        recommended_topologies: vec![],
        supports_tools: true,
        supports_json_mode: false,
        supports_vision: false,
        context_window: 128000,
    };
    assert_eq!(card.best_system(), CognitiveSystem::S2);
}

#[test]
fn cognitive_system_display() {
    assert_eq!(format!("{}", CognitiveSystem::S1), "S1");
    assert_eq!(format!("{}", CognitiveSystem::S3), "S3");
}

#[test]
fn estimate_cost_matches_per_million_pricing() {
    let card = ModelCard {
        id: "cost-test".into(),
        provider: "test".into(),
        family: "test".into(),
        code_score: 0.5,
        reasoning_score: 0.5,
        tool_use_score: 0.5,
        math_score: 0.5,
        formal_z3_strength: 0.5,
        cost_input_per_m: 0.075,
        cost_output_per_m: 0.30,
        latency_ttft_ms: 200.0,
        tokens_per_sec: 200.0,
        s1_affinity: 0.5,
        s2_affinity: 0.5,
        s3_affinity: 0.5,
        recommended_topologies: vec![],
        supports_tools: true,
        supports_json_mode: true,
        supports_vision: false,
        context_window: 128000,
    };
    // 1M input tokens at $0.075 + 1M output tokens at $0.30 = $0.375
    let cost = card.estimate_cost(1_000_000, 1_000_000);
    assert!((cost - 0.375).abs() < 0.001);
}

#[test]
fn affinity_for_each_system() {
    let card = ModelCard {
        id: "affinity-test".into(),
        provider: "test".into(),
        family: "test".into(),
        code_score: 0.5,
        reasoning_score: 0.5,
        tool_use_score: 0.5,
        math_score: 0.5,
        formal_z3_strength: 0.5,
        cost_input_per_m: 1.0,
        cost_output_per_m: 1.0,
        latency_ttft_ms: 200.0,
        tokens_per_sec: 100.0,
        s1_affinity: 0.1,
        s2_affinity: 0.5,
        s3_affinity: 0.9,
        recommended_topologies: vec![],
        supports_tools: false,
        supports_json_mode: false,
        supports_vision: false,
        context_window: 32000,
    };
    assert!((card.affinity_for(CognitiveSystem::S1) - 0.1).abs() < 0.001);
    assert!((card.affinity_for(CognitiveSystem::S2) - 0.5).abs() < 0.001);
    assert!((card.affinity_for(CognitiveSystem::S3) - 0.9).abs() < 0.001);
}

#[test]
fn load_real_cards_toml() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/config/cards.toml");
    let cards = ModelCard::load_from_file(path).expect("Failed to load cards.toml");
    assert!(cards.len() >= 18, "Expected at least 18 models, got {}", cards.len());
    for card in &cards {
        assert!(card.s1_affinity >= 0.0 && card.s1_affinity <= 1.0, "Bad s1 for {}", card.id);
        assert!(card.s2_affinity >= 0.0 && card.s2_affinity <= 1.0, "Bad s2 for {}", card.id);
        assert!(card.s3_affinity >= 0.0 && card.s3_affinity <= 1.0, "Bad s3 for {}", card.id);
    }
}
