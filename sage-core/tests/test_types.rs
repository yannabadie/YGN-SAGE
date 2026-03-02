use sage_core::types::*;

#[test]
fn test_agent_config_creation() {
    let config = AgentConfig::new(
        "test-agent".to_string(),
        "claude-opus-4-6".to_string(),
        "You are a helpful assistant.".to_string(),
    );
    assert_eq!(config.name, "test-agent");
    assert_eq!(config.model, "claude-opus-4-6");
    assert!(config.tools.is_empty());
    assert!(config.parent_id.is_none());
}

#[test]
fn test_agent_config_with_tools() {
    let config = AgentConfig::new(
        "coder".to_string(),
        "gpt-5".to_string(),
        "You write code.".to_string(),
    )
    .with_tools(vec!["bash".to_string(), "file_io".to_string()])
    .with_max_steps(50);

    assert_eq!(config.tools.len(), 2);
    assert_eq!(config.max_steps, 50);
}

#[test]
fn test_memory_scope_default() {
    let config = AgentConfig::new(
        "agent".to_string(),
        "model".to_string(),
        "prompt".to_string(),
    );
    assert_eq!(config.memory_scope, MemoryScope::Isolated);
}

#[test]
fn test_agent_config_serialization() {
    let config = AgentConfig::new(
        "agent".to_string(),
        "model".to_string(),
        "prompt".to_string(),
    );
    let json = serde_json::to_string(&config).unwrap();
    let deserialized: AgentConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(config.name, deserialized.name);
    assert_eq!(config.model, deserialized.model);
}

#[test]
fn test_tool_spec_creation() {
    let spec = ToolSpec {
        name: "bash".to_string(),
        description: "Execute commands".to_string(),
        parameters_schema: r#"{"type":"object"}"#.to_string(),
        category: "system".to_string(),
        requires_sandbox: false,
    };
    assert_eq!(spec.name, "bash");
    assert!(!spec.requires_sandbox);
}

#[test]
fn test_topology_role_default() {
    let config = AgentConfig::new("a".into(), "m".into(), "p".into());
    assert_eq!(config.topology_role, TopologyRole::Root);
}

#[test]
fn test_agent_config_builder_chain() {
    let config = AgentConfig::new("agent".into(), "model".into(), "prompt".into())
        .with_tools(vec!["bash".into()])
        .with_max_steps(25)
        .with_parent("parent-id".into())
        .with_memory_scope(MemoryScope::Shared)
        .with_topology_role(TopologyRole::Vertical);

    assert_eq!(config.tools, vec!["bash"]);
    assert_eq!(config.max_steps, 25);
    assert_eq!(config.parent_id, Some("parent-id".to_string()));
    assert_eq!(config.memory_scope, MemoryScope::Shared);
    assert_eq!(config.topology_role, TopologyRole::Vertical);
}
