use sage_core::types::*;
use sage_core::pool::AgentPool;

#[test]
fn test_create_and_get_agent() {
    let pool = AgentPool::new();
    let config = AgentConfig::new(
        "test-agent".to_string(),
        "claude-opus-4-6".to_string(),
        "You help.".to_string(),
    );
    let id = config.id.clone();
    pool.register(config);

    let agent = pool.get(&id);
    assert!(agent.is_some());
    assert_eq!(agent.unwrap().config.name, "test-agent");
}

#[test]
fn test_search_agents_by_name() {
    let pool = AgentPool::new();
    pool.register(AgentConfig::new("coder".into(), "m".into(), "p".into()));
    pool.register(AgentConfig::new("debugger".into(), "m".into(), "p".into()));
    pool.register(AgentConfig::new("code-reviewer".into(), "m".into(), "p".into()));

    let results = pool.search("code");
    assert_eq!(results.len(), 2); // "coder" and "code-reviewer"
}

#[test]
fn test_list_agents() {
    let pool = AgentPool::new();
    pool.register(AgentConfig::new("a".into(), "m".into(), "p".into()));
    pool.register(AgentConfig::new("b".into(), "m".into(), "p".into()));

    assert_eq!(pool.list().len(), 2);
}

#[test]
fn test_terminate_agent() {
    let pool = AgentPool::new();
    let config = AgentConfig::new("a".into(), "m".into(), "p".into());
    let id = config.id.clone();
    pool.register(config);

    assert!(pool.terminate(&id));
    let agent = pool.get(&id).unwrap();
    assert_eq!(agent.status, AgentStatus::Terminated);
}

#[test]
fn test_terminate_nonexistent() {
    let pool = AgentPool::new();
    assert!(!pool.terminate("nonexistent"));
}
