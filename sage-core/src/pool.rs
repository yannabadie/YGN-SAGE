use dashmap::DashMap;
use pyo3::prelude::*;

use crate::agent::Agent;
use crate::types::*;

/// Thread-safe pool for managing agents
#[pyclass]
pub struct AgentPool {
    agents: DashMap<String, Agent>,
}

#[pymethods]
impl AgentPool {
    #[new]
    pub fn new() -> Self {
        Self {
            agents: DashMap::new(),
        }
    }

    /// Register a new agent from config
    pub fn register(&self, config: AgentConfig) -> String {
        let id = config.id.clone();
        let parent_id = config.parent_id.clone();
        
        let agent = Agent::new(config);
        self.agents.insert(id.clone(), agent);
        
        // Update parent's children list if this agent has a parent
        if let Some(pid) = parent_id {
            if let Some(mut parent) = self.agents.get_mut(&pid) {
                parent.children_ids.push(id.clone());
            }
        }
        
        id
    }

    /// Search agents by name substring match
    pub fn search(&self, query: &str) -> Vec<AgentConfig> {
        let query_lower = query.to_lowercase();
        self.agents
            .iter()
            .filter(|entry| {
                let agent = entry.value();
                agent.config.name.to_lowercase().contains(&query_lower)
                    || agent.config.system_prompt.to_lowercase().contains(&query_lower)
            })
            .map(|entry| entry.value().config.clone())
            .collect()
    }

    /// List all agents
    pub fn list(&self) -> Vec<AgentConfig> {
        self.agents
            .iter()
            .map(|entry| entry.value().config.clone())
            .collect()
    }

    /// Get children config of a specific agent
    pub fn get_children(&self, parent_id: &str) -> Vec<AgentConfig> {
        if let Some(parent) = self.agents.get(parent_id) {
            parent.children_ids.iter()
                .filter_map(|cid| self.agents.get(cid))
                .map(|child| child.value().config.clone())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Terminate an agent by ID
    pub fn terminate(&self, id: &str) -> bool {
        if let Some(mut entry) = self.agents.get_mut(id) {
            entry.status = AgentStatus::Terminated;
            true
        } else {
            false
        }
    }

    /// Get agent count
    pub fn len(&self) -> usize {
        self.agents.len()
    }
}

impl AgentPool {
    /// Get an agent by ID (Rust-only, returns clone)
    pub fn get(&self, id: &str) -> Option<Agent> {
        self.agents.get(id).map(|entry| entry.value().clone())
    }
}
