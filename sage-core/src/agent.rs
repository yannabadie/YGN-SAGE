use crate::types::*;
use serde::{Deserialize, Serialize};

/// Runtime representation of an agent in the pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub config: AgentConfig,
    pub status: AgentStatus,
    pub step_count: u32,
    pub result: Option<String>,
    pub children_ids: Vec<String>,
}

impl Agent {
    pub fn new(config: AgentConfig) -> Self {
        Self {
            config,
            status: AgentStatus::Created,
            step_count: 0,
            result: None,
            children_ids: Vec::new(),
        }
    }
}
