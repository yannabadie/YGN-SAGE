use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Memory scope determines how an agent accesses shared memory.
#[pyclass]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryScope {
    /// Agent has its own isolated memory
    Isolated,
    /// Agent shares memory with parent
    Shared,
    /// Agent inherits a snapshot of parent's memory
    Inherited,
}

impl Default for MemoryScope {
    fn default() -> Self {
        Self::Isolated
    }
}

/// Agent topology type
#[pyclass]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyRole {
    /// Root agent (no parent)
    Root,
    /// Vertical sub-agent (sequential task decomposition)
    Vertical,
    /// Horizontal sub-agent (parallel exploration)
    Horizontal,
    /// Mesh participant (interconnected)
    Mesh,
}

impl Default for TopologyRole {
    fn default() -> Self {
        Self::Root
    }
}

/// Status of an agent in its lifecycle
#[pyclass]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentStatus {
    Created,
    Running,
    Paused,
    Completed,
    Failed,
    Terminated,
}

/// Configuration for creating an agent
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub model: String,
    #[pyo3(get, set)]
    pub system_prompt: String,
    #[pyo3(get, set)]
    pub tools: Vec<String>,
    #[pyo3(get, set)]
    pub memory_scope: MemoryScope,
    #[pyo3(get, set)]
    pub max_steps: u32,
    #[pyo3(get, set)]
    pub parent_id: Option<String>,
    #[pyo3(get, set)]
    pub topology_role: TopologyRole,
    #[pyo3(get)]
    pub created_at: String,
}

#[pymethods]
impl AgentConfig {
    #[new]
    #[pyo3(signature = (name, model, system_prompt))]
    pub fn py_new(name: String, model: String, system_prompt: String) -> Self {
        Self::new(name, model, system_prompt)
    }
}

impl AgentConfig {
    pub fn new(name: String, model: String, system_prompt: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name,
            model,
            system_prompt,
            tools: Vec::new(),
            memory_scope: MemoryScope::default(),
            max_steps: 100,
            parent_id: None,
            topology_role: TopologyRole::default(),
            created_at: Utc::now().to_rfc3339(),
        }
    }

    pub fn with_tools(mut self, tools: Vec<String>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_max_steps(mut self, max_steps: u32) -> Self {
        self.max_steps = max_steps;
        self
    }

    pub fn with_parent(mut self, parent_id: String) -> Self {
        self.parent_id = Some(parent_id);
        self
    }

    pub fn with_memory_scope(mut self, scope: MemoryScope) -> Self {
        self.memory_scope = scope;
        self
    }

    pub fn with_topology_role(mut self, role: TopologyRole) -> Self {
        self.topology_role = role;
        self
    }
}

/// Specification for a tool that an agent can use
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub description: String,
    #[pyo3(get)]
    pub parameters_schema: String,
    #[pyo3(get)]
    pub category: String,
    #[pyo3(get)]
    pub requires_sandbox: bool,
}

#[pymethods]
impl ToolSpec {
    #[new]
    #[pyo3(signature = (name, description, parameters_schema, category="system".to_string(), requires_sandbox=false))]
    pub fn py_new(
        name: String,
        description: String,
        parameters_schema: String,
        category: String,
        requires_sandbox: bool,
    ) -> Self {
        Self { name, description, parameters_schema, category, requires_sandbox }
    }
}

/// A message in the agent's conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub tool_results: Vec<ToolResult>,
    pub timestamp: DateTime<Utc>,
}

/// Role in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A tool call request from the LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub tool_name: String,
    pub arguments: serde_json::Value,
}

/// Result of executing a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: String,
    pub output: String,
    pub is_error: bool,
}
