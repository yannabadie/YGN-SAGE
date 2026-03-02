# YGN-SAGE Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build YGN-SAGE, a next-generation ADK with 5 cognitive pillars (Topology, Tools, Memory, Evolution, Strategy) + a flagship Research & Discovery agent.

**Architecture:** Rust core orchestrator (`sage-core`) exposed to Python via PyO3 bindings, a Python SDK (`sage-python`) for building agents with multi-provider LLM support, and a flagship agent (`sage-discover`). Incremental build: foundation first, then pillars one by one.

**Tech Stack:** Rust 1.90+, Python 3.13+, PyO3, Neo4j, Qdrant, Docker, gRPC, asyncio, httpx

---

## Phase 1: Project Foundation & Rust Core

### Task 1: Initialize Rust workspace and Python project

**Files:**
- Create: `Cargo.toml` (workspace root)
- Create: `sage-core/Cargo.toml`
- Create: `sage-core/src/lib.rs`
- Create: `sage-python/pyproject.toml`
- Create: `sage-python/src/sage/__init__.py`
- Create: `sage-discover/pyproject.toml`
- Create: `sage-discover/src/discover/__init__.py`
- Create: `.gitignore`
- Create: `CLAUDE.md`

**Step 1: Initialize git repository**

```bash
cd C:/Code/YGN-SAGE
git init
```

**Step 2: Create workspace Cargo.toml**

```toml
# Cargo.toml (workspace root)
[workspace]
members = ["sage-core"]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2024"
license = "MIT"
```

**Step 3: Create sage-core Rust crate**

```bash
mkdir -p sage-core/src
```

```toml
# sage-core/Cargo.toml
[package]
name = "sage-core"
version.workspace = true
edition.workspace = true

[lib]
name = "sage_core"
crate-type = ["cdylib", "rlib"]

[dependencies]
pyo3 = { version = "0.24", features = ["extension-module"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["full"] }
uuid = { version = "1", features = ["v4", "serde"] }
thiserror = "2"
tracing = "0.1"
tracing-subscriber = "0.3"
dashmap = "6"
petgraph = "0.7"
chrono = { version = "0.4", features = ["serde"] }
```

```rust
// sage-core/src/lib.rs
use pyo3::prelude::*;

pub mod types;
pub mod agent;
pub mod pool;
pub mod memory;

#[pymodule]
fn sage_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<types::AgentConfig>()?;
    m.add_class::<pool::AgentPool>()?;
    Ok(())
}
```

**Step 4: Create sage-python package**

```bash
mkdir -p sage-python/src/sage
mkdir -p sage-python/tests
```

```toml
# sage-python/pyproject.toml
[project]
name = "ygn-sage"
version = "0.1.0"
description = "YGN-SAGE: Self-Adaptive Generation Engine"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.28",
    "pydantic>=2.10",
    "rich>=13",
    "anyio>=4",
]

[project.optional-dependencies]
anthropic = ["anthropic>=0.52"]
openai = ["openai>=1.82"]
google = ["google-genai>=1"]
all = ["ygn-sage[anthropic,openai,google]"]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.25",
    "pytest-cov>=6",
    "ruff>=0.11",
    "mypy>=1.15",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/sage"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
target-version = "py312"
line-length = 100
```

```python
# sage-python/src/sage/__init__.py
"""YGN-SAGE: Self-Adaptive Generation Engine."""

__version__ = "0.1.0"
```

**Step 5: Create sage-discover package**

```bash
mkdir -p sage-discover/src/discover
mkdir -p sage-discover/tests
```

```toml
# sage-discover/pyproject.toml
[project]
name = "sage-discover"
version = "0.1.0"
description = "YGN-SAGE Flagship Research & Discovery Agent"
requires-python = ">=3.12"
dependencies = [
    "ygn-sage[all]>=0.1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/discover"]
```

```python
# sage-discover/src/discover/__init__.py
"""sage-discover: Flagship Research & Discovery Agent."""

__version__ = "0.1.0"
```

**Step 6: Create .gitignore**

```gitignore
# Rust
target/
*.rs.bk

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.venv/
venv/

# IDE
.idea/
.vscode/
*.swp

# Environment
.env
.env.local

# OS
Thumbs.db
.DS_Store

# Docker
docker-compose.override.yml

# Memory databases
*.db
neo4j-data/
qdrant-data/
```

**Step 7: Create CLAUDE.md**

```markdown
# YGN-SAGE - CLAUDE.md

## Project Overview
YGN-SAGE (Yann's Generative Neural Self-Adaptive Generation Engine) is a next-generation
Agent Development Kit built on 5 cognitive pillars: Topology, Tools, Memory, Evolution, Strategy.

## Architecture
- `sage-core/` - Rust orchestrator (PyO3 bindings to Python)
- `sage-python/` - Python SDK for building agents
- `sage-discover/` - Flagship Research & Discovery agent
- `docs/plans/` - Architecture and implementation plans

## Development Commands

### Rust Core
```bash
cargo build                    # Build Rust core
cargo test                     # Run Rust tests
cargo clippy                   # Lint Rust code
```

### Python SDK
```bash
cd sage-python
pip install -e ".[dev]"        # Install in dev mode
pytest                         # Run tests
ruff check src/                # Lint
mypy src/                      # Type check
```

### Full Build
```bash
cargo build --release          # Build Rust
cd sage-python && pip install -e ".[all,dev]"
```

## Tech Stack
- Rust 1.90+ (orchestrator)
- Python 3.13+ (SDK, agents)
- PyO3 (Rust-Python bridge)
- Neo4j (graph memory)
- Qdrant (vector memory)
- Docker (sandboxing)
```

**Step 8: Commit initial scaffold**

```bash
git add -A
git commit -m "feat: initialize YGN-SAGE workspace scaffold

- Rust workspace with sage-core crate (PyO3, tokio, petgraph)
- Python SDK package (sage-python) with multi-provider LLM deps
- Flagship agent package (sage-discover)
- Project documentation and .gitignore"
```

---

### Task 2: Core types in Rust (AgentConfig, ToolSpec, MemoryScope)

**Files:**
- Create: `sage-core/src/types.rs`
- Create: `sage-core/tests/test_types.rs`

**Step 1: Write tests for core types**

```rust
// sage-core/tests/test_types.rs
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
}
```

**Step 2: Run tests to verify they fail**

```bash
cd C:/Code/YGN-SAGE
cargo test
```

Expected: FAIL with compilation error (types module doesn't exist yet)

**Step 3: Implement core types**

```rust
// sage-core/src/types.rs
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub tool_name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: String,
    pub output: String,
    pub is_error: bool,
}
```

**Step 4: Run tests to verify they pass**

```bash
cargo test
```

Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add sage-core/src/types.rs sage-core/tests/test_types.rs sage-core/src/lib.rs
git commit -m "feat(core): add core types - AgentConfig, ToolSpec, MemoryScope, Message"
```

---

### Task 3: Agent Pool in Rust (create, search, run, terminate)

**Files:**
- Create: `sage-core/src/agent.rs`
- Create: `sage-core/src/pool.rs`
- Create: `sage-core/tests/test_pool.rs`

**Step 1: Write tests for agent pool**

```rust
// sage-core/tests/test_pool.rs
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
```

**Step 2: Run tests to verify they fail**

```bash
cargo test
```

Expected: FAIL (modules don't exist)

**Step 3: Implement Agent struct**

```rust
// sage-core/src/agent.rs
use crate::types::*;
use serde::{Deserialize, Serialize};

/// Runtime representation of an agent in the pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub config: AgentConfig,
    pub status: AgentStatus,
    pub step_count: u32,
    pub result: Option<String>,
}

impl Agent {
    pub fn new(config: AgentConfig) -> Self {
        Self {
            config,
            status: AgentStatus::Created,
            step_count: 0,
            result: None,
        }
    }
}
```

**Step 4: Implement AgentPool**

```rust
// sage-core/src/pool.rs
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
        let agent = Agent::new(config);
        self.agents.insert(id.clone(), agent);
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
```

**Step 5: Update lib.rs to include new modules**

```rust
// sage-core/src/lib.rs
use pyo3::prelude::*;

pub mod types;
pub mod agent;
pub mod pool;
pub mod memory;

#[pymodule]
fn sage_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<types::AgentConfig>()?;
    m.add_class::<types::ToolSpec>()?;
    m.add_class::<types::MemoryScope>()?;
    m.add_class::<types::AgentStatus>()?;
    m.add_class::<types::TopologyRole>()?;
    m.add_class::<pool::AgentPool>()?;
    Ok(())
}
```

**Step 6: Create empty memory module (to satisfy lib.rs)**

```rust
// sage-core/src/memory.rs
// Working memory graph - implemented in Task 4
```

**Step 7: Run tests to verify they pass**

```bash
cargo test
```

Expected: All tests PASS

**Step 8: Commit**

```bash
git add sage-core/src/agent.rs sage-core/src/pool.rs sage-core/src/memory.rs sage-core/tests/test_pool.rs sage-core/src/lib.rs
git commit -m "feat(core): add Agent struct and thread-safe AgentPool with search/terminate"
```

---

### Task 4: Working Memory Graph (in-memory, Rust)

**Files:**
- Modify: `sage-core/src/memory.rs`
- Create: `sage-core/tests/test_memory.rs`

**Step 1: Write tests for working memory**

```rust
// sage-core/tests/test_memory.rs
use sage_core::memory::WorkingMemory;

#[test]
fn test_add_and_get_event() {
    let mut mem = WorkingMemory::new("agent-1".to_string());

    let event_id = mem.add_event("tool_call", "Called bash with 'ls'");
    let event = mem.get_event(&event_id);

    assert!(event.is_some());
    let e = event.unwrap();
    assert_eq!(e.event_type, "tool_call");
    assert_eq!(e.content, "Called bash with 'ls'");
}

#[test]
fn test_add_child_agent() {
    let mut mem = WorkingMemory::new("parent".to_string());
    mem.add_child_agent("child-1".to_string());

    let children = mem.child_agents();
    assert_eq!(children.len(), 1);
    assert_eq!(children[0], "child-1");
}

#[test]
fn test_get_recent_events() {
    let mut mem = WorkingMemory::new("agent".to_string());
    for i in 0..10 {
        mem.add_event("step", &format!("Step {i}"));
    }

    let recent = mem.recent_events(3);
    assert_eq!(recent.len(), 3);
    assert_eq!(recent[0].content, "Step 7");
    assert_eq!(recent[2].content, "Step 9");
}

#[test]
fn test_summarize_compresses() {
    let mut mem = WorkingMemory::new("agent".to_string());
    for i in 0..20 {
        mem.add_event("step", &format!("Step {i}"));
    }

    assert_eq!(mem.event_count(), 20);
    mem.compress_old_events(5, "Summary of steps 0-14");
    // Should have 5 recent events + 1 summary event
    assert_eq!(mem.event_count(), 6);
}
```

**Step 2: Run tests to verify they fail**

```bash
cargo test test_memory
```

Expected: FAIL

**Step 3: Implement WorkingMemory**

```rust
// sage-core/src/memory.rs
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// A single event in working memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEvent {
    pub id: String,
    pub event_type: String,
    pub content: String,
    pub timestamp: DateTime<Utc>,
    pub is_summary: bool,
}

impl MemoryEvent {
    pub fn new(event_type: &str, content: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            event_type: event_type.to_string(),
            content: content.to_string(),
            timestamp: Utc::now(),
            is_summary: false,
        }
    }

    pub fn summary(content: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            event_type: "summary".to_string(),
            content: content.to_string(),
            timestamp: Utc::now(),
            is_summary: true,
        }
    }
}

/// In-memory working memory for a single agent execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemory {
    pub agent_id: String,
    events: Vec<MemoryEvent>,
    children: Vec<String>,
}

impl WorkingMemory {
    pub fn new(agent_id: String) -> Self {
        Self {
            agent_id,
            events: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Add an event and return its ID
    pub fn add_event(&mut self, event_type: &str, content: &str) -> String {
        let event = MemoryEvent::new(event_type, content);
        let id = event.id.clone();
        self.events.push(event);
        id
    }

    /// Get an event by ID
    pub fn get_event(&self, id: &str) -> Option<&MemoryEvent> {
        self.events.iter().find(|e| e.id == id)
    }

    /// Get recent N events (oldest first within the window)
    pub fn recent_events(&self, n: usize) -> Vec<&MemoryEvent> {
        let start = self.events.len().saturating_sub(n);
        self.events[start..].iter().collect()
    }

    /// Total event count
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Register a child agent
    pub fn add_child_agent(&mut self, child_id: String) {
        self.children.push(child_id);
    }

    /// Get child agent IDs
    pub fn child_agents(&self) -> &[String] {
        &self.children
    }

    /// Compress old events: keep the last `keep_recent` events,
    /// replace everything before with a single summary event.
    pub fn compress_old_events(&mut self, keep_recent: usize, summary_text: &str) {
        if self.events.len() <= keep_recent {
            return;
        }
        let split_point = self.events.len() - keep_recent;
        let recent: Vec<MemoryEvent> = self.events.drain(split_point..).collect();
        self.events.clear();
        self.events.push(MemoryEvent::summary(summary_text));
        self.events.extend(recent);
    }
}
```

**Step 4: Run tests to verify they pass**

```bash
cargo test test_memory
```

Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add sage-core/src/memory.rs sage-core/tests/test_memory.rs
git commit -m "feat(core): add WorkingMemory graph with events, children, compression"
```

---

## Phase 2: Python SDK (sage-python)

### Task 5: LLM Abstraction Layer (multi-provider)

**Files:**
- Create: `sage-python/src/sage/llm/__init__.py`
- Create: `sage-python/src/sage/llm/base.py`
- Create: `sage-python/src/sage/llm/anthropic.py`
- Create: `sage-python/src/sage/llm/openai.py`
- Create: `sage-python/src/sage/llm/google.py`
- Create: `sage-python/src/sage/llm/registry.py`
- Create: `sage-python/tests/test_llm.py`

**Step 1: Write tests for LLM abstraction**

```python
# sage-python/tests/test_llm.py
"""Tests for the LLM abstraction layer."""
import pytest
from sage.llm.base import LLMConfig, Message, ToolDef, LLMResponse, Role
from sage.llm.registry import LLMRegistry


def test_message_creation():
    msg = Message(role=Role.USER, content="Hello")
    assert msg.role == Role.USER
    assert msg.content == "Hello"


def test_llm_config():
    config = LLMConfig(
        provider="anthropic",
        model="claude-opus-4-6",
        max_tokens=4096,
        temperature=0.7,
    )
    assert config.provider == "anthropic"
    assert config.model == "claude-opus-4-6"


def test_tool_def():
    tool = ToolDef(
        name="bash",
        description="Execute a bash command",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to run"}
            },
            "required": ["command"],
        },
    )
    assert tool.name == "bash"


def test_registry_register_and_get():
    registry = LLMRegistry()

    class FakeProvider:
        name = "fake"
        async def generate(self, messages, tools, config):
            return LLMResponse(content="fake response", tool_calls=[])

    registry.register("fake", FakeProvider)
    provider_cls = registry.get("fake")
    assert provider_cls is not None
    assert provider_cls.name == "fake"


def test_registry_unknown_provider():
    registry = LLMRegistry()
    with pytest.raises(KeyError):
        registry.get("nonexistent")


def test_llm_config_defaults():
    config = LLMConfig(provider="openai", model="gpt-5")
    assert config.max_tokens == 8192
    assert config.temperature == 0.0
```

**Step 2: Run tests to verify they fail**

```bash
cd C:/Code/YGN-SAGE/sage-python
pip install -e ".[dev]"
pytest tests/test_llm.py -v
```

Expected: FAIL (modules don't exist)

**Step 3: Implement LLM base types**

```python
# sage-python/src/sage/llm/__init__.py
"""LLM abstraction layer with multi-provider support."""
from sage.llm.base import LLMConfig, Message, Role, ToolDef, ToolCall, LLMResponse
from sage.llm.registry import LLMRegistry

__all__ = [
    "LLMConfig",
    "Message",
    "Role",
    "ToolDef",
    "ToolCall",
    "LLMResponse",
    "LLMRegistry",
]
```

```python
# sage-python/src/sage/llm/base.py
"""Base types for the LLM abstraction layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    role: Role
    content: str
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] | None = None
    model: str | None = None
    stop_reason: str | None = None


@dataclass
class LLMConfig:
    provider: str
    model: str
    max_tokens: int = 8192
    temperature: float = 0.0
    top_p: float = 1.0
    api_key: str | None = None
    base_url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol that all LLM providers must implement."""

    name: str

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        config: LLMConfig | None = None,
    ) -> LLMResponse: ...
```

```python
# sage-python/src/sage/llm/registry.py
"""Registry for LLM providers."""
from __future__ import annotations

from typing import Any


class LLMRegistry:
    """Central registry for LLM provider classes."""

    def __init__(self) -> None:
        self._providers: dict[str, Any] = {}

    def register(self, name: str, provider_cls: Any) -> None:
        """Register a provider class by name."""
        self._providers[name] = provider_cls

    def get(self, name: str) -> Any:
        """Get a provider class by name. Raises KeyError if not found."""
        if name not in self._providers:
            raise KeyError(f"Unknown LLM provider: {name!r}. Available: {list(self._providers)}")
        return self._providers[name]

    def list_providers(self) -> list[str]:
        """List all registered provider names."""
        return list(self._providers.keys())


# Global registry instance
default_registry = LLMRegistry()
```

**Step 4: Create provider stubs (Anthropic, OpenAI, Google)**

```python
# sage-python/src/sage/llm/anthropic.py
"""Anthropic Claude provider."""
from __future__ import annotations

import os
from sage.llm.base import LLMConfig, LLMProvider, LLMResponse, Message, ToolCall, ToolDef


class AnthropicProvider:
    """LLM provider for Anthropic Claude models."""

    name = "anthropic"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install 'ygn-sage[anthropic]'")

        client = anthropic.AsyncAnthropic(api_key=self.api_key)

        # Convert messages to Anthropic format
        system_msg = ""
        api_messages = []
        for msg in messages:
            if msg.role.value == "system":
                system_msg = msg.content
            else:
                api_messages.append({"role": msg.role.value, "content": msg.content})

        model = config.model if config else "claude-sonnet-4-6"
        max_tokens = config.max_tokens if config else 8192

        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": api_messages,
        }
        if system_msg:
            kwargs["system"] = system_msg
        if tools:
            kwargs["tools"] = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]

        response = await client.messages.create(**kwargs)

        # Extract content and tool calls
        content_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input)
                )

        return LLMResponse(
            content="\n".join(content_parts),
            tool_calls=tool_calls,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            model=response.model,
            stop_reason=response.stop_reason,
        )
```

```python
# sage-python/src/sage/llm/openai.py
"""OpenAI GPT provider."""
from __future__ import annotations

import json
import os
from sage.llm.base import LLMConfig, LLMResponse, Message, ToolCall, ToolDef


class OpenAIProvider:
    """LLM provider for OpenAI GPT models."""

    name = "openai"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Install openai: pip install 'ygn-sage[openai]'")

        client = AsyncOpenAI(api_key=self.api_key)

        model = config.model if config else "gpt-4o"
        max_tokens = config.max_tokens if config else 8192

        api_messages = [{"role": m.role.value, "content": m.content} for m in messages]

        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": api_messages,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            usage={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            } if response.usage else None,
            model=response.model,
            stop_reason=choice.finish_reason,
        )
```

```python
# sage-python/src/sage/llm/google.py
"""Google Gemini provider."""
from __future__ import annotations

import os
from sage.llm.base import LLMConfig, LLMResponse, Message, ToolCall, ToolDef


class GoogleProvider:
    """LLM provider for Google Gemini models."""

    name = "google"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolDef] | None = None,
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        try:
            from google import genai
        except ImportError:
            raise ImportError("Install google-genai: pip install 'ygn-sage[google]'")

        client = genai.Client(api_key=self.api_key)

        model = config.model if config else "gemini-2.0-flash"

        # Convert messages to Gemini format
        contents = []
        system_instruction = None
        for msg in messages:
            if msg.role.value == "system":
                system_instruction = msg.content
            else:
                role = "model" if msg.role.value == "assistant" else "user"
                contents.append({"role": role, "parts": [{"text": msg.content}]})

        generate_config = {}
        if config:
            generate_config["max_output_tokens"] = config.max_tokens
            generate_config["temperature"] = config.temperature

        kwargs: dict = {
            "model": model,
            "contents": contents,
            "config": generate_config,
        }
        if system_instruction:
            kwargs["config"]["system_instruction"] = system_instruction

        response = await client.aio.models.generate_content(**kwargs)

        return LLMResponse(
            content=response.text or "",
            tool_calls=[],  # TODO: implement function calling
            model=model,
        )
```

**Step 5: Run tests to verify they pass**

```bash
cd C:/Code/YGN-SAGE/sage-python
pytest tests/test_llm.py -v
```

Expected: All 6 tests PASS

**Step 6: Commit**

```bash
cd C:/Code/YGN-SAGE
git add sage-python/
git commit -m "feat(sdk): add LLM abstraction layer with Anthropic, OpenAI, Google providers"
```

---

### Task 6: Tool System (Python SDK)

**Files:**
- Create: `sage-python/src/sage/tools/__init__.py`
- Create: `sage-python/src/sage/tools/base.py`
- Create: `sage-python/src/sage/tools/registry.py`
- Create: `sage-python/src/sage/tools/builtin.py`
- Create: `sage-python/tests/test_tools.py`

**Step 1: Write tests for tool system**

```python
# sage-python/tests/test_tools.py
"""Tests for the tool system."""
import pytest
from sage.tools.base import Tool, ToolResult
from sage.tools.registry import ToolRegistry


def test_tool_creation():
    @Tool.define(
        name="greet",
        description="Greet someone",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    )
    async def greet(name: str) -> str:
        return f"Hello, {name}!"

    assert greet.spec.name == "greet"
    assert greet.spec.description == "Greet someone"


@pytest.mark.asyncio
async def test_tool_execution():
    @Tool.define(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    )
    async def add(a: float, b: float) -> str:
        return str(a + b)

    result = await add.execute({"a": 2, "b": 3})
    assert isinstance(result, ToolResult)
    assert result.output == "5.0"
    assert not result.is_error


@pytest.mark.asyncio
async def test_tool_error_handling():
    @Tool.define(name="fail", description="Always fails", parameters={"type": "object"})
    async def fail() -> str:
        raise ValueError("intentional error")

    result = await fail.execute({})
    assert result.is_error
    assert "intentional error" in result.output


def test_registry_register_and_list():
    registry = ToolRegistry()

    @Tool.define(name="tool_a", description="A", parameters={"type": "object"})
    async def tool_a() -> str:
        return "a"

    registry.register(tool_a)
    assert "tool_a" in registry.list_tools()


def test_registry_get_tool():
    registry = ToolRegistry()

    @Tool.define(name="my_tool", description="Mine", parameters={"type": "object"})
    async def my_tool() -> str:
        return "mine"

    registry.register(my_tool)
    tool = registry.get("my_tool")
    assert tool is not None
    assert tool.spec.name == "my_tool"


def test_registry_search():
    registry = ToolRegistry()

    @Tool.define(name="bash", description="Execute bash commands", parameters={"type": "object"})
    async def bash() -> str:
        return ""

    @Tool.define(name="python", description="Execute Python code", parameters={"type": "object"})
    async def python() -> str:
        return ""

    registry.register(bash)
    registry.register(python)

    results = registry.search("execute")
    assert len(results) == 2

    results = registry.search("bash")
    assert len(results) == 1
```

**Step 2: Run tests to verify they fail**

```bash
cd C:/Code/YGN-SAGE/sage-python
pytest tests/test_tools.py -v
```

Expected: FAIL

**Step 3: Implement tool base and registry**

```python
# sage-python/src/sage/tools/__init__.py
"""Tool system for YGN-SAGE agents."""
from sage.tools.base import Tool, ToolResult
from sage.tools.registry import ToolRegistry

__all__ = ["Tool", "ToolResult", "ToolRegistry"]
```

```python
# sage-python/src/sage/tools/base.py
"""Base tool types and decorator."""
from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

from sage.llm.base import ToolDef


@dataclass
class ToolResult:
    output: str
    is_error: bool = False


class Tool:
    """A tool that an agent can use."""

    def __init__(self, spec: ToolDef, handler: Callable[..., Awaitable[str]]):
        self.spec = spec
        self._handler = handler

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool with given arguments."""
        try:
            output = await self._handler(**arguments)
            return ToolResult(output=output, is_error=False)
        except Exception as e:
            return ToolResult(
                output=f"Error: {type(e).__name__}: {e}\n{traceback.format_exc()}",
                is_error=True,
            )

    @staticmethod
    def define(
        name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> Callable[[Callable[..., Awaitable[str]]], Tool]:
        """Decorator to define a tool from an async function."""

        def decorator(func: Callable[..., Awaitable[str]]) -> Tool:
            spec = ToolDef(name=name, description=description, parameters=parameters)
            return Tool(spec=spec, handler=func)

        return decorator
```

```python
# sage-python/src/sage/tools/registry.py
"""Tool registry for managing available tools."""
from __future__ import annotations

from sage.tools.base import Tool


class ToolRegistry:
    """Registry for managing tools available to agents."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.spec.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    def search(self, query: str) -> list[Tool]:
        """Search tools by name or description."""
        query_lower = query.lower()
        return [
            tool
            for tool in self._tools.values()
            if query_lower in tool.spec.name.lower()
            or query_lower in tool.spec.description.lower()
        ]

    def get_tool_defs(self, names: list[str] | None = None) -> list:
        """Get ToolDef list for LLM calls."""
        if names is None:
            return [t.spec for t in self._tools.values()]
        return [self._tools[n].spec for n in names if n in self._tools]
```

```python
# sage-python/src/sage/tools/builtin.py
"""Built-in tools for YGN-SAGE agents."""
from __future__ import annotations

import asyncio
import subprocess
from sage.tools.base import Tool


bash_tool = Tool.define(
    name="bash",
    description="Execute a bash command and return its output.",
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The bash command to execute"},
            "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 120},
        },
        "required": ["command"],
    },
)(lambda command, timeout=120: _run_bash(command, timeout))


async def _run_bash(command: str, timeout: int = 120) -> str:
    """Execute a bash command asynchronously."""
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return f"Error: Command timed out after {timeout}s"

    output = stdout.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="replace")
        output += f"\nSTDERR:\n{err}\nExit code: {proc.returncode}"
    return output
```

**Step 4: Run tests to verify they pass**

```bash
cd C:/Code/YGN-SAGE/sage-python
pytest tests/test_tools.py -v
```

Expected: All 6 tests PASS

**Step 5: Commit**

```bash
cd C:/Code/YGN-SAGE
git add sage-python/src/sage/tools/ sage-python/tests/test_tools.py
git commit -m "feat(sdk): add tool system with Tool decorator, ToolRegistry, and builtin bash tool"
```

---

### Task 7: Agent Runtime (the core agent loop)

**Files:**
- Create: `sage-python/src/sage/agent.py`
- Create: `sage-python/tests/test_agent.py`

**Step 1: Write tests for agent runtime**

```python
# sage-python/tests/test_agent.py
"""Tests for the agent runtime."""
import pytest
from unittest.mock import AsyncMock
from sage.agent import Agent, AgentConfig
from sage.llm.base import LLMConfig, LLMResponse, Message, Role
from sage.tools.base import Tool, ToolResult
from sage.tools.registry import ToolRegistry


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    provider = AsyncMock()
    provider.name = "mock"
    provider.generate = AsyncMock(
        return_value=LLMResponse(content="I'm done.", tool_calls=[])
    )
    return provider


@pytest.fixture
def basic_config():
    return AgentConfig(
        name="test-agent",
        llm=LLMConfig(provider="mock", model="mock-model"),
        system_prompt="You are a helpful assistant.",
        max_steps=5,
    )


def test_agent_creation(basic_config, mock_llm):
    agent = Agent(config=basic_config, llm_provider=mock_llm)
    assert agent.config.name == "test-agent"
    assert agent.step_count == 0


@pytest.mark.asyncio
async def test_agent_simple_response(basic_config, mock_llm):
    agent = Agent(config=basic_config, llm_provider=mock_llm)
    result = await agent.run("Hello!")
    assert result == "I'm done."
    assert agent.step_count == 1


@pytest.mark.asyncio
async def test_agent_with_tool_use(basic_config, mock_llm):
    from sage.llm.base import ToolCall as LLMToolCall

    # First call returns tool use, second returns final response
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="",
                tool_calls=[LLMToolCall(id="tc1", name="greet", arguments={"name": "World"})],
            ),
            LLMResponse(content="The greeting is: Hello, World!", tool_calls=[]),
        ]
    )

    @Tool.define(
        name="greet",
        description="Greet someone",
        parameters={"type": "object", "properties": {"name": {"type": "string"}}},
    )
    async def greet(name: str) -> str:
        return f"Hello, {name}!"

    registry = ToolRegistry()
    registry.register(greet)

    agent = Agent(config=basic_config, llm_provider=mock_llm, tool_registry=registry)
    result = await agent.run("Greet the world")

    assert "Hello, World!" in result
    assert agent.step_count == 2


@pytest.mark.asyncio
async def test_agent_max_steps(basic_config, mock_llm):
    from sage.llm.base import ToolCall as LLMToolCall

    basic_config.max_steps = 2

    # Always returns tool calls, never stops
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(
            content="",
            tool_calls=[LLMToolCall(id="tc", name="greet", arguments={"name": "X"})],
        )
    )

    @Tool.define(name="greet", description="G", parameters={"type": "object", "properties": {"name": {"type": "string"}}})
    async def greet(name: str) -> str:
        return f"Hi {name}"

    registry = ToolRegistry()
    registry.register(greet)

    agent = Agent(config=basic_config, llm_provider=mock_llm, tool_registry=registry)
    result = await agent.run("go")

    assert agent.step_count == 2  # Stopped at max_steps
```

**Step 2: Run tests to verify they fail**

```bash
cd C:/Code/YGN-SAGE/sage-python
pytest tests/test_agent.py -v
```

Expected: FAIL

**Step 3: Implement Agent runtime**

```python
# sage-python/src/sage/agent.py
"""Core agent runtime for YGN-SAGE."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sage.llm.base import LLMConfig, LLMProvider, LLMResponse, Message, Role, ToolDef
from sage.tools.base import Tool
from sage.tools.registry import ToolRegistry


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    llm: LLMConfig
    system_prompt: str = "You are a helpful AI assistant."
    max_steps: int = 100
    tools: list[str] | None = None  # Tool names to use (None = all in registry)


class Agent:
    """Core agent that runs the LLM -> Tool -> LLM loop."""

    def __init__(
        self,
        config: AgentConfig,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry | None = None,
    ):
        self.config = config
        self._llm = llm_provider
        self._tools = tool_registry or ToolRegistry()
        self._messages: list[Message] = []
        self.step_count: int = 0
        self.result: str | None = None

    async def run(self, task: str) -> str:
        """Execute the agent on a task. Returns the final text response."""
        # Initialize conversation
        self._messages = [
            Message(role=Role.SYSTEM, content=self.config.system_prompt),
            Message(role=Role.USER, content=task),
        ]

        # Get tool definitions
        tool_defs = self._get_tool_defs()

        # Agent loop
        while self.step_count < self.config.max_steps:
            self.step_count += 1

            # Call LLM
            response = await self._llm.generate(
                messages=self._messages,
                tools=tool_defs if tool_defs else None,
                config=self.config.llm,
            )

            # If no tool calls, we're done
            if not response.tool_calls:
                self.result = response.content
                self._messages.append(
                    Message(role=Role.ASSISTANT, content=response.content)
                )
                return response.content

            # Add assistant message with tool calls
            self._messages.append(
                Message(role=Role.ASSISTANT, content=response.content)
            )

            # Execute tool calls
            for tc in response.tool_calls:
                tool = self._tools.get(tc.name)
                if tool is None:
                    tool_output = f"Error: Unknown tool '{tc.name}'"
                    is_error = True
                else:
                    result = await tool.execute(tc.arguments)
                    tool_output = result.output
                    is_error = result.is_error

                self._messages.append(
                    Message(
                        role=Role.TOOL,
                        content=tool_output,
                        tool_call_id=tc.id,
                        name=tc.name,
                    )
                )

        # Max steps reached
        self.result = f"Agent reached max steps ({self.config.max_steps})"
        return self.result

    def _get_tool_defs(self) -> list[ToolDef]:
        """Get tool definitions for the LLM."""
        if self.config.tools is not None:
            return self._tools.get_tool_defs(self.config.tools)
        return self._tools.get_tool_defs()
```

**Step 4: Update sage/__init__.py exports**

```python
# sage-python/src/sage/__init__.py
"""YGN-SAGE: Self-Adaptive Generation Engine."""

__version__ = "0.1.0"

from sage.agent import Agent, AgentConfig
from sage.llm import LLMConfig, LLMRegistry
from sage.tools import Tool, ToolRegistry, ToolResult

__all__ = [
    "Agent",
    "AgentConfig",
    "LLMConfig",
    "LLMRegistry",
    "Tool",
    "ToolRegistry",
    "ToolResult",
]
```

**Step 5: Run tests to verify they pass**

```bash
cd C:/Code/YGN-SAGE/sage-python
pytest tests/ -v
```

Expected: All tests PASS across all test files

**Step 6: Commit**

```bash
cd C:/Code/YGN-SAGE
git add sage-python/src/sage/agent.py sage-python/tests/test_agent.py sage-python/src/sage/__init__.py
git commit -m "feat(sdk): add Agent runtime with LLM->Tool->LLM loop, max_steps, tool execution"
```

---

## Phase 3: Memory System (Python)

### Task 8: Memory interfaces and episodic memory

**Files:**
- Create: `sage-python/src/sage/memory/__init__.py`
- Create: `sage-python/src/sage/memory/base.py`
- Create: `sage-python/src/sage/memory/working.py`
- Create: `sage-python/src/sage/memory/episodic.py`
- Create: `sage-python/tests/test_memory.py`

**Step 1: Write tests**

```python
# sage-python/tests/test_memory.py
"""Tests for the memory system."""
import pytest
from sage.memory.working import WorkingMemory
from sage.memory.episodic import EpisodicMemory


def test_working_memory_add_event():
    mem = WorkingMemory(agent_id="agent-1")
    event_id = mem.add_event("tool_call", "Called bash with 'ls'")
    assert event_id is not None
    event = mem.get_event(event_id)
    assert event is not None
    assert event["content"] == "Called bash with 'ls'"


def test_working_memory_recent():
    mem = WorkingMemory(agent_id="agent-1")
    for i in range(10):
        mem.add_event("step", f"Step {i}")
    recent = mem.recent_events(3)
    assert len(recent) == 3
    assert recent[0]["content"] == "Step 7"


def test_working_memory_to_messages():
    mem = WorkingMemory(agent_id="agent-1")
    mem.add_event("user", "Hello")
    mem.add_event("assistant", "Hi there!")
    messages = mem.to_context_string()
    assert "Hello" in messages
    assert "Hi there!" in messages


@pytest.mark.asyncio
async def test_episodic_memory_store_and_search():
    mem = EpisodicMemory()
    await mem.store(
        key="fix-auth-bug",
        content="Fixed the auth bug by checking token expiry before validation.",
        metadata={"task": "bug-fix", "files": ["auth.py"]},
    )

    results = await mem.search("authentication bug fix")
    assert len(results) >= 1
    assert "auth" in results[0]["content"].lower()


@pytest.mark.asyncio
async def test_episodic_memory_empty_search():
    mem = EpisodicMemory()
    results = await mem.search("nonexistent query")
    assert len(results) == 0
```

**Step 2: Run tests to verify they fail**

```bash
cd C:/Code/YGN-SAGE/sage-python
pytest tests/test_memory.py -v
```

Expected: FAIL

**Step 3: Implement memory modules**

```python
# sage-python/src/sage/memory/__init__.py
"""Memory system for YGN-SAGE agents."""
from sage.memory.working import WorkingMemory
from sage.memory.episodic import EpisodicMemory

__all__ = ["WorkingMemory", "EpisodicMemory"]
```

```python
# sage-python/src/sage/memory/base.py
"""Base memory types."""
from __future__ import annotations

from typing import Any, Protocol


class MemoryStore(Protocol):
    """Protocol for memory stores."""

    async def store(self, key: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        ...

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        ...
```

```python
# sage-python/src/sage/memory/working.py
"""Working memory - short-term, in-memory, per-agent execution."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any


class WorkingMemory:
    """In-memory working memory for a single agent execution."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._events: list[dict[str, Any]] = []
        self._children: list[str] = []

    def add_event(self, event_type: str, content: str) -> str:
        """Add an event and return its ID."""
        event_id = str(uuid.uuid4())
        self._events.append({
            "id": event_id,
            "type": event_type,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_summary": False,
        })
        return event_id

    def get_event(self, event_id: str) -> dict[str, Any] | None:
        """Get an event by ID."""
        for event in self._events:
            if event["id"] == event_id:
                return event
        return None

    def recent_events(self, n: int) -> list[dict[str, Any]]:
        """Get the N most recent events (oldest first within window)."""
        return self._events[-n:] if n < len(self._events) else list(self._events)

    def event_count(self) -> int:
        return len(self._events)

    def add_child_agent(self, child_id: str) -> None:
        self._children.append(child_id)

    def child_agents(self) -> list[str]:
        return list(self._children)

    def compress(self, keep_recent: int, summary: str) -> None:
        """Compress old events into a summary."""
        if len(self._events) <= keep_recent:
            return
        recent = self._events[-keep_recent:]
        self._events = [{
            "id": str(uuid.uuid4()),
            "type": "summary",
            "content": summary,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_summary": True,
        }] + recent

    def to_context_string(self) -> str:
        """Render working memory as a context string for LLM."""
        parts = []
        for event in self._events:
            parts.append(f"[{event['type']}] {event['content']}")
        return "\n".join(parts)
```

```python
# sage-python/src/sage/memory/episodic.py
"""Episodic memory - medium-term, cross-session experience storage."""
from __future__ import annotations

from typing import Any


class EpisodicMemory:
    """Simple in-memory episodic store. Replace with Qdrant/Neo4j for production."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    async def store(
        self,
        key: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store an episodic memory entry."""
        self._entries.append({
            "key": key,
            "content": content,
            "metadata": metadata or {},
        })

    async def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search episodic memory by keyword matching.

        In production, this uses vector similarity via Qdrant.
        This in-memory version uses simple substring matching.
        """
        query_lower = query.lower()
        scored = []
        for entry in self._entries:
            # Simple relevance: count matching words
            content_lower = entry["content"].lower()
            key_lower = entry["key"].lower()
            score = sum(
                1 for word in query_lower.split()
                if word in content_lower or word in key_lower
            )
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]
```

**Step 4: Run tests to verify they pass**

```bash
cd C:/Code/YGN-SAGE/sage-python
pytest tests/test_memory.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
cd C:/Code/YGN-SAGE
git add sage-python/src/sage/memory/ sage-python/tests/test_memory.py
git commit -m "feat(sdk): add memory system with WorkingMemory and EpisodicMemory"
```

---

## Phases 4-8: Outlined (detailed plans written per-phase when reached)

### Phase 4: Docker Sandboxing & Tool Synthesis
- Task 9: Docker sandbox manager (create, snapshot, destroy containers)
- Task 10: Meta-tools (create_tool, search_tool) that let agents write new tools
- Task 11: Async tool execution with background handles

### Phase 5: Full Topology Engine
- Task 12: Vertical topology (parent delegates sequential sub-tasks)
- Task 13: Horizontal topology (parallel agents + ensemble merge)
- Task 14: Agent communication via message board and event bus

### Phase 6: Evolution Engine
- Task 15: Population database (MAP-Elites grid)
- Task 16: LLM mutation pipeline (SEARCH/REPLACE diff format)
- Task 17: Evaluation cascade (progressive difficulty filtering)
- Task 18: Evolutionary loop orchestrator

### Phase 7: Strategy Engine
- Task 19: Regret matching meta-solver
- Task 20: Projected Replicator Dynamics (PRD) solver
- Task 21: SHOR-PSRO hybrid solver
- Task 22: Dynamic resource allocation based on strategy

### Phase 8: Flagship Agent (sage-discover)
- Task 23: Research workflow (explore -> hypothesize -> evolve -> evaluate)
- Task 24: Paper search and analysis tools
- Task 25: Experiment formalization (problem -> evaluable code)
- Task 26: End-to-end integration test on a real discovery problem
