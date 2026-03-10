# Rust Cognitive Engine — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate the cognitive decision pipeline into sage-core (Rust). Python modules kept as frozen shadow oracles until Rust parity proven. Evidence-first: no deletion without proof.

**Architecture:** SystemRouter in Rust: hard constraints → structural scoring → contextual bandit → hybrid verification. Template-first topologies (8 typed templates) unified with existing Contract IR. Three-flow edge model (control/message/state). Contextual bandit (LinUCB) with per-arm posteriors and global Pareto front at decision time. LLM-synthesized topology generation (5th path). Dual-mode executor (static Kahn's + dynamic gate-based).

**Tech Stack:** Rust 1.90+, PyO3 0.25, petgraph 0.6, toml 0.8, rand 0.9, rusqlite 0.33 (bundled), serde, dashmap, arrow 55, ort 2.0 (optional/onnx feature)

**Design doc:** `docs/plans/2026-03-10-rust-cognitive-engine-design.md`

**Revised phasing (5 phases, per expert review #2):**
1. ModelCard + SystemRouter (DONE — Tasks 1-7)
2. Phase 1.5: Shadow mode + evidence collection (NEW — Tasks 8-9)
3. Phase 2: Hybrid verifier + runtime contracts + template topologies (Tasks 10-13)
4. Phase 3: Contextual bandit + persistence (Tasks 14-16)
5. Phase 4: MAP-Elites + LLM synthesis + dual-mode executor + Phase 5: Python cleanup (Tasks 17-22)

---

## Phase 1: ModelCard + SystemRouter

### Task 1: Add new dependencies to Cargo.toml

**Files:**
- Modify: `sage-core/Cargo.toml`

**Step 1: Add dependencies**

Add these to `[dependencies]`:
```toml
toml = "0.8"
rand = "0.9"
rusqlite = { version = "0.33", features = ["bundled"], optional = true }
```

Add a new feature:
```toml
cognitive = ["dep:rusqlite"]
```

**Step 2: Verify compilation**

Run: `cd sage-core && cargo check`
Expected: compiles with no errors

**Step 3: Commit**

```bash
git add sage-core/Cargo.toml
git commit -m "feat(core): add toml, rand, rusqlite deps for cognitive engine"
```

---

### Task 2: ModelCard struct + TOML parsing

**Files:**
- Create: `sage-core/src/routing/model_card.rs`
- Test: `sage-core/tests/test_model_card.rs`

**Step 1: Write the failing test**

Create `sage-core/tests/test_model_card.rs`:
```rust
use sage_core::routing::model_card::{ModelCard, CognitiveSystem};

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
```

**Step 2: Run test to verify it fails**

Run: `cd sage-core && cargo test --test test_model_card`
Expected: FAIL — module `model_card` not found

**Step 3: Write ModelCard implementation**

Create `sage-core/src/routing/model_card.rs`:
```rust
//! ModelCard — capability declaration for LLM models.
//!
//! Inspired by Google A2A Agent Cards but specialized for LLM model selection.
//! Each card declares what a model is good at and how it fits S1/S2/S3 cognitive systems.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt;

// ── CognitiveSystem ────────────────────────────────────────────────────────

/// The three cognitive modes (Kahneman-inspired).
/// S1 = Fast/Intuitive, S2 = Deliberate/Tools, S3 = Formal/Reasoning.
#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CognitiveSystem {
    S1 = 1,
    S2 = 2,
    S3 = 3,
}

impl fmt::Display for CognitiveSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::S1 => write!(f, "S1"),
            Self::S2 => write!(f, "S2"),
            Self::S3 => write!(f, "S3"),
        }
    }
}

#[pymethods]
impl CognitiveSystem {
    fn __repr__(&self) -> String {
        format!("CognitiveSystem.{self}")
    }
}

// ── ModelCard ──────────────────────────────────────────────────────────────

#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    #[pyo3(get)] pub id: String,
    #[pyo3(get)] pub provider: String,
    #[pyo3(get)] pub family: String,
    // Capability scores [0.0, 1.0]
    #[pyo3(get)] pub code_score: f32,
    #[pyo3(get)] pub reasoning_score: f32,
    #[pyo3(get)] pub tool_use_score: f32,
    #[pyo3(get)] pub math_score: f32,
    #[pyo3(get)] pub formal_z3_strength: f32,
    // Cost & latency
    #[pyo3(get)] pub cost_input_per_m: f32,
    #[pyo3(get)] pub cost_output_per_m: f32,
    #[pyo3(get)] pub latency_ttft_ms: f32,
    #[pyo3(get)] pub tokens_per_sec: f32,
    // System affinity [0.0, 1.0]
    #[pyo3(get)] pub s1_affinity: f32,
    #[pyo3(get)] pub s2_affinity: f32,
    #[pyo3(get)] pub s3_affinity: f32,
    // Topology preferences
    #[pyo3(get)] pub recommended_topologies: Vec<String>,
    // Capability flags
    #[pyo3(get)] pub supports_tools: bool,
    #[pyo3(get)] pub supports_json_mode: bool,
    #[pyo3(get)] pub supports_vision: bool,
    #[pyo3(get)] pub context_window: u32,
}

/// TOML wrapper for deserialization.
#[derive(Deserialize)]
struct CardsFile {
    models: Vec<ModelCard>,
}

#[pymethods]
impl ModelCard {
    /// Return the cognitive system this model is best suited for.
    pub fn best_system(&self) -> CognitiveSystem {
        if self.s3_affinity >= self.s2_affinity && self.s3_affinity >= self.s1_affinity {
            CognitiveSystem::S3
        } else if self.s2_affinity >= self.s1_affinity {
            CognitiveSystem::S2
        } else {
            CognitiveSystem::S1
        }
    }

    /// Affinity score for a given system.
    pub fn affinity_for(&self, system: CognitiveSystem) -> f32 {
        match system {
            CognitiveSystem::S1 => self.s1_affinity,
            CognitiveSystem::S2 => self.s2_affinity,
            CognitiveSystem::S3 => self.s3_affinity,
        }
    }

    /// Estimated cost for a given number of input+output tokens (in USD).
    pub fn estimate_cost(&self, input_tokens: u32, output_tokens: u32) -> f32 {
        (input_tokens as f32 / 1_000_000.0) * self.cost_input_per_m
            + (output_tokens as f32 / 1_000_000.0) * self.cost_output_per_m
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelCard(id='{}', provider='{}', s1={:.2}, s2={:.2}, s3={:.2})",
            self.id, self.provider, self.s1_affinity, self.s2_affinity, self.s3_affinity
        )
    }
}

impl ModelCard {
    /// Parse a TOML string containing [[models]] array.
    pub fn parse_toml(toml_str: &str) -> Result<Vec<Self>, toml::de::Error> {
        let file: CardsFile = toml::from_str(toml_str)?;
        Ok(file.models)
    }

    /// Load cards from a TOML file path.
    pub fn load_from_file(path: &str) -> Result<Vec<Self>, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        Ok(Self::parse_toml(&content)?)
    }
}
```

**Step 4: Update routing/mod.rs to export model_card**

In `sage-core/src/routing/mod.rs`, add:
```rust
pub mod model_card;
```

**Step 5: Update lib.rs to register PyClasses**

In `sage-core/src/lib.rs`, add after existing routing exports:
```rust
m.add_class::<routing::model_card::ModelCard>()?;
m.add_class::<routing::model_card::CognitiveSystem>()?;
```

**Step 6: Run tests**

Run: `cd sage-core && cargo test --test test_model_card`
Expected: 3 tests PASS

**Step 7: Commit**

```bash
git add sage-core/src/routing/model_card.rs sage-core/src/routing/mod.rs sage-core/src/lib.rs sage-core/tests/test_model_card.rs
git commit -m "feat(core): add ModelCard struct with TOML parsing and CognitiveSystem enum"
```

---

### Task 3: ModelRegistry — card management + selection

**Files:**
- Create: `sage-core/src/routing/model_registry.rs`
- Test: `sage-core/tests/test_model_registry.rs`

**Step 1: Write the failing test**

Create `sage-core/tests/test_model_registry.rs`:
```rust
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
    // Should return models sorted by s1_affinity descending
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
```

**Step 2: Run test to verify it fails**

Run: `cd sage-core && cargo test --test test_model_registry`
Expected: FAIL — module not found

**Step 3: Write ModelRegistry implementation**

Create `sage-core/src/routing/model_registry.rs`:
```rust
//! ModelRegistry — manages ModelCards and selects best models per CognitiveSystem.

use pyo3::prelude::*;
use std::collections::HashMap;

use super::model_card::{CognitiveSystem, ModelCard};

#[pyclass]
#[derive(Debug, Clone)]
pub struct ModelRegistry {
    cards: HashMap<String, ModelCard>,
}

#[pymethods]
impl ModelRegistry {
    /// Load from a TOML file path.
    #[staticmethod]
    pub fn from_toml_file(path: &str) -> PyResult<Self> {
        let cards_vec = ModelCard::load_from_file(path)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let cards = cards_vec.into_iter().map(|c| (c.id.clone(), c)).collect();
        Ok(Self { cards })
    }

    /// Number of registered cards.
    pub fn len(&self) -> usize {
        self.cards.len()
    }

    /// Get a card by model ID.
    pub fn get(&self, id: &str) -> Option<ModelCard> {
        self.cards.get(id).cloned()
    }

    /// Register a new card (or update existing).
    pub fn register(&mut self, card: ModelCard) {
        self.cards.insert(card.id.clone(), card);
    }

    /// Remove a card by ID.
    pub fn unregister(&mut self, id: &str) {
        self.cards.remove(id);
    }

    /// Return all cards sorted by affinity for the given system (descending).
    pub fn select_for_system(&self, system: CognitiveSystem) -> Vec<ModelCard> {
        let mut candidates: Vec<_> = self.cards.values().cloned().collect();
        candidates.sort_by(|a, b| {
            b.affinity_for(system)
                .partial_cmp(&a.affinity_for(system))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates
    }

    /// Return all card IDs.
    pub fn list_ids(&self) -> Vec<String> {
        self.cards.keys().cloned().collect()
    }

    fn __repr__(&self) -> String {
        format!("ModelRegistry(models={})", self.cards.len())
    }

    fn __len__(&self) -> usize {
        self.cards.len()
    }
}

impl ModelRegistry {
    /// Load from a TOML string (for tests and embedding).
    pub fn from_toml_str(toml_str: &str) -> Result<Self, toml::de::Error> {
        let cards_vec = ModelCard::parse_toml(toml_str)?;
        let cards = cards_vec.into_iter().map(|c| (c.id.clone(), c)).collect();
        Ok(Self { cards })
    }
}
```

**Step 4: Update routing/mod.rs**

Add to `sage-core/src/routing/mod.rs`:
```rust
pub mod model_registry;
```

**Step 5: Update lib.rs**

Add to `sage-core/src/lib.rs`:
```rust
m.add_class::<routing::model_registry::ModelRegistry>()?;
```

**Step 6: Run tests**

Run: `cd sage-core && cargo test --test test_model_registry`
Expected: 6 tests PASS

**Step 7: Commit**

```bash
git add sage-core/src/routing/model_registry.rs sage-core/src/routing/mod.rs sage-core/src/lib.rs sage-core/tests/test_model_registry.rs
git commit -m "feat(core): add ModelRegistry with TOML loading and system-based selection"
```

---

### Task 4: Create cards.toml config

**Files:**
- Create: `sage-core/config/cards.toml`

**Step 1: Create config directory**

Run: `mkdir -p sage-core/config`

**Step 2: Create cards.toml**

Migrate from `sage-python/config/model_profiles.toml`, adding s1/s2/s3 affinity + new fields. All 18 models from the existing TOML must be included. Affinity values derived from: S1 = cheap+fast models, S2 = code+tool models, S3 = reasoning+formal models.

Fields to add per model: `math_score`, `formal_z3_strength`, `s1_affinity`, `s2_affinity`, `s3_affinity`, `recommended_topologies`, `supports_json_mode`, `supports_vision`, `context_window`.

Derivation rules:
- `s1_affinity = high` if `cost_input < 0.1` and `latency_ttft_ms < 1500`
- `s2_affinity = high` if `code_score > 0.7` and `supports_tools`
- `s3_affinity = high` if `reasoning_score > 0.85`
- `math_score` ≈ `reasoning_score * 0.9` (heuristic, refine later)
- `formal_z3_strength` ≈ `reasoning_score * 0.7` (heuristic)

**Step 3: Write test that loads the real file**

Add to `sage-core/tests/test_model_card.rs`:
```rust
#[test]
fn load_real_cards_toml() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/config/cards.toml");
    let cards = ModelCard::load_from_file(path).expect("Failed to load cards.toml");
    assert!(cards.len() >= 18, "Expected at least 18 models, got {}", cards.len());
    // Every card must have valid affinities
    for card in &cards {
        assert!(card.s1_affinity >= 0.0 && card.s1_affinity <= 1.0, "Bad s1 for {}", card.id);
        assert!(card.s2_affinity >= 0.0 && card.s2_affinity <= 1.0, "Bad s2 for {}", card.id);
        assert!(card.s3_affinity >= 0.0 && card.s3_affinity <= 1.0, "Bad s3 for {}", card.id);
    }
}
```

**Step 4: Run test**

Run: `cd sage-core && cargo test --test test_model_card -- load_real_cards_toml`
Expected: PASS

**Step 5: Commit**

```bash
git add sage-core/config/cards.toml sage-core/tests/test_model_card.rs
git commit -m "feat(core): add cards.toml with 18+ model profiles and S1/S2/S3 affinities"
```

---

### Task 5: SystemRouter — cognitive system decision engine

**Files:**
- Create: `sage-core/src/routing/system_router.rs`
- Test: `sage-core/tests/test_system_router.rs`

**Step 1: Write the failing test**

Create `sage-core/tests/test_system_router.rs`:
```rust
use sage_core::routing::model_card::CognitiveSystem;
use sage_core::routing::model_registry::ModelRegistry;
use sage_core::routing::system_router::{SystemRouter, RoutingDecision};

fn test_registry() -> ModelRegistry {
    ModelRegistry::from_toml_str(r#"
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
    "#).unwrap()
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
    let decision = router.route("Write a Python function to sort a list using quicksort", 10.0);
    assert_eq!(decision.system, CognitiveSystem::S2);
    assert_eq!(decision.model_id, "coder");
}

#[test]
fn route_formal_task_to_s3() {
    let reg = test_registry();
    let router = SystemRouter::new(reg);
    let decision = router.route("Prove by induction that the sum of first n natural numbers is n(n+1)/2", 10.0);
    assert_eq!(decision.system, CognitiveSystem::S3);
    assert_eq!(decision.model_id, "reasoner");
}

#[test]
fn budget_constraint_downgrades() {
    let reg = test_registry();
    let router = SystemRouter::new(reg);
    // Very low budget — should pick cheapest model regardless of system
    let decision = router.route("Write a complex distributed system", 0.001);
    assert_eq!(decision.model_id, "fast", "Should pick cheapest model under tight budget");
}

#[test]
fn routing_decision_has_confidence() {
    let reg = test_registry();
    let router = SystemRouter::new(reg);
    let decision = router.route("Hello world", 10.0);
    assert!(decision.confidence > 0.0 && decision.confidence <= 1.0);
}
```

**Step 2: Run test to verify it fails**

Run: `cd sage-core && cargo test --test test_system_router`
Expected: FAIL — module not found

**Step 3: Write SystemRouter implementation**

Create `sage-core/src/routing/system_router.rs`:
```rust
//! SystemRouter — decides which cognitive System (S1/S2/S3) to activate.
//!
//! This is NOT a sequential pipeline. It chooses a complete mode of thought:
//! - System 1 (Fast/Intuitive): direct response, cheap model, no verification
//! - System 2 (Deliberate/Tools): code execution, AVR loop, sandbox
//! - System 3 (Formal/Reasoning): deep reasoning, Z3 bounds, formal proofs

use pyo3::prelude::*;

use super::features::StructuralFeatures;
use super::model_card::{CognitiveSystem, ModelCard};
use super::model_registry::ModelRegistry;

// ── RoutingDecision ────────────────────────────────────────────────────────

#[pyclass]
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    #[pyo3(get)]
    pub system: CognitiveSystem,
    #[pyo3(get)]
    pub model_id: String,
    #[pyo3(get)]
    pub confidence: f32,
    #[pyo3(get)]
    pub estimated_cost: f32,
}

#[pymethods]
impl RoutingDecision {
    fn __repr__(&self) -> String {
        format!(
            "RoutingDecision(system={}, model='{}', confidence={:.2}, cost={:.4})",
            self.system, self.model_id, self.confidence, self.estimated_cost
        )
    }
}

// ── SystemRouter ───────────────────────────────────────────────────────────

#[pyclass]
pub struct SystemRouter {
    registry: ModelRegistry,
}

#[pymethods]
impl SystemRouter {
    #[new]
    pub fn new(registry: ModelRegistry) -> Self {
        Self { registry }
    }

    /// Route a task to the best cognitive system + model.
    ///
    /// 1. Structural analysis → S1/S2/S3 scores
    /// 2. ModelCard affinity matching → best model per system
    /// 3. Budget constraint → downgrade if needed
    pub fn route(&self, task: &str, budget: f32) -> RoutingDecision {
        // Step 1: Structural analysis
        let features = StructuralFeatures::extract_from(task);
        let (system, confidence) = self.decide_system(&features);

        // Step 2: Select best model for chosen system
        let candidates = self.registry.select_for_system(system);

        // Step 3: Budget-constrained selection
        let (model, estimated_cost) = self.select_within_budget(&candidates, budget);

        // If budget forced a downgrade, recalculate system from model's best affinity
        let final_system = if model.id != candidates[0].id {
            model.best_system()
        } else {
            system
        };

        RoutingDecision {
            system: final_system,
            model_id: model.id.clone(),
            confidence,
            estimated_cost,
        }
    }

    fn __repr__(&self) -> String {
        format!("SystemRouter(models={})", self.registry.len())
    }
}

impl SystemRouter {
    /// Decide cognitive system from structural features.
    fn decide_system(&self, features: &StructuralFeatures) -> (CognitiveSystem, f32) {
        let complexity = features.keyword_complexity;
        let uncertainty = features.keyword_uncertainty;

        // S3: high complexity + formal/proof indicators
        if complexity >= 0.65 && uncertainty < 0.3 {
            return (CognitiveSystem::S3, 0.7 + complexity * 0.3);
        }

        // S2: medium complexity or tool/code required
        if features.tool_required || features.has_code_block || complexity >= 0.35 {
            return (CognitiveSystem::S2, 0.6 + complexity * 0.3);
        }

        // S1: simple, fast, intuitive
        let confidence = (1.0 - complexity).clamp(0.5, 0.95);
        (CognitiveSystem::S1, confidence)
    }

    /// Select best model within budget. Returns (model, estimated_cost).
    /// Assumes ~1000 input + ~2000 output tokens as estimate.
    fn select_within_budget<'a>(
        &self,
        candidates: &'a [ModelCard],
        budget: f32,
    ) -> (&'a ModelCard, f32) {
        let est_input = 1000_u32;
        let est_output = 2000_u32;

        for card in candidates {
            let cost = card.estimate_cost(est_input, est_output);
            if cost <= budget {
                return (card, cost);
            }
        }

        // Fallback: cheapest model overall
        let mut cheapest = &candidates[0];
        let mut min_cost = cheapest.estimate_cost(est_input, est_output);
        for card in candidates.iter().skip(1) {
            let cost = card.estimate_cost(est_input, est_output);
            if cost < min_cost {
                cheapest = card;
                min_cost = cost;
            }
        }
        (cheapest, min_cost)
    }
}
```

**Step 4: Update routing/mod.rs**

Add:
```rust
pub mod system_router;
```

**Step 5: Update lib.rs**

Add:
```rust
m.add_class::<routing::system_router::SystemRouter>()?;
m.add_class::<routing::system_router::RoutingDecision>()?;
```

**Step 6: Run tests**

Run: `cd sage-core && cargo test --test test_system_router`
Expected: 5 tests PASS

**Step 7: Run ALL existing tests to check no regressions**

Run: `cd sage-core && cargo test`
Expected: all existing tests PASS + new tests PASS

**Step 8: Commit**

```bash
git add sage-core/src/routing/system_router.rs sage-core/src/routing/mod.rs sage-core/src/lib.rs sage-core/tests/test_system_router.rs
git commit -m "feat(core): add SystemRouter — cognitive system decision engine (S1/S2/S3)"
```

---

### Task 6: Wire SystemRouter into Python boot.py (shadow mode — NO deletion)

**Files:**
- Modify: `sage-python/src/sage/boot.py`
- Modify: `sage-python/src/sage/agent_loop.py` (routing call site)
- Keep: `sage-python/src/sage/strategy/metacognition.py` (shadow oracle)
- Keep: `sage-python/src/sage/strategy/adaptive_router.py` (shadow oracle)
- Keep: `sage-python/src/sage/strategy/training.py` (shadow oracle)

**Step 1: Build and install sage_core with new exports**

Run: `cd sage-core && maturin develop --features onnx`
Expected: builds and installs sage_core wheel

**Step 2: Verify Python can import new classes**

Run: `python -c "from sage_core import SystemRouter, ModelRegistry, ModelCard, CognitiveSystem; print('OK')"`
Expected: `OK`

**Step 3: Update boot.py**

Replace the AdaptiveRouter/ComplexityRouter wiring with SystemRouter.
Key change in `boot()` function:
- Replace `metacognition = AdaptiveRouter(...)` with `sage_core.SystemRouter(registry)`
- Keep fallback to Python ComplexityRouter if sage_core not available (graceful degradation during transition)

**Step 4: Update agent_loop.py routing call**

Replace the Python routing call with `system.router.route(task, budget)`.

**Step 5: Keep Python routers as shadow oracles (NOT deleted)**

Per expert review #2: Python routers are frozen reference oracles. Both Rust and Python run in parallel, divergences logged. Deletion gated on < 5% divergence on 1000+ traces (Phase 5, Task 18).

**Step 6: Run Python test suite**

Run: `cd sage-python && python -m pytest tests/ -x -v --timeout=60`
Expected: tests pass (routing now goes through Rust with Python shadow)

**Step 7: Commit**

```bash
git add -u
git commit -m "feat: wire Rust SystemRouter into boot.py with Python shadow oracle mode"
```

---

### Task 7: Build + run routing benchmark with new SystemRouter

**Files:**
- Modify: `sage-python/src/sage/bench/routing.py` (if needed)

**Step 1: Run routing benchmark**

Run: `cd sage-python && python -m sage.bench --type routing`
Expected: 30/30 accuracy (same or better than before)

**Step 2: Run E2E proof**

Run: `cd /c/Code/YGN-SAGE && python tests/e2e_proof.py`
Expected: all tests pass

**Step 3: Commit benchmark results**

```bash
git add docs/benchmarks/
git commit -m "bench: routing benchmark with Rust SystemRouter — Phase 1 complete"
```

---

## Phase 1.5: Shadow Mode + Evidence Collection

### Task 8: Shadow mode — dual routing in boot.py

**Files:**
- Modify: `sage-python/src/sage/boot.py`
- Create: `sage-python/src/sage/routing/shadow.py`

**Implementation:** When both Rust and Python routers available, run both on every task. Log divergences (system mismatch, model mismatch) to structured JSON. Track divergence rate. Gate for Phase 5 deletion: < 5% divergence on 1000+ traces.

Key: this must be **zero-overhead when Rust router unavailable** (pure Python path unchanged).

---

### Task 9: Evidence baselines — downstream quality tracking

**Files:**
- Modify: `sage-python/src/sage/bench/routing_downstream.py`
- Create: `sage-python/src/sage/bench/baseline_comparison.py`

**Implementation:** Add baselines to DownstreamEvaluator:
- "always best model" (most expensive)
- "cheapest under SLA" (min cost, quality ≥ threshold)
- "fixed reviewer topology" (always AVR)
- "routing ON vs OFF" (A/B paired)

Evidence gate: Rust router must match or beat "always best model" on cost-quality Pareto.

---

## Phase 2: Hybrid Verifier + Runtime Contracts + Template Topologies

### Task 10: RoutingConstraints struct + hard constraint filter

**Files:**
- Modify: `sage-core/src/routing/system_router.rs`
- Test: `sage-core/tests/test_system_router.rs`

**Implementation:** Add `RoutingConstraints` PyO3 class with `max_cost_usd`, `max_latency_ms`, `min_quality`, `required_capabilities`, `security_label`, `exploration_budget`. Update `route()` to accept constraints and filter models before scoring. Add `decision_id: Ulid` to RoutingDecision.

---

### Task 11: TopologyGraph — unified with Contract IR

**Files:**
- Create: `sage-core/src/topology/mod.rs`
- Create: `sage-core/src/topology/topology_graph.rs`
- Test: `sage-core/tests/test_topology.rs`

**Implementation:** TopologyGraph wrapping `petgraph::DiGraph<TopologyNode, TopologyEdge>` with TopologyNode extending Contract IR concepts (node_id, role, model_id, required_capabilities, security_label, budget). TopologyTemplate enum (8 templates: Sequential, Parallel, AVR, SelfMoA, Hierarchical, Hub, Debate, Brainstorming, Custom). TopologyEdge with three-flow model (EdgeType::Control|Message|State, field_mapping: Option<HashMap<String,String>>, gate: Gate::Open|Closed, condition: Option<String>).

---

### Task 12: Template catalogue + hybrid verifier

**Files:**
- Create: `sage-core/src/topology/templates.rs`
- Create: `sage-core/src/topology/verifier.rs`
- Test: `sage-core/tests/test_verifier.rs`

**Implementation:** 8 built-in templates (Sequential, Parallel, AVR, SelfMoA, Hierarchical, Hub, Debate, Brainstorming). HybridVerifier: DAG validity (petgraph), capability coverage (bitset), budget feasibility (arithmetic), fan-in/out (degree check), security labels (lattice comparison). OxiZ only for genuinely symbolic constraints (behind `smt` feature). **Semantic Diagnosis Layer** (inspired by MASFactory diagnose_node): role coherence, switch condition completeness, loop termination guarantee, entry/exit reachability (BFS), field mapping consistency, capability aggregation.

---

### Task 13: Wire Phase 2 into boot.py

**Files:**
- Modify: `sage-python/src/sage/boot.py`
- Modify: `sage-core/src/lib.rs`

**Implementation:** Register new PyClasses. Wire TemplateStore + HybridVerifier into SystemRouter. Python topology kept as shadow.

---

## Phase 3: Contextual Bandit + Persistence

### Task 14: ContextualBandit with per-arm posteriors

**Files:**
- Create: `sage-core/src/routing/bandit.rs`
- Test: `sage-core/tests/test_bandit.rs`

**Implementation:** Per-arm `ArmPosterior` (Beta for quality, Gamma for cost/latency). `ContextualBandit` builds global Pareto front at decision time from current posteriors. LinUCB-inspired context features. `select(system, constraints, features)` returns RoutingDecision. `record(decision_id, quality, cost, latency)` updates posteriors with decay.

---

### Task 15: SQLite persistence for ContextualBandit

**Files:**
- Create: `sage-core/src/routing/persistence.rs`

**Implementation:** `save_to_sqlite(path)` / `load_from_sqlite(path)` for ContextualBandit. Schema: `arms(model_id, template, quality_alpha, quality_beta, cost_shape, cost_rate, latency_shape, latency_rate, obs_count, updated_at)`. WAL mode + MPSC async flush. DashMap as hot state, SQLite as durable truth.

---

### Task 16: S-MMU integration + tracing

**Files:**
- Modify: `sage-core/src/routing/system_router.rs`

**Implementation:** On `record()`, register template as S-MMU chunk with task embedding. On `route()`, query S-MMU for similar past tasks → retrieve best template. Add `tracing` spans for all routing decisions.

---

## Phase 4: S-MMU Topology Retrieval + MAP-Elites + Custom DAGs

### Task 17a: S-MMU topology bridge

**Files:**
- Create: `sage-core/src/topology/smmu_bridge.rs`
- Test: `sage-core/tests/test_smmu_bridge.rs`

**Implementation:** Write path: on `record_outcome()`, store `TopologyChunk` (task_embedding + topology_id + quality/cost/latency + structural features) as S-MMU chunk via `register_chunk()`. Read path: on `route()`, query S-MMU with task embedding for top-5 similar past tasks, return `(topology_id, quality, similarity_score)`. Inject as prior into contextual bandit to boost arms that worked for similar tasks.

This creates the **task→topology learning feedback loop** — the core mechanism for adaptive topology selection.

---

### Task 17b: MAP-Elites archive

**Files:**
- Create: `sage-core/src/topology/map_elites.rs`
- Test: `sage-core/tests/test_map_elites.rs`

**Implementation:** N-dimensional grid archive with `BehaviorDescriptor` (agent_count_bucket, max_depth_bucket, cost_bucket, model_diversity_bucket). Each cell holds best `TopologyGraph` by Pareto dominance. Insert validates via hybrid verifier (DAG + capabilities + budget + optional OxiZ for security labels). `save()`/`load()` to SQLite for cross-session persistence.

---

### Task 17c: 7 mutation operators

**Files:**
- Create: `sage-core/src/topology/mutations.rs`
- Test: `sage-core/tests/test_mutations.rs`

**Implementation:** 7 operators: `add_node` (insert agent with model from registry), `remove_node` (prune + rewire), `swap_model` (change model_id), `rewire_edge` (add/remove/redirect), `split_node` (one→two specialized), `merge_nodes` (two→one generalist), `mutate_prompt` (LLM-guided via PyO3 callback, async). Every mutation product validated by hybrid verifier — invalid mutations discarded, not retried.

---

### Task 17d: DynamicTopologyEngine

**Files:**
- Create: `sage-core/src/topology/engine.rs`
- Test: `sage-core/tests/test_engine.rs`

**Implementation:** Ties together S-MMU retrieval, archive lookup, mutation, LLM synthesis, and evolution.
- `generate(task, system, constraints)`: 5-path strategy: S-MMU hit → archive hit → LLM synthesis (if exploration_budget > 0.3) → mutation → template fallback.
- `evolve(pop_size, generations)`: async/background. Sample N from archive, apply 1-3 mutations each, validate, estimate fitness via surrogate model or S-MMU replay, insert survivors. Called every 100 requests or on explicit trigger.
- `record_outcome(topology_id, task_embedding, quality, cost, latency)`: feeds S-MMU + archive + bandit.
- `get_topology(id)`: lazy-load for opaque Ulid from RoutingDecision.

This is where **task-adaptive custom DAGs** emerge: mutations on archive entries AND LLM-synthesized topologies create novel topologies (e.g., "coder→tester→fixer" loop discovered by evolution, or a hub-and-spoke delegator synthesized by LLM for a complex multi-domain task).

---

### Task 17e: LLM-synthesized topology generation

**Files:**
- Create: `sage-core/src/topology/llm_synthesis.rs`
- Test: `sage-core/tests/test_llm_synthesis.rs`

**Implementation:** `TopologySynthesizer` with 3-stage pipeline (inspired by MASFactory Vibe Graphing, arXiv 2603.06007):
1. **Role Assignment**: LLM decomposes task into agent roles + required capabilities
2. **Structure Design**: LLM generates adjacency matrix + node specs (JSON output format — LLMs produce matrices more reliably than edge lists)
3. **Validation + Instantiation**: Parse JSON → TopologyGraph → HybridVerifier + semantic diagnosis → accept or discard

LLM call via PyO3 callback (`py.allow_threads()` released). Result cached in MAP-Elites archive + S-MMU. Rate-limited to 1/min, budget deducted from RoutingConstraints. Only invoked when S-MMU miss + archive miss + exploration_budget > 0.3 (~5-10% of requests).

---

### Task 17f: Dual-mode topology executor

**Files:**
- Create: `sage-core/src/topology/executor.rs`
- Test: `sage-core/tests/test_executor.rs`

**Implementation:** `TopologyExecutor` with two modes:
- **Static** (petgraph `Topo` iterator): Deterministic O(V+E) topological ordering. Used for Sequential, Parallel, Hierarchical, Brainstorming.
- **Dynamic** (gate-based readiness polling): Nodes fire when all input gates are Open. Supports loops (controller reopens gates), switches (closes non-taken branches), handoffs (activates/deactivates spokes). Used for AVR, Hub, Debate, SelfMoA, Custom. Safety limit: max_iterations (default 1000).

`mode_for(template)` auto-selects execution mode. `next_ready(graph)` returns next executable node(s). Python `AgentLoop` calls `next_ready()` in a loop, keeping scheduling in Rust while LLM calls stay in Python.

---

## Phase 5: Python Cleanup (only after all evidence gates passed)

### Task 18: Freeze + delete Python routing modules

**Prerequisite:** Shadow mode divergence < 5% on 1000+ traces, downstream quality ≥ baseline.

**Files to freeze-then-delete:**
- `sage-python/src/sage/strategy/metacognition.py`
- `sage-python/src/sage/strategy/adaptive_router.py`
- `sage-python/src/sage/strategy/training.py`
- `sage-python/src/sage/topology/evo_topology.py`
- `sage-python/src/sage/topology/engine.py`
- `sage-python/src/sage/topology/patterns.py`
- `sage-python/src/sage/strategy/solvers.py`
- `sage-python/src/sage/evolution/` (all files)

---

### Task 21: Embedding model upgrade (arctic-embed-m)

**Files:**
- Modify: `sage-core/models/download_model.py` — MODEL_ID → "Snowflake/snowflake-arctic-embed-m"
- Modify: `sage-core/src/memory/embedder.rs` — update docs, dimension 384→768
- Modify: `sage-python/src/sage/memory/embedder.py` — EMBEDDING_DIM=768, _MODEL_NAME
- Modify: `sage-core/src/memory/mod.rs` — update EMBEDDING_DIM constant if present
- Test all embedder tests pass with new dimension

**Note:** This task can be done independently at any point. Dimension change from 384→768 requires clearing existing S-MMU data (one-time migration).

---

### Task 22: Final integration test + benchmark

**Step 1:** Run full Python test suite: `cd sage-python && python -m pytest tests/ -v`
**Step 2:** Run routing quality benchmark: `python -m sage.bench --type routing`
**Step 3:** Run E2E proof: `python tests/e2e_proof.py`
**Step 4:** Run Rust test suite: `cd sage-core && cargo test --features cognitive,onnx`
**Step 5:** Run shadow divergence report: verify < 5% on collected traces
**Step 6:** Run downstream quality comparison: verify ≥ baseline
**Step 7:** Update CLAUDE.md with new architecture docs
**Step 8:** Final commit + tag

```bash
git tag v0.2.0-cognitive-engine
```
