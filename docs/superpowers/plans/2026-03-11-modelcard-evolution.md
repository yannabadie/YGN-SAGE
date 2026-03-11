# ModelCard Evolution — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evolve ModelCard from a static TOML descriptor to an intelligent, telemetry-aware capability profile with domain affinity, safety ratings, and a `select_best_for_domain()` API — enabling truly informed routing ("this math task → Gemini 2.5 Pro with 94% GSM8K").

**Architecture:** Extend Rust `ModelCard` with 3 new fields (`domain_scores`, `safety_rating`, `observed_latency_p95`). Extend `ModelRegistry` with `select_best_for_domain(domain, budget)`. Extend `TelemetryRecord` with latency tracking. Wire `record_outcome` to update both registry telemetry AND bandit. Update `cards.toml` with domain scores for all 19 models.

**Tech Stack:** Rust (PyO3 0.25, serde, toml), Python 3.12 (sage-python)

---

## File Structure

| File | Change | Responsibility |
|------|--------|---------------|
| `sage-core/src/routing/model_card.rs` | Modify | Add `domain_scores`, `safety_rating` fields |
| `sage-core/src/routing/model_registry.rs` | Modify | Add `select_best_for_domain()`, extend `TelemetryRecord` with latency |
| `sage-core/src/routing/system_router.rs` | Modify | Add `domain_hint` to RoutingConstraints, wire into `route_integrated()`, record_outcome telemetry via decision→model map |
| `sage-core/config/cards.toml` | Modify | Add domain_scores + safety_rating for all 19 models |

---

### Task 1: Add Domain Scores & Safety Rating to ModelCard (Rust)

**Files:**
- Modify: `sage-core/src/routing/model_card.rs:51-135`

- [ ] **Step 1: Write the failing test**

Add to `sage-core/src/routing/model_card.rs` test module:

```rust
#[test]
fn domain_scores_default_empty() {
    let card = make_test_card(0.5, 0.5, 0.5);
    assert!(card.domain_scores.is_empty());
    assert!((card.safety_rating - 0.5).abs() < 0.001); // default
}

#[test]
fn domain_score_for_known_domain() {
    let mut card = make_test_card(0.5, 0.5, 0.5);
    card.domain_scores.insert("math".to_string(), 0.94);
    card.domain_scores.insert("code".to_string(), 0.87);
    assert!((card.domain_score("math") - 0.94).abs() < 0.001);
    assert!((card.domain_score("code") - 0.87).abs() < 0.001);
    assert!((card.domain_score("unknown") - 0.5).abs() < 0.001); // fallback
}

#[test]
fn parse_toml_with_domain_scores() {
    let toml_str = r#"
        [[models]]
        id = "test-model"
        provider = "test"
        family = "test"
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
        recommended_topologies = ["sequential"]
        supports_tools = true
        supports_json_mode = true
        supports_vision = false
        context_window = 1048576
        safety_rating = 0.85

        [models.domain_scores]
        math = 0.94
        code = 0.87
        reasoning = 0.80
    "#;
    let cards = ModelCard::parse_toml(toml_str).unwrap();
    assert_eq!(cards.len(), 1);
    assert!((cards[0].safety_rating - 0.85).abs() < 0.001);
    assert!((cards[0].domain_score("math") - 0.94).abs() < 0.001);
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd sage-core && cargo test --no-default-features --lib -- model_card::tests::domain
```
Expected: FAIL (fields don't exist)

- [ ] **Step 3: Add fields to ModelCard struct**

In `sage-core/src/routing/model_card.rs`, add to ModelCard struct after `context_window`:

```rust
    // ── Domain-specific scores (0.0–1.0) ──────────────────────────────
    /// Per-domain benchmark scores (e.g. "math" → 0.94, "code" → 0.87).
    /// Parsed from TOML `[models.domain_scores]` sub-table.
    #[pyo3(get)]
    #[serde(default)]
    pub domain_scores: HashMap<String, f32>,

    /// Safety rating (0.0–1.0). Higher = safer output, fewer refusals.
    #[pyo3(get)]
    #[serde(default = "default_safety")]
    pub safety_rating: f32,
```

Add at top of file:

```rust
use std::collections::HashMap;
```

Add default function:

```rust
fn default_safety() -> f32 { 0.5 }
```

Add PyO3 method:

```rust
    /// Get domain-specific score. Returns 0.5 (neutral) if domain unknown.
    pub fn domain_score(&self, domain: &str) -> f32 {
        self.domain_scores.get(domain).copied().unwrap_or(0.5)
    }
```

Update `make_test_card()` in tests to include `domain_scores: HashMap::new()` and `safety_rating: 0.5`.

- [ ] **Step 4: Run test to verify it passes**

```bash
cd sage-core && cargo test --no-default-features --lib -- model_card::tests
```
Expected: ALL pass

- [ ] **Step 5: Run clippy**

```bash
cd sage-core && cargo clippy --no-default-features -- -D warnings
```
Expected: No warnings

- [ ] **Step 6: Commit**

```bash
git add sage-core/src/routing/model_card.rs
git commit -m "feat(routing): add domain_scores HashMap + safety_rating to ModelCard"
```

---

### Task 2: Extend TelemetryRecord with Latency Tracking (Rust)

**Files:**
- Modify: `sage-core/src/routing/model_registry.rs:1-26`

**Reviewer fixes applied:**
- Use `VecDeque<f32>` instead of `Vec<f32>` for O(1) `pop_front()` (was O(n) `remove(0)`)
- Guard latency push: `if latency_ms > 0.0` — prevents `record_telemetry()` delegation from corrupting P95 with 0.0 values

- [ ] **Step 1: Write the failing test**

Add to `sage-core/src/routing/model_registry.rs` test module:

```rust
#[test]
fn telemetry_tracks_latency() {
    let mut reg = ModelRegistry::from_toml_str(test_toml()).unwrap();
    reg.record_telemetry_full("fast-model", 0.8, 0.01, 150.0);
    reg.record_telemetry_full("fast-model", 0.9, 0.02, 250.0);
    let latency = reg.observed_latency_p95("fast-model");
    // With 2 samples, p95 should be close to max (250.0)
    assert!(latency > 200.0);
}

#[test]
fn observed_latency_unknown_model_returns_zero() {
    let reg = ModelRegistry::from_toml_str(test_toml()).unwrap();
    assert!((reg.observed_latency_p95("nonexistent") - 0.0).abs() < 0.001);
}

#[test]
fn record_telemetry_without_latency_does_not_corrupt_p95() {
    let mut reg = ModelRegistry::from_toml_str(test_toml()).unwrap();
    // record_telemetry (no latency) should NOT push 0.0 into buffer
    reg.record_telemetry("fast-model", 0.8, 0.01);
    reg.record_telemetry("fast-model", 0.9, 0.02);
    // Now record one real latency
    reg.record_telemetry_full("fast-model", 0.7, 0.01, 200.0);
    // P95 should be 200.0 (the only real sample), not corrupted by 0.0
    let latency = reg.observed_latency_p95("fast-model");
    assert!((latency - 200.0).abs() < 0.001);
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd sage-core && cargo test --no-default-features --lib -- model_registry::tests::telemetry_tracks
```
Expected: FAIL

- [ ] **Step 3: Extend TelemetryRecord**

Add at top of file:

```rust
use std::collections::VecDeque;
```

Replace the `TelemetryRecord` struct and impl:

```rust
#[derive(Debug, Clone, Default)]
pub struct TelemetryRecord {
    pub quality_sum: f64,
    pub cost_sum: f64,
    pub count: u32,
    /// Recent latencies for P95 estimation (bounded ring buffer, last 100).
    /// Uses VecDeque for O(1) pop_front.
    pub latencies: VecDeque<f32>,
}

impl TelemetryRecord {
    pub fn avg_quality(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.quality_sum / self.count as f64 }
    }

    /// Approximate P95 latency from recent samples.
    pub fn latency_p95(&self) -> f32 {
        if self.latencies.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f32> = self.latencies.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((sorted.len() as f64) * 0.95).ceil() as usize;
        sorted[idx.min(sorted.len()) - 1]
    }
}
```

Add methods to ModelRegistry (in the `impl ModelRegistry` block):

```rust
    /// Record telemetry with latency (full version).
    pub fn record_telemetry_full(&mut self, model_id: &str, quality: f32, cost: f32, latency_ms: f32) {
        let record = self.telemetry.entry(model_id.to_string()).or_default();
        record.quality_sum += quality as f64;
        record.cost_sum += cost as f64;
        record.count += 1;
        // Only record real latencies — guard against 0.0 from record_telemetry() delegation
        if latency_ms > 0.0 {
            if record.latencies.len() >= 100 {
                record.latencies.pop_front(); // O(1) with VecDeque
            }
            record.latencies.push_back(latency_ms);
        }
        info!(model = model_id, count = record.count, "telemetry_recorded");
    }

    /// Get observed P95 latency for a model (0.0 if no observations).
    #[pyo3(name = "observed_latency_p95")]
    pub fn py_observed_latency_p95(&self, model_id: &str) -> f32 {
        self.observed_latency_p95(model_id)
    }

    pub fn observed_latency_p95(&self, model_id: &str) -> f32 {
        self.telemetry.get(model_id).map(|r| r.latency_p95()).unwrap_or(0.0)
    }
```

Update existing `record_telemetry` to delegate (latency_ms=0.0 will be guarded):

```rust
    pub fn record_telemetry(&mut self, model_id: &str, quality: f32, cost: f32) {
        self.record_telemetry_full(model_id, quality, cost, 0.0);
    }
```

- [ ] **Step 4: Run tests**

```bash
cd sage-core && cargo test --no-default-features --lib -- model_registry
```
Expected: ALL pass

- [ ] **Step 5: Commit**

```bash
git add sage-core/src/routing/model_registry.rs
git commit -m "feat(routing): extend TelemetryRecord with latency P95 tracking (VecDeque ring buffer, last 100)"
```

---

### Task 3: Add `select_best_for_domain()` to ModelRegistry (Rust)

**Files:**
- Modify: `sage-core/src/routing/model_registry.rs`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn select_best_for_domain_uses_domain_score() {
    // Two models: one with high math score, one without
    let toml_str = r#"
        [[models]]
        id = "math-model"
        provider = "test"
        family = "test"
        code_score = 0.5
        reasoning_score = 0.5
        tool_use_score = 0.5
        math_score = 0.5
        formal_z3_strength = 0.5
        cost_input_per_m = 1.0
        cost_output_per_m = 2.0
        latency_ttft_ms = 500.0
        tokens_per_sec = 100.0
        s1_affinity = 0.5
        s2_affinity = 0.5
        s3_affinity = 0.5
        recommended_topologies = []
        supports_tools = true
        supports_json_mode = false
        supports_vision = false
        context_window = 128000
        safety_rating = 0.8
        [models.domain_scores]
        math = 0.94

        [[models]]
        id = "general-model"
        provider = "test"
        family = "test"
        code_score = 0.5
        reasoning_score = 0.5
        tool_use_score = 0.5
        math_score = 0.5
        formal_z3_strength = 0.5
        cost_input_per_m = 0.5
        cost_output_per_m = 1.0
        latency_ttft_ms = 200.0
        tokens_per_sec = 200.0
        s1_affinity = 0.5
        s2_affinity = 0.5
        s3_affinity = 0.5
        recommended_topologies = []
        supports_tools = true
        supports_json_mode = false
        supports_vision = false
        context_window = 128000
        safety_rating = 0.7
    "#;
    let reg = ModelRegistry::from_toml_str(toml_str).unwrap();
    // For math domain, math-model should win (0.94 > 0.5 default)
    let best = reg.select_best_for_domain("math", 10.0);
    assert!(best.is_some());
    assert_eq!(best.unwrap().id, "math-model");

    // For unknown domain, should return cheapest (general-model at $0.5/M)
    let best_gen = reg.select_best_for_domain("unknown", 10.0);
    assert!(best_gen.is_some());
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd sage-core && cargo test --no-default-features --lib -- model_registry::tests::select_best_for_domain
```

- [ ] **Step 3: Implement `select_best_for_domain()`**

**Reviewer fixes applied:**
- Infer CognitiveSystem from domain ("math"/"formal" → S3, "code"/"reasoning" → S2, else → S1) instead of using `card.best_system()` which picks the model's strongest system regardless of domain
- Use `0.001_f32` instead of `f32::MIN` for max_cost floor (f32::MIN is ~1.2e-38, not negative, but confusing and overly small)

```rust
    /// Select best model for a given task domain within budget.
    ///
    /// Scoring: domain_score * 0.6 + calibrated_system_affinity * 0.3 + (1 - cost_norm) * 0.1
    /// System inferred from domain: math/formal → S3, code/reasoning → S2, else → S1.
    /// Budget filter: estimate_cost(1000, 500) must be <= max_cost_usd.
    #[pyo3(name = "select_best_for_domain")]
    pub fn py_select_best_for_domain(&self, domain: &str, max_cost_usd: f32) -> Option<ModelCard> {
        self.select_best_for_domain(domain, max_cost_usd)
    }

    pub fn select_best_for_domain(&self, domain: &str, max_cost_usd: f32) -> Option<ModelCard> {
        let mut candidates: Vec<_> = self.cards.values()
            .filter(|c| max_cost_usd <= 0.0 || c.estimate_cost(1000, 500) <= max_cost_usd)
            .cloned()
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // Find max cost for normalization (floor at 0.001 to avoid division issues)
        let max_cost = candidates.iter()
            .map(|c| c.estimate_cost(1000, 500))
            .fold(0.001_f32, f32::max);

        candidates.sort_by(|a, b| {
            let score_a = self.domain_routing_score(a, domain, max_cost);
            let score_b = self.domain_routing_score(b, domain, max_cost);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates.into_iter().next()
    }
```

Add helpers:

```rust
    /// Infer CognitiveSystem from a domain name.
    /// "math" / "formal" → S3, "code" / "reasoning" / "tool_use" → S2, else → S1.
    fn system_for_domain(domain: &str) -> CognitiveSystem {
        match domain {
            "math" | "formal" => CognitiveSystem::S3,
            "code" | "reasoning" | "tool_use" => CognitiveSystem::S2,
            _ => CognitiveSystem::S1,
        }
    }

    fn domain_routing_score(&self, card: &ModelCard, domain: &str, max_cost: f32) -> f32 {
        let domain_score = card.domain_score(domain);
        let system = Self::system_for_domain(domain);
        let affinity = self.calibrated_affinity(&card.id, system);
        let cost_norm = card.estimate_cost(1000, 500) / max_cost;
        domain_score * 0.6 + affinity * 0.3 + (1.0 - cost_norm) * 0.1
    }
```

- [ ] **Step 4: Run tests**

```bash
cd sage-core && cargo test --no-default-features --lib -- model_registry
```
Expected: ALL pass

- [ ] **Step 5: Commit**

```bash
git add sage-core/src/routing/model_registry.rs
git commit -m "feat(routing): add select_best_for_domain() with domain-aware scoring to ModelRegistry"
```

---

### Task 4: Wire Domain Scoring into SystemRouter (Rust)

**Files:**
- Modify: `sage-core/src/routing/system_router.rs:46-87` (struct + constructor)
- Modify: `sage-core/src/routing/system_router.rs:326-418` (route_integrated)

**Reviewer fixes applied:**
- **CRITICAL**: Add `domain_hint=String::new()` to `#[pyo3(signature)]` on `RoutingConstraints::new()` — without this, all existing Python call sites and Rust tests break (they don't pass `domain_hint`)
- Specify exact insertion point: domain hint override goes AFTER constraint filtering (Step 2, line ~354) and BEFORE bandit selection (Step 3, line ~357), replacing the candidate list with a domain-preferred ordering
- No `selected_model` variable exists — the domain override reorders `candidates` so that subsequent budget/bandit selection picks from domain-preferred models first

- [ ] **Step 1: Write the failing test**

Add to SystemRouter tests:

```rust
#[test]
fn route_integrated_uses_domain_hint() {
    let toml_str = r#"
        [[models]]
        id = "math-model"
        provider = "test"
        family = "test"
        code_score = 0.5
        reasoning_score = 0.5
        tool_use_score = 0.5
        math_score = 0.5
        formal_z3_strength = 0.5
        cost_input_per_m = 1.0
        cost_output_per_m = 2.0
        latency_ttft_ms = 500.0
        tokens_per_sec = 100.0
        s1_affinity = 0.5
        s2_affinity = 0.5
        s3_affinity = 0.9
        recommended_topologies = []
        supports_tools = true
        supports_json_mode = false
        supports_vision = false
        context_window = 128000
        safety_rating = 0.8
        [models.domain_scores]
        math = 0.94

        [[models]]
        id = "general-model"
        provider = "test"
        family = "test"
        code_score = 0.5
        reasoning_score = 0.5
        tool_use_score = 0.5
        math_score = 0.5
        formal_z3_strength = 0.5
        cost_input_per_m = 0.1
        cost_output_per_m = 0.2
        latency_ttft_ms = 100.0
        tokens_per_sec = 200.0
        s1_affinity = 0.9
        s2_affinity = 0.5
        s3_affinity = 0.2
        recommended_topologies = ["sequential"]
        supports_tools = true
        supports_json_mode = false
        supports_vision = false
        context_window = 128000
    "#;
    let reg = ModelRegistry::from_toml_str(toml_str).unwrap();
    let mut router = SystemRouter::new(reg);

    // Without domain hint — no domain preference
    let no_hint = RoutingConstraints::new(0.0, 0.0, 0.0, vec![], String::new(), 0.0, String::new());
    let d1 = router.route_integrated("prove that sqrt(2) is irrational", &no_hint, "").unwrap();
    assert!(!d1.model_id.is_empty());

    // With math domain hint — should prefer math-model
    let math_hint = RoutingConstraints::new(0.0, 0.0, 0.0, vec![], String::new(), 0.0, "math".to_string());
    let d2 = router.route_integrated("prove that sqrt(2) is irrational", &math_hint, "").unwrap();
    assert_eq!(d2.model_id, "math-model");
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd sage-core && cargo test --no-default-features --lib -- system_router::tests::route_integrated_uses_domain
```
Expected: FAIL (field `domain_hint` doesn't exist)

- [ ] **Step 3: Add `domain_hint` field to RoutingConstraints**

In `sage-core/src/routing/system_router.rs`, add to `RoutingConstraints` struct (after `exploration_budget`):

```rust
    /// Optional domain hint for domain-aware model selection (e.g. "math", "code").
    /// Empty string = no hint (default scoring).
    #[pyo3(get, set)]
    pub domain_hint: String,
```

**CRITICAL**: Update the `#[pyo3(signature)]` on `new()` to include `domain_hint` with a default:

```rust
    #[new]
    #[pyo3(signature = (max_cost_usd=0.0, max_latency_ms=0.0, min_quality=0.0, required_capabilities=vec![], security_label=String::new(), exploration_budget=0.0, domain_hint=String::new()))]
    pub fn new(
        max_cost_usd: f32,
        max_latency_ms: f32,
        min_quality: f32,
        required_capabilities: Vec<String>,
        security_label: String,
        exploration_budget: f32,
        domain_hint: String,
    ) -> Self {
        Self {
            max_cost_usd,
            max_latency_ms,
            min_quality,
            required_capabilities,
            security_label,
            exploration_budget,
            domain_hint,
        }
    }
```

Also update `__repr__` to include domain_hint.

- [ ] **Step 4: Use domain hint in route_integrated()**

In `route_integrated()`, insert AFTER constraint filtering (after line 354 `candidates = self.registry.all_models();`) and BEFORE bandit selection (before line 357 `let (model_id, estimated_cost, decision_id) = if let Some(...)`):

```rust
        // Step 2.5: Domain hint — reorder candidates by domain preference
        if !constraints.domain_hint.is_empty() {
            candidates.sort_by(|a, b| {
                let sa = a.domain_score(&constraints.domain_hint);
                let sb = b.domain_score(&constraints.domain_hint);
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });
        }
```

This reorders candidates so that when bandit has no opinion or fallback to budget selection is used, the domain-preferred model is selected first. If the bandit overrides with its own choice, that's fine — bandit has learned data.

- [ ] **Step 5: Run tests**

```bash
cd sage-core && cargo test --no-default-features --lib -- system_router
```
Expected: ALL pass (existing tests use `RoutingConstraints::new()` which now has `domain_hint=String::new()` default — no breakage)

- [ ] **Step 6: Commit**

```bash
git add sage-core/src/routing/system_router.rs
git commit -m "feat(routing): add domain_hint to RoutingConstraints, wire into route_integrated()"
```

---

### Task 5: Update cards.toml with Domain Scores

**Files:**
- Modify: `sage-core/config/cards.toml`

- [ ] **Step 1: Add domain_scores and safety_rating to all 19 models**

For each model, add based on known benchmark performance:

```toml
# Example for gemini-3.1-pro-preview:
safety_rating = 0.85
[models.domain_scores]
math = 0.83
code = 0.81
reasoning = 0.92
tool_use = 0.85
formal = 0.64

# Example for gpt-5.3-codex:
safety_rating = 0.80
[models.domain_scores]
math = 0.78
code = 0.95
reasoning = 0.85
tool_use = 0.90
formal = 0.70
```

Domain keys: `math`, `code`, `reasoning`, `tool_use`, `formal`, `creative`, `factual`.
Safety ratings: conservative estimates (0.5 default, higher for models with known safety tuning).

- [ ] **Step 2: Verify TOML parses correctly**

```bash
cd sage-core && cargo test --no-default-features --lib -- model_card::tests::parse_toml
```

- [ ] **Step 3: Commit**

```bash
git add sage-core/config/cards.toml
git commit -m "feat(routing): add domain_scores + safety_rating to all 19 model cards"
```

---

### Task 6: Wire record_outcome to Update Registry Telemetry (Rust-side)

**Files:**
- Modify: `sage-core/src/routing/system_router.rs` (SystemRouter struct + record_outcome + route_integrated)

**Reviewer fixes applied:**
- `self.rust_registry` does NOT exist on the Python `AgentSystem` dataclass — it's a local variable in `build_agent_system()`. Wiring from Python would require either storing it on AgentSystem or exposing it through SystemRouter.
- **Cleanest fix**: Handle telemetry entirely inside Rust `record_outcome()`. SystemRouter already owns the `ModelRegistry`. We just need to store a `decision_id → model_id` mapping so `record_outcome()` can look up which model to record telemetry for.

- [ ] **Step 1: Write the failing test**

Add to SystemRouter tests:

```rust
#[test]
fn record_outcome_updates_registry_telemetry() {
    let toml_str = /* use test_toml with domain_scores */;
    let reg = ModelRegistry::from_toml_str(toml_str).unwrap();
    let mut router = SystemRouter::new(reg);
    let constraints = RoutingConstraints::new(0.0, 0.0, 0.0, vec![], String::new(), 0.0, String::new());
    let decision = router.route_integrated("test task", &constraints, "").unwrap();
    let decision_id = decision.decision_id.clone();
    let model_id = decision.model_id.clone();

    // Record outcome
    router.record_outcome(&decision_id, 0.9, 0.05, 150.0).unwrap();

    // Verify registry telemetry was updated
    let p95 = router.registry.observed_latency_p95(&model_id);
    assert!((p95 - 150.0).abs() < 0.001);
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd sage-core && cargo test --no-default-features --lib -- system_router::tests::record_outcome_updates
```
Expected: FAIL (record_outcome doesn't update registry, and `observed_latency_p95` doesn't exist yet)

- [ ] **Step 3: Add decision→model mapping to SystemRouter**

In SystemRouter struct, add a bounded lookup map:

```rust
use std::collections::VecDeque;

pub struct SystemRouter {
    registry: ModelRegistry,
    bandit: Option<ContextualBandit>,
    /// Recent decision_id → model_id mapping for record_outcome telemetry.
    /// Bounded to last 1000 decisions.
    decision_models: HashMap<String, String>,
    decision_order: VecDeque<String>,
}
```

Update `SystemRouter::new()` to initialize the new fields:

```rust
    pub fn new(registry: ModelRegistry) -> Self {
        Self {
            registry,
            bandit: None,
            decision_models: HashMap::new(),
            decision_order: VecDeque::new(),
        }
    }
```

- [ ] **Step 4: Store mapping in route_integrated()**

At the end of `route_integrated()`, before returning the decision, store the mapping:

```rust
        // Store decision → model mapping for record_outcome telemetry
        if self.decision_models.len() >= 1000 {
            if let Some(old_id) = self.decision_order.pop_front() {
                self.decision_models.remove(&old_id);
            }
        }
        self.decision_models.insert(decision.decision_id.clone(), decision.model_id.clone());
        self.decision_order.push_back(decision.decision_id.clone());
```

- [ ] **Step 5: Update record_outcome() to call registry.record_telemetry_full()**

Replace the comment at lines 445-447 with actual registry telemetry recording:

```rust
    pub fn record_outcome(
        &mut self,
        decision_id: &str,
        quality: f32,
        cost: f32,
        latency_ms: f32,
    ) -> Result<(), String> {
        // ... existing span + bandit forwarding ...

        // Record telemetry on registry using stored decision→model mapping
        if let Some(model_id) = self.decision_models.get(decision_id) {
            self.registry.record_telemetry_full(model_id, quality, cost, latency_ms);
        } else {
            info!(decision_id = decision_id, "no_model_mapping_for_telemetry");
        }

        Ok(())
    }
```

- [ ] **Step 6: Run tests**

```bash
cd sage-core && cargo test --no-default-features --lib -- system_router
```
Expected: ALL pass

- [ ] **Step 7: Commit**

```bash
git add sage-core/src/routing/system_router.rs
git commit -m "feat(routing): record_outcome now updates registry telemetry via decision→model mapping"
```

---

### Task 7: Update CLAUDE.md and Documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update ModelCard description in CLAUDE.md**

Add to the model_card.rs description:
```
- `routing/model_card.rs` - ModelCard + CognitiveSystem + domain_scores (HashMap<String, f32>) + safety_rating. TOML parsing with `[models.domain_scores]` sub-tables. PyO3 class with `domain_score(domain)` method.
```

Add to the model_registry.rs description:
```
- `routing/model_registry.rs` - ModelRegistry with telemetry calibration (quality + latency P95 ring buffer), `select_best_for_domain(domain, budget)` for domain-aware model selection. PyO3 class.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with ModelCard domain_scores + select_best_for_domain"
```

---

## Summary

| Task | What | LOC Changed | Risk |
|------|------|-------------|------|
| 1. Domain scores + safety_rating | New ModelCard fields | ~40 Rust | Low |
| 2. Latency P95 tracking | TelemetryRecord extension (VecDeque, guard) | ~60 Rust | Low |
| 3. select_best_for_domain() | New ModelRegistry API (domain→system inference) | ~70 Rust | Low |
| 4. Domain hint in SystemRouter | RoutingConstraints + route_integrated domain sort | ~50 Rust | Medium |
| 5. Update cards.toml | Domain scores for 19 models | ~100 TOML | None |
| 6. record_outcome telemetry | Rust-side decision→model map + registry.record_telemetry_full() | ~40 Rust | Low |
| 7. Documentation | CLAUDE.md update | ~10 docs | None |

**Total:** ~370 LOC added. ModelCard goes from "static TOML descriptor" to "telemetry-calibrated, domain-aware capability profile." All telemetry handled in Rust — no Python wiring needed.
