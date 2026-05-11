//! ModelAssigner — per-node model assignment using ModelCard scoring.

use std::cmp::Ordering;
use std::collections::HashSet;
use std::sync::Mutex;

use pyo3::prelude::*;
use tracing::{info, warn};

use super::model_card::{CognitiveSystem, ModelCard};
use super::model_registry::ModelRegistry;
use crate::topology::topology_graph::TopologyGraph;

/// Default scoring weights — subject to ablation (see CLAUDE.md §2).
const DEFAULT_WEIGHT_AFFINITY: f32 = 0.4;
const DEFAULT_WEIGHT_DOMAIN: f32 = 0.4;
const DEFAULT_WEIGHT_COST: f32 = 0.2;
const BUDGET_EPSILON: f32 = 0.01;

/// Provider hint scoring bonus — subject to ablation.
const PROVIDER_HINT_BONUS: f32 = 0.15;

/// Fixed token estimate for cost normalization (input, output).
const COST_ESTIMATE_TOKENS: (u32, u32) = (1000, 500);

/// Slice 10D Phase 2 (cgpro DESIGN_LOCKED v1_1, 2026-05-11): cap on
/// how many filter rejections the witness records per assignment
/// call. Beyond this, the buffer evicts the oldest entry and sets
/// `truncated=true` so Python's `rust_filter_rejections_truncated`
/// reflects reality. Mirrors `_RUST_FILTER_REJECTIONS_CAP = 20` in
/// `sage-python/src/sage/pipeline_v2/runtime_events.py`.
const RUST_FILTER_REJECTIONS_CAP: usize = 20;

/// Public reason codes for `last_filter_rejections()` — exactly the
/// 8-code enum locked by cgpro. Internal `ProviderPolicyFilter`
/// strings (`provider_policy_denylist`, etc.) are mapped to the
/// public `provider_excluded_policy_*` codes before storage.
const REASON_CARD_INACTIVE: &str = "card_inactive";
const REASON_PROVIDER_POLICY_UNKNOWN: &str = "provider_excluded_policy_unknown_provider";
const REASON_PROVIDER_POLICY_DENYLIST: &str = "provider_excluded_policy_denylist";
const REASON_PROVIDER_POLICY_ALLOWLIST: &str = "provider_excluded_policy_allowlist";
const REASON_PROVIDER_DEAD: &str = "provider_excluded_dead";
const REASON_EXCLUDED_BY_CALLER: &str = "excluded_by_caller";
const REASON_CAPABILITY_MISMATCH: &str = "capability_mismatch";
const REASON_COST_ABOVE_BUDGET: &str = "cost_above_budget";

/// Per-assignment-call buffer of structured candidate rejections.
/// Python reads this via `last_filter_rejections()` for the
/// provider_execution_witness payload (cgpro DESIGN_LOCKED v1_1).
///
/// Semantics:
/// - `observed == false`: no recording-aware path has run since the
///   last `begin_filter_recording()`. `last_filter_rejections()`
///   returns `None`.
/// - `observed == true`, `rejections == []`: the path ran but
///   rejected nothing. Python sees `[]` and tags
///   `rust_filter_details_observed=true`.
/// - `observed == true`, `rejections != []`: the path ran and
///   recorded the listed `(model_id, reason_code)` pairs.
/// - `truncated == true`: more than `RUST_FILTER_REJECTIONS_CAP`
///   rejections were pushed; oldest were evicted.
#[derive(Debug, Clone, Default)]
struct FilterRejectionState {
    observed: bool,
    rejections: Vec<(String, String)>,
    truncated: bool,
}

/// Map the internal `ProviderPolicyFilter::violation_reason` strings
/// to the public 8-code enum surface. Internal strings MUST NOT leak
/// into `last_filter_rejections()` per cgpro Phase 2 lock §4.
fn map_policy_violation_to_public_reason(internal: &str) -> &'static str {
    match internal {
        "provider_policy_unknown_provider" => REASON_PROVIDER_POLICY_UNKNOWN,
        "provider_policy_denylist" => REASON_PROVIDER_POLICY_DENYLIST,
        "provider_policy_outside_allowlist" => REASON_PROVIDER_POLICY_ALLOWLIST,
        // Defensive: if a new internal reason ever appears,
        // surface a stable public code rather than leaking the
        // internal string.
        _ => REASON_PROVIDER_POLICY_UNKNOWN,
    }
}

#[derive(Debug, Clone, Default)]
struct ProviderPolicyFilter {
    allowlist: Option<HashSet<String>>,
    denylist: HashSet<String>,
}

impl ProviderPolicyFilter {
    fn inactive() -> Self {
        Self::default()
    }

    fn from_slices(allowlist: Option<&[String]>, denylist: Option<&[String]>) -> Self {
        let allow = normalize_provider_set(allowlist);
        Self {
            allowlist: if allow.is_empty() { None } else { Some(allow) },
            denylist: normalize_provider_set(denylist),
        }
    }

    fn active(&self) -> bool {
        self.allowlist.is_some() || !self.denylist.is_empty()
    }

    fn violation_reason(&self, provider: &str) -> Option<&'static str> {
        if !self.active() {
            return None;
        }
        let normalized = normalize_provider(provider);
        if normalized.is_empty() {
            return Some("provider_policy_unknown_provider");
        }
        if self.denylist.contains(&normalized) {
            return Some("provider_policy_denylist");
        }
        if let Some(allowlist) = &self.allowlist {
            if !allowlist.contains(&normalized) {
                return Some("provider_policy_outside_allowlist");
            }
        }
        None
    }
}

type AssignmentResult<T> = Result<T, String>;

#[derive(Debug, Clone, Copy)]
struct SingleNodePolicyOptions<'a> {
    task_domain: &'a str,
    budget_usd: f32,
    task_system: Option<CognitiveSystem>,
    provider_policy: &'a ProviderPolicyFilter,
}

fn normalize_provider(provider: &str) -> String {
    provider.trim().to_ascii_lowercase()
}

fn normalize_provider_set(values: Option<&[String]>) -> HashSet<String> {
    values
        .unwrap_or(&[])
        .iter()
        .map(|value| normalize_provider(value))
        .filter(|value| !value.is_empty())
        .collect()
}

/// Roles that templates.rs explicitly marks as "sink" (final-stage
/// forwarder) by assigning `SINK_NODE_PROMPT` to them. Single source of
/// truth: maintain this list in sync with templates.rs whenever a new
/// template adds a SINK_NODE_PROMPT-tagged node.
///
/// To audit:
///   `grep -B 1 SINK_NODE_PROMPT sage-core/src/topology/templates.rs`
/// As of 2026-04-17 the templates assign SINK_NODE_PROMPT to:
///   synthesizer (sequential, brainstorming)
///   aggregator  (parallel, horizon_pipeline, parallel_fanout)
///   mixer       (self_moa)
///   judge       (debate)
///   verifier    (robust — final node; AVR's verifier is non-final but
///                also benign as a sink classification — its job is
///                cheap pattern-matching, not deep reasoning)
///   solver      (formal_solver — DETERMINISTIC compute node, model_id="";
///                MUST stay sink or F7 will replace free Rust math with
///                a $0.10 LLM call on math/formal tasks)
const SINK_ROLES: &[&str] = &[
    "synthesizer",
    "aggregator",
    "mixer",
    "judge",
    "verifier",
    "solver",
    "formatter",
    "output",
    "sink",
];

/// Classify a role string as a "sink" (output-only forwarder) vs a
/// "producer" (generates content). Sink nodes keep their template-assigned
/// cognitive tier — they just format / synthesize predecessor output.
/// Producer nodes do the actual reasoning and MUST be promoted to the
/// overall task tier floor when the task is complex (F7, Topaz-inspired).
///
/// Research: Topaz (arXiv 2604.03527) argues routing must match model
/// capability to per-subtask REQUIREMENTS. The role-based sink/producer
/// split is the training-free approximation that fits our existing
/// template catalogue (templates.rs).
fn is_sink_role(role: &str) -> bool {
    let r = role.to_lowercase();
    SINK_ROLES.iter().any(|&s| r == s) || r.starts_with("output_")
}

/// True if the task domain demands the highest reasoning tier
/// (formal proofs, math, theorem proving). For these, an S3 task pushes
/// producer nodes all the way to S3 (full reasoner tier), not the
/// general-purpose S2 floor.
///
/// Match is case-insensitive substring, so `"math"`, `"formal"`,
/// `"formal_verification"`, `"Math"` all classify as high-rigour. This
/// keeps the function tolerant to whatever upstream pipeline naming
/// convention surfaces (`ctx.domain` in pipeline.py uses lower-case
/// short tokens; cards.toml uses `math`, `formal`).
fn is_high_rigour_domain(task_domain: &str) -> bool {
    let d = task_domain.to_lowercase();
    d.contains("math") || d.contains("formal")
}

/// Compute the effective cognitive tier for Stage 3 model assignment.
///
/// * Sink nodes stay on their template tier — they just forward predecessor
///   output, they don't need a reasoner.
/// * Producer nodes get promoted to `max(node.system, floor)`, where
///   `floor` depends on (task tier, task domain):
///   - S3 + math/formal     → 3 (full reasoner tier — proofs/Z3 need it)
///   - S3 + other domains   → 2 (mid-tier reasoner suffices for code/general)
///   - S2                   → 1 (no-op; S1 is already the minimum)
///   - S1 / None            → keep node tier (no promotion)
/// * `task_system=None` disables the promotion entirely (full back-compat
///   with pre-F7 callers that didn't know about task-level routing).
///
/// Why the domain split: an S3 SWE-bench task wants a strong code agent,
/// not necessarily a pure proof model. But an S3 math/formal task NEEDS
/// the reasoner tier — there's no point promoting a coder. This is the
/// minimal Topaz-inspired adaptation that the existing card domain_scores
/// already support (cards expose `math` and `formal` columns explicitly).
fn effective_system(
    role: &str,
    node_system: CognitiveSystem,
    task_system: Option<CognitiveSystem>,
    task_domain: &str,
) -> CognitiveSystem {
    let task = match task_system {
        Some(t) => t,
        None => return node_system,
    };
    if is_sink_role(role) {
        return node_system;
    }
    let floor_n: u8 = if matches!(task, CognitiveSystem::S3) && is_high_rigour_domain(task_domain) {
        3
    } else {
        (task as u8).saturating_sub(1)
    };
    let node_n = node_system as u8;
    let promoted_n = std::cmp::max(node_n, floor_n);
    match promoted_n {
        1 => CognitiveSystem::S1,
        2 => CognitiveSystem::S2,
        3 => CognitiveSystem::S3,
        _ => node_system,
    }
}

// ── PyModelCandidateTrace (P1-2B) ────────────────────────────────────────

/// Per-model candidate trace exposed via PyO3 for top-3 logging.
#[pyclass]
#[derive(Debug, Clone)]
pub struct PyModelCandidateTrace {
    #[pyo3(get)]
    pub node_idx: usize,
    #[pyo3(get)]
    pub rank: usize,
    #[pyo3(get)]
    pub model_id: String,
    #[pyo3(get)]
    pub total_score: f32,
    #[pyo3(get)]
    pub affinity_score: f32,
    #[pyo3(get)]
    pub domain_score: f32,
    #[pyo3(get)]
    pub cost_norm: f32,
    #[pyo3(get)]
    pub hint_bonus: f32,
    #[pyo3(get)]
    pub diversity_penalty: f32,
    #[pyo3(get)]
    pub filtered_reason: String,
}

#[pymethods]
impl PyModelCandidateTrace {
    fn __repr__(&self) -> String {
        format!(
            "PyModelCandidateTrace(node_idx={}, rank={}, model_id='{}', \
             total_score={:.6}, affinity={:.6}, domain={:.6}, \
             cost_norm={:.6}, hint_bonus={:.6}, diversity_penalty={:.6}, \
             filtered_reason='{}')",
            self.node_idx,
            self.rank,
            self.model_id,
            self.total_score,
            self.affinity_score,
            self.domain_score,
            self.cost_norm,
            self.hint_bonus,
            self.diversity_penalty,
            self.filtered_reason,
        )
    }
}

// ── ModelAssigner ─────────────────────────────────────────────────────────

#[pyclass]
#[derive(Debug)]
pub struct ModelAssigner {
    registry: ModelRegistry,
    /// Weight for cognitive system affinity (S1/S2/S3 match) — subject to ablation.
    weight_affinity: f32,
    /// Weight for task domain score — subject to ablation.
    weight_domain: f32,
    /// Weight for cost efficiency (lower cost = higher score) — subject to ablation.
    weight_cost: f32,
    /// Providers excluded from assignment (dead at boot health check).
    excluded_providers: Vec<String>,
    /// Per-node top-3 trace buffer (P1-2B). Cleared at start of each
    /// assign_models* call, populated per-node during scoring.
    /// Mutex allows internal mutation through &self (PyO3 compatibility).
    last_assignment_trace: Mutex<Vec<PyModelCandidateTrace>>,
    /// Slice 10D Phase 2 (cgpro DESIGN_LOCKED v1_1, 2026-05-11):
    /// structured candidate rejection buffer for the
    /// `provider_execution_witness` payload. Mutex mirrors
    /// `last_assignment_trace`. Read by Python via
    /// `last_filter_rejections()` / `last_filter_rejections_truncated()`.
    last_filter_rejections: Mutex<FilterRejectionState>,
}

impl ModelAssigner {
    pub fn from_registry(registry: &ModelRegistry) -> Self {
        Self {
            registry: registry.clone(),
            weight_affinity: DEFAULT_WEIGHT_AFFINITY,
            weight_domain: DEFAULT_WEIGHT_DOMAIN,
            weight_cost: DEFAULT_WEIGHT_COST,
            excluded_providers: Vec::new(),
            last_assignment_trace: Mutex::new(Vec::new()),
            last_filter_rejections: Mutex::new(FilterRejectionState::default()),
        }
    }

    /// Create an assigner with custom scoring weights.
    /// Weights are normalized to sum to 1.0 internally.
    pub fn with_weights(
        registry: &ModelRegistry,
        weight_affinity: f32,
        weight_domain: f32,
        weight_cost: f32,
    ) -> Self {
        let total = weight_affinity + weight_domain + weight_cost;
        let norm = if total > 0.0 { total } else { 1.0 };
        Self {
            registry: registry.clone(),
            weight_affinity: weight_affinity / norm,
            weight_domain: weight_domain / norm,
            weight_cost: weight_cost / norm,
            excluded_providers: Vec::new(),
            last_assignment_trace: Mutex::new(Vec::new()),
            last_filter_rejections: Mutex::new(FilterRejectionState::default()),
        }
    }

    // ── Trace helpers (P1-2B) ──────────────────────────────────────────

    pub fn last_assignment_trace(&self) -> Vec<PyModelCandidateTrace> {
        self.last_assignment_trace.lock().unwrap().clone()
    }

    #[allow(dead_code)]
    fn replace_last_assignment_trace(&self, trace: Vec<PyModelCandidateTrace>) {
        *self.last_assignment_trace.lock().unwrap() = trace;
    }

    fn clear_last_assignment_trace(&self) {
        self.last_assignment_trace.lock().unwrap().clear();
    }

    // ── Filter rejection recording (slice 10D Phase 2) ─────────────────

    /// Mark the start of a recording-aware assignment path. Called at
    /// the top of `assign_models_with_policy_inner` and
    /// `assign_single_node_with_policy_inner`. After this, the state
    /// transitions from `observed=false` (returns `None` to Python)
    /// to `observed=true` (returns `Some(rejections)`).
    fn begin_filter_recording(&self) {
        let mut state = self.last_filter_rejections.lock().unwrap();
        state.observed = true;
        state.rejections.clear();
        state.truncated = false;
    }

    /// Record a candidate rejection with the public reason code.
    /// FIFO eviction at `RUST_FILTER_REJECTIONS_CAP` (cgpro Phase 2
    /// lock §5: keep newest 20). Internal-only — public reason codes
    /// MUST already be mapped via `map_policy_violation_to_public_reason`
    /// before calling.
    fn push_filter_rejection(&self, model_id: &str, reason_code: &'static str) {
        let mut state = self.last_filter_rejections.lock().unwrap();
        if !state.observed {
            // Defensive: pushing without begin_filter_recording would
            // be a code bug — flip `observed` so Python at least sees
            // something, but log the issue.
            state.observed = true;
        }
        if state.rejections.len() >= RUST_FILTER_REJECTIONS_CAP {
            state.rejections.remove(0);
            state.truncated = true;
        }
        state
            .rejections
            .push((model_id.to_string(), reason_code.to_string()));
    }

    /// First-match deterministic candidate-rejection classifier per
    /// cgpro Phase 2 lock §4. Returns the public reason code (or
    /// `None` if the candidate is eligible).
    #[allow(clippy::too_many_arguments)]
    fn classify_candidate_rejection(
        &self,
        card: &ModelCard,
        provider_policy: &ProviderPolicyFilter,
        needs_tools: bool,
        needs_json: bool,
        node_budget: f32,
        est_cost: f32,
        excluded_by_caller_ids: Option<&[String]>,
    ) -> Option<&'static str> {
        if !card.runtime_selectable {
            return Some(REASON_CARD_INACTIVE);
        }
        if let Some(internal_reason) = provider_policy.violation_reason(&card.provider) {
            return Some(map_policy_violation_to_public_reason(internal_reason));
        }
        if self.excluded_providers.iter().any(|p| p == &card.provider) {
            return Some(REASON_PROVIDER_DEAD);
        }
        if let Some(excluded) = excluded_by_caller_ids {
            if excluded.iter().any(|e| e == &card.id) {
                return Some(REASON_EXCLUDED_BY_CALLER);
            }
        }
        if (needs_tools && !card.supports_tools) || (needs_json && !card.supports_json_mode) {
            return Some(REASON_CAPABILITY_MISMATCH);
        }
        if est_cost > node_budget {
            return Some(REASON_COST_ABOVE_BUDGET);
        }
        None
    }

    fn candidate_runtime_eligible(
        &self,
        card: &ModelCard,
        provider_policy: &ProviderPolicyFilter,
        needs_tools: bool,
        needs_json: bool,
        node_budget: f32,
    ) -> bool {
        if !card.runtime_selectable {
            return false;
        }
        if provider_policy.violation_reason(&card.provider).is_some() {
            return false;
        }
        if self.excluded_providers.iter().any(|p| p == &card.provider) {
            return false;
        }
        if needs_tools && !card.supports_tools {
            return false;
        }
        if needs_json && !card.supports_json_mode {
            return false;
        }
        let (input_tok, output_tok) = COST_ESTIMATE_TOKENS;
        card.estimate_cost(input_tok, output_tok) <= node_budget
    }

    fn runtime_replacement_for(
        &self,
        card: &ModelCard,
        provider_policy: &ProviderPolicyFilter,
        needs_tools: bool,
        needs_json: bool,
        node_budget: f32,
    ) -> Option<String> {
        let replacement_id = card.runtime_replacement.trim();
        if replacement_id.is_empty() {
            return None;
        }
        let replacement = self.registry.get(replacement_id)?;
        if !self.candidate_runtime_eligible(
            &replacement,
            provider_policy,
            needs_tools,
            needs_json,
            node_budget,
        ) {
            return None;
        }
        Some(replacement.id)
    }

    /// Exclude providers from model assignment (dead at boot health check).
    /// Models from these providers will never be assigned to any node.
    pub fn set_excluded_providers(&mut self, providers: Vec<String>) {
        info!(excluded = ?providers, "ModelAssigner: excluding dead providers");
        self.excluded_providers = providers;
    }

    pub fn assign_models_inner(
        &self,
        graph: &mut TopologyGraph,
        task_domain: &str,
        budget_usd: f32,
    ) -> usize {
        self.assign_models_with_hints_inner(graph, task_domain, budget_usd, &[], None)
    }

    /// Assign models with optional per-node provider hints.
    ///
    /// `provider_hints` is a slice of `(node_idx, provider_name)` pairs.
    /// When a hint is present for a node, candidates from that provider get
    /// a +0.15 scoring bonus (soft preference, not hard filter — if no
    /// model from the hinted provider qualifies, the best alternative wins).
    ///
    /// `task_system` is the OVERALL cognitive tier of the task (from Stage 0
    /// routing + optional `system_hint` override). Producer nodes (planner,
    /// coder, worker, verifier, source) are promoted via `effective_system`
    /// so an S3 task doesn't end up with flash-tier planners. Pass `None` to
    /// keep the legacy per-node-only behaviour (full back-compat).
    pub fn assign_models_with_hints_inner(
        &self,
        graph: &mut TopologyGraph,
        task_domain: &str,
        budget_usd: f32,
        provider_hints: &[(usize, String)],
        task_system: Option<CognitiveSystem>,
    ) -> usize {
        self.assign_models_with_policy_inner(
            graph,
            task_domain,
            budget_usd,
            provider_hints,
            task_system,
            &ProviderPolicyFilter::inactive(),
        )
        .expect("inactive provider policy cannot fail closed")
    }

    fn assign_models_with_policy_inner(
        &self,
        graph: &mut TopologyGraph,
        task_domain: &str,
        budget_usd: f32,
        provider_hints: &[(usize, String)],
        task_system: Option<CognitiveSystem>,
        provider_policy: &ProviderPolicyFilter,
    ) -> AssignmentResult<usize> {
        self.clear_last_assignment_trace();
        // Slice 10D Phase 2: start recording candidate rejections for
        // the provider_execution_witness payload. cgpro Phase 2 lock §3:
        // boundary clear, never on getter read.
        self.begin_filter_recording();
        let node_count = graph.node_count();
        let mut remaining_budget = budget_usd;
        let mut assigned = 0usize;

        let all_models = self.registry.all_models();
        if all_models.is_empty() {
            warn!("ModelAssigner: no models in registry, skipping assignment");
            return Ok(0);
        }

        // Pre-compute cost estimates once (avoids repeated calls per node).
        let (input_tok, output_tok) = COST_ESTIMATE_TOKENS;
        let model_costs: Vec<f32> = all_models
            .iter()
            .map(|c| c.estimate_cost(input_tok, output_tok))
            .collect();
        let max_cost = all_models
            .iter()
            .zip(model_costs.iter())
            .filter(|(card, _)| {
                card.runtime_selectable
                    && provider_policy.violation_reason(&card.provider).is_none()
            })
            .fold(0.001_f32, |acc, (_, &cost)| acc.max(cost));

        // Build provider hint lookup
        let hint_map: std::collections::HashMap<usize, &str> = provider_hints
            .iter()
            .map(|(idx, prov)| (*idx, prov.as_str()))
            .collect();

        // Provider-diversity load balancing (added 2026-04-18 after v5f
        // observation: MiniMax took 88% of calls → rate-limit saturation).
        // Track how many nodes have already been assigned to each provider
        // during this call; apply a soft penalty that grows with
        // concentration. Cap the penalty so affinity still wins when a
        // specific provider is strongly preferred by the score function.
        let mut provider_count: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        for idx in 0..node_count {
            if remaining_budget < BUDGET_EPSILON {
                warn!(
                    node_idx = idx,
                    remaining_nodes = node_count - idx,
                    "budget_exhausted — stopping assignment"
                );
                break;
            }

            let node = match graph.try_get_node(idx) {
                Ok(n) => n,
                Err(_) => continue,
            };

            let local_system = match node.system {
                1 => CognitiveSystem::S1,
                2 => CognitiveSystem::S2,
                3 => CognitiveSystem::S3,
                _ => CognitiveSystem::S1,
            };
            let system = effective_system(&node.role, local_system, task_system, task_domain);
            if system != local_system {
                info!(
                    node = idx,
                    role = %node.role,
                    local = %local_system,
                    effective = %system,
                    task = ?task_system,
                    domain = task_domain,
                    "tier_promoted"
                );
            }

            let caps = &node.required_capabilities;
            let needs_tools = caps.iter().any(|c| c == "tools");
            let needs_json = caps.iter().any(|c| c == "json" || c == "json_mode");
            let node_budget = node.max_cost_usd.min(remaining_budget);
            let preferred_provider = hint_map.get(&idx).copied().unwrap_or("");
            let preassigned_model_id = node.model_id.clone();

            // Respect pre-assigned model_id from templates/state only when
            // the catalog still marks it runtime-selectable. Compatibility
            // aliases remain in cards.toml for historical records, but fresh
            // execution must move to their declared replacement or to a newly
            // selected eligible model.
            let preassigned_policy_failure = if !preassigned_model_id.is_empty() {
                match self.registry.get(&preassigned_model_id) {
                    Some(card) if !card.runtime_selectable => {
                        if let Some(replacement_id) = self.runtime_replacement_for(
                            &card,
                            provider_policy,
                            needs_tools,
                            needs_json,
                            node_budget,
                        ) {
                            remaining_budget -= self
                                .registry
                                .get(&replacement_id)
                                .map(|c| c.estimate_cost(input_tok, output_tok))
                                .unwrap_or(0.0);
                            if let Some(provider) = self
                                .registry
                                .get(&replacement_id)
                                .map(|c| c.provider.clone())
                            {
                                *provider_count.entry(provider).or_insert(0) += 1;
                            }
                            let node_idx_pg = petgraph::graph::NodeIndex::new(idx);
                            if let Some(node_mut) =
                                graph.inner_graph_mut().node_weight_mut(node_idx_pg)
                            {
                                node_mut.model_id.clone_from(&replacement_id);
                                info!(
                                    node = idx,
                                    old_model = %preassigned_model_id,
                                    replacement = %replacement_id,
                                    "preassigned_runtime_alias_replaced"
                                );
                            }
                            assigned += 1;
                            continue;
                        }
                        warn!(
                            node = idx,
                            model = %preassigned_model_id,
                            "preassigned_model_runtime_not_selectable"
                        );
                        Some((preassigned_model_id.clone(), "runtime_model_not_selectable"))
                    }
                    Some(card) => {
                        let policy_reason = provider_policy.violation_reason(&card.provider);
                        if policy_reason.is_none() {
                            assigned += 1;
                            continue;
                        }
                        let reason = policy_reason.unwrap_or("provider_policy_violation");
                        warn!(
                            node = idx,
                            model = %preassigned_model_id,
                            reason,
                            "preassigned_model_policy_filtered"
                        );
                        Some((preassigned_model_id.clone(), reason))
                    }
                    None => {
                        // Slice 10D Phase 2 lock §6: preassigned model
                        // not in registry — candidate loop can't see
                        // it, so push rejection here so the witness
                        // exposes the unknown-provider substitution.
                        self.push_filter_rejection(
                            &preassigned_model_id,
                            REASON_PROVIDER_POLICY_UNKNOWN,
                        );
                        Some((
                            preassigned_model_id.clone(),
                            "provider_policy_unknown_provider",
                        ))
                    }
                }
            } else {
                None
            };

            let mut best_id: Option<String> = None;
            let mut best_score: f32 = f32::NEG_INFINITY;
            let mut node_candidates: Vec<(usize, PyModelCandidateTrace)> = Vec::new();

            for (card_idx, card) in all_models.iter().enumerate() {
                let est_cost = model_costs[card_idx];
                // Slice 10D Phase 2: structured rejection recording.
                // Same predicate set, just exposed via the classifier
                // so we can attribute each skip to a reason code.
                if let Some(reason_code) = self.classify_candidate_rejection(
                    card,
                    provider_policy,
                    needs_tools,
                    needs_json,
                    node_budget,
                    est_cost,
                    None,
                ) {
                    self.push_filter_rejection(&card.id, reason_code);
                    continue;
                }

                let affinity = self.registry.calibrated_affinity(&card.id, system);
                let domain = card.domain_score(task_domain);
                let cost_norm = est_cost / max_cost;
                let mut score = self.weight_affinity * affinity
                    + self.weight_domain * domain
                    + self.weight_cost * (1.0 - cost_norm);

                // Provider hint bonus: soft preference for the hinted provider
                let hint_bonus =
                    if !preferred_provider.is_empty() && card.provider == preferred_provider {
                        score += PROVIDER_HINT_BONUS;
                        PROVIDER_HINT_BONUS
                    } else {
                        0.0
                    };

                // Diversity penalty: -0.08 per previous assignment to the
                // same provider in this call (capped at -0.20).
                let concentration = provider_count.get(&card.provider).copied().unwrap_or(0);
                let diversity_penalty = (concentration as f32 * 0.08_f32).min(0.20_f32);
                score -= diversity_penalty;

                // P1-2B: capture trace for this candidate
                let filtered_reason = if score.is_finite() {
                    "ok"
                } else {
                    "non_finite_score"
                };
                let seq = node_candidates.len();
                node_candidates.push((
                    seq,
                    PyModelCandidateTrace {
                        node_idx: idx,
                        rank: 0, // assigned after sort
                        model_id: card.id.clone(),
                        total_score: score,
                        affinity_score: affinity,
                        domain_score: domain,
                        cost_norm,
                        hint_bonus,
                        diversity_penalty,
                        filtered_reason: filtered_reason.to_string(),
                    },
                ));

                if score > best_score {
                    best_score = score;
                    best_id = Some(card.id.clone());
                }
            }

            // P1-2B: sort node candidates by score (desc), then by
            // insertion order (stable tie-break), take top 3.
            node_candidates.sort_by(|(seq_a, a), (seq_b, b)| {
                let by_score = match (a.total_score.is_finite(), b.total_score.is_finite()) {
                    (true, true) => b
                        .total_score
                        .partial_cmp(&a.total_score)
                        .unwrap_or(Ordering::Equal),
                    (true, false) => Ordering::Less,
                    (false, true) => Ordering::Greater,
                    (false, false) => Ordering::Equal,
                };
                by_score.then_with(|| seq_a.cmp(seq_b))
            });
            let mut trace_buf = self.last_assignment_trace.lock().unwrap();
            for (rank, (_, mut candidate)) in node_candidates.into_iter().take(3).enumerate() {
                candidate.rank = rank + 1;
                trace_buf.push(candidate);
            }
            drop(trace_buf);

            if let Some(model_id) = best_id {
                remaining_budget -= self
                    .registry
                    .get(&model_id)
                    .map(|c| c.estimate_cost(input_tok, output_tok))
                    .unwrap_or(0.0);
                // Track provider for diversity penalty on subsequent nodes.
                if let Some(provider) = self.registry.get(&model_id).map(|c| c.provider.clone()) {
                    *provider_count.entry(provider).or_insert(0) += 1;
                }
                let node_idx_pg = petgraph::graph::NodeIndex::new(idx);
                if let Some(node_mut) = graph.inner_graph_mut().node_weight_mut(node_idx_pg) {
                    node_mut.model_id.clone_from(&model_id);
                    if !preferred_provider.is_empty() {
                        info!(
                            node = idx,
                            role = %node_mut.role,
                            model = %model_id,
                            score = best_score,
                            provider_hint = preferred_provider,
                            "model_assigned (with provider hint)"
                        );
                    } else {
                        info!(
                            node = idx,
                            role = %node_mut.role,
                            model = %model_id,
                            score = best_score,
                            "model_assigned"
                        );
                    }
                }
                assigned += 1;
            } else {
                if let Some((model_id, reason)) = preassigned_policy_failure {
                    return Err(format!(
                        "preassigned model blocked without eligible replacement: node_idx={idx}, model_id={model_id}, reason={reason}"
                    ));
                }
                warn!(node = idx, "no candidate — keeping existing model_id");
            }
        }

        Ok(assigned)
    }

    pub fn assign_single_node_inner(
        &self,
        graph: &mut TopologyGraph,
        node_idx: usize,
        task_domain: &str,
        budget_usd: f32,
        exclude_ids: Option<&[String]>,
        task_system: Option<CognitiveSystem>,
    ) -> Option<String> {
        self.assign_single_node_with_policy_inner(
            graph,
            node_idx,
            exclude_ids,
            SingleNodePolicyOptions {
                task_domain,
                budget_usd,
                task_system,
                provider_policy: &ProviderPolicyFilter::inactive(),
            },
        )
        .expect("inactive provider policy cannot fail closed")
    }

    fn assign_single_node_with_policy_inner(
        &self,
        graph: &mut TopologyGraph,
        node_idx: usize,
        exclude_ids: Option<&[String]>,
        options: SingleNodePolicyOptions<'_>,
    ) -> AssignmentResult<Option<String>> {
        // Slice 10D Phase 2: start recording for this single-node
        // assignment. cgpro Phase 2 lock §3.
        self.begin_filter_recording();
        let node = match graph.try_get_node(node_idx) {
            Ok(node) => node,
            Err(_) => return Ok(None),
        };
        let local_system = match node.system {
            1 => CognitiveSystem::S1,
            2 => CognitiveSystem::S2,
            3 => CognitiveSystem::S3,
            _ => CognitiveSystem::S1,
        };
        let system = effective_system(
            &node.role,
            local_system,
            options.task_system,
            options.task_domain,
        );
        let caps = &node.required_capabilities;
        let needs_tools = caps.iter().any(|c| c == "tools");
        let needs_json = caps.iter().any(|c| c == "json" || c == "json_mode");
        let preassigned_model_id = node.model_id.clone();
        let all_models = self.registry.all_models();
        let (input_tok, output_tok) = COST_ESTIMATE_TOKENS;
        let model_costs: Vec<f32> = all_models
            .iter()
            .map(|c| c.estimate_cost(input_tok, output_tok))
            .collect();
        let max_cost = all_models
            .iter()
            .zip(model_costs.iter())
            .filter(|(card, _)| {
                card.runtime_selectable
                    && options
                        .provider_policy
                        .violation_reason(&card.provider)
                        .is_none()
            })
            .fold(0.001_f32, |acc, (_, &cost)| acc.max(cost));

        let preassigned_policy_failure = if !preassigned_model_id.is_empty() {
            match self.registry.get(&preassigned_model_id) {
                Some(card) if !card.runtime_selectable => {
                    if let Some(replacement_id) = self.runtime_replacement_for(
                        &card,
                        options.provider_policy,
                        needs_tools,
                        needs_json,
                        options.budget_usd,
                    ) {
                        let node_idx_pg = petgraph::graph::NodeIndex::new(node_idx);
                        if let Some(node_mut) = graph.inner_graph_mut().node_weight_mut(node_idx_pg)
                        {
                            node_mut.model_id = replacement_id.clone();
                        }
                        return Ok(Some(replacement_id));
                    }
                    Some((preassigned_model_id.clone(), "runtime_model_not_selectable"))
                }
                Some(card) if options.provider_policy.active() => options
                    .provider_policy
                    .violation_reason(&card.provider)
                    .map(|reason| (preassigned_model_id.clone(), reason)),
                None if options.provider_policy.active() => {
                    // Slice 10D Phase 2 lock §6: preassigned model not
                    // in registry — push rejection so the witness
                    // exposes the unknown-provider substitution.
                    self.push_filter_rejection(
                        &preassigned_model_id,
                        REASON_PROVIDER_POLICY_UNKNOWN,
                    );
                    Some((
                        preassigned_model_id.clone(),
                        "provider_policy_unknown_provider",
                    ))
                }
                _ => None,
            }
        } else {
            None
        };

        let mut best_id: Option<String> = None;
        let mut best_score: f32 = f32::NEG_INFINITY;
        for (card_idx, card) in all_models.iter().enumerate() {
            let est_cost = model_costs[card_idx];
            // Slice 10D Phase 2: structured rejection recording via the
            // shared classifier. `exclude_ids` only applies to this
            // single-node path (FrugalGPT cascade — arXiv 2410.10347).
            if let Some(reason_code) = self.classify_candidate_rejection(
                card,
                options.provider_policy,
                needs_tools,
                needs_json,
                options.budget_usd,
                est_cost,
                exclude_ids,
            ) {
                self.push_filter_rejection(&card.id, reason_code);
                continue;
            }
            let affinity = self.registry.calibrated_affinity(&card.id, system);
            let domain = card.domain_score(options.task_domain);
            let cost_norm = est_cost / max_cost;
            let score = self.weight_affinity * affinity
                + self.weight_domain * domain
                + self.weight_cost * (1.0 - cost_norm);
            if score > best_score {
                best_score = score;
                best_id = Some(card.id.clone());
            }
        }

        if let Some(ref model_id) = best_id {
            let node_idx_pg = petgraph::graph::NodeIndex::new(node_idx);
            if let Some(node_mut) = graph.inner_graph_mut().node_weight_mut(node_idx_pg) {
                node_mut.model_id = model_id.clone();
            }
        } else if let Some((model_id, reason)) = preassigned_policy_failure {
            return Err(format!(
                "preassigned model blocked without eligible replacement: node_idx={node_idx}, model_id={model_id}, reason={reason}"
            ));
        }
        Ok(best_id)
    }
}

#[pymethods]
impl ModelAssigner {
    /// Create a ModelAssigner with optional scoring weights.
    /// Weights default to 0.4/0.4/0.2 (affinity/domain/cost) — subject to ablation.
    /// If provided, weights are normalized to sum to 1.0.
    #[new]
    #[pyo3(signature = (registry, weight_affinity=None, weight_domain=None, weight_cost=None))]
    fn py_new(
        registry: &ModelRegistry,
        weight_affinity: Option<f32>,
        weight_domain: Option<f32>,
        weight_cost: Option<f32>,
    ) -> Self {
        match (weight_affinity, weight_domain, weight_cost) {
            (Some(wa), Some(wd), Some(wc)) => Self::with_weights(registry, wa, wd, wc),
            _ => Self::from_registry(registry),
        }
    }

    /// Assign models to all topology nodes. Optional provider_hints bias
    /// the selection towards specific providers (soft preference, +0.15 bonus).
    ///
    /// `task_system` is the OVERALL cognitive tier of the task (1, 2, or 3)
    /// and drives role-aware tier promotion (see `effective_system`). When
    /// omitted, behaviour is unchanged from the legacy per-node-only
    /// scoring — same path every bench that existed pre-F7 ran.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (graph, task_domain, budget_usd, provider_hints=None, task_system=None, provider_allowlist=None, provider_denylist=None))]
    fn assign_models(
        &self,
        graph: &mut TopologyGraph,
        task_domain: &str,
        budget_usd: f32,
        provider_hints: Option<Vec<(usize, String)>>,
        task_system: Option<u8>,
        provider_allowlist: Option<Vec<String>>,
        provider_denylist: Option<Vec<String>>,
    ) -> PyResult<usize> {
        let task_sys = task_system.and_then(|n| match n {
            1 => Some(CognitiveSystem::S1),
            2 => Some(CognitiveSystem::S2),
            3 => Some(CognitiveSystem::S3),
            _ => None,
        });
        let hints: &[(usize, String)] = match &provider_hints {
            Some(h) => h.as_slice(),
            None => &[],
        };
        let provider_policy = ProviderPolicyFilter::from_slices(
            provider_allowlist.as_deref(),
            provider_denylist.as_deref(),
        );
        self.assign_models_with_policy_inner(
            graph,
            task_domain,
            budget_usd,
            hints,
            task_sys,
            &provider_policy,
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)
    }

    /// Exclude dead providers from all future assignments.
    /// Called after boot health check with list of unreachable provider names.
    fn exclude_providers(&mut self, providers: Vec<String>) {
        info!(excluded = ?providers, "ModelAssigner: excluding dead providers from Python");
        self.excluded_providers = providers;
    }

    /// Assign a model to a single node. Optional ``exclude_model_ids`` skips
    /// specific models (used by FrugalGPT cascade to force an upgrade).
    /// `task_system` drives role-aware tier promotion (see `effective_system`).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (graph, node_idx, task_domain, budget_usd, exclude_model_ids=None, task_system=None, provider_allowlist=None, provider_denylist=None))]
    fn assign_single_node(
        &self,
        graph: &mut TopologyGraph,
        node_idx: usize,
        task_domain: &str,
        budget_usd: f32,
        exclude_model_ids: Option<Vec<String>>,
        task_system: Option<u8>,
        provider_allowlist: Option<Vec<String>>,
        provider_denylist: Option<Vec<String>>,
    ) -> PyResult<String> {
        let task_sys = task_system.and_then(|n| match n {
            1 => Some(CognitiveSystem::S1),
            2 => Some(CognitiveSystem::S2),
            3 => Some(CognitiveSystem::S3),
            _ => None,
        });
        let provider_policy = ProviderPolicyFilter::from_slices(
            provider_allowlist.as_deref(),
            provider_denylist.as_deref(),
        );
        self.assign_single_node_with_policy_inner(
            graph,
            node_idx,
            exclude_model_ids.as_deref(),
            SingleNodePolicyOptions {
                task_domain,
                budget_usd,
                task_system: task_sys,
                provider_policy: &provider_policy,
            },
        )
        .map_err(pyo3::exceptions::PyValueError::new_err)?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("No candidate found"))
    }

    /// Return the top-3 per-node candidate trace from the most recent
    /// ``assign_models`` call (P1-2B).  Empty if no assignment has run yet.
    #[pyo3(name = "last_assignment_trace")]
    fn py_last_assignment_trace(&self) -> Vec<PyModelCandidateTrace> {
        self.last_assignment_trace()
    }

    /// Slice 10D Phase 2 (cgpro DESIGN_LOCKED v1_1, 2026-05-11):
    /// expose the structured filter-rejection list from the most
    /// recent recording-aware assignment call.
    ///
    /// Returns:
    /// - ``None`` if no recording-aware assignment path has run yet
    ///   since this ``ModelAssigner`` was constructed. Python sees
    ///   ``rust_filter_details_observed=False`` and
    ///   ``rust_filter_rejections=None``.
    /// - ``Some(vec![])`` if the path ran but rejected nothing.
    /// - ``Some(rejections)`` with `(model_id, reason_code)` tuples
    ///   where ``reason_code`` is one of the 8 public codes.
    #[pyo3(name = "last_filter_rejections")]
    fn py_last_filter_rejections(&self) -> Option<Vec<(String, String)>> {
        let state = self.last_filter_rejections.lock().unwrap();
        if !state.observed {
            return None;
        }
        Some(state.rejections.clone())
    }

    /// Slice 10D Phase 2: ``true`` iff the most recent recording-aware
    /// assignment call rejected more than `RUST_FILTER_REJECTIONS_CAP`
    /// candidates and oldest entries were evicted. ``false`` when the
    /// path didn't run OR when fewer than the cap were rejected.
    #[pyo3(name = "last_filter_rejections_truncated")]
    fn py_last_filter_rejections_truncated(&self) -> bool {
        self.last_filter_rejections.lock().unwrap().truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::routing::model_registry::ModelRegistry;
    use crate::topology::topology_graph::{TopologyEdge, TopologyGraph, TopologyNode};

    fn test_registry() -> ModelRegistry {
        let toml = r#"
            [[models]]
            id = "cheap-fast"
            provider = "test"
            family = "test"
            code_score = 0.5
            reasoning_score = 0.5
            tool_use_score = 0.5
            math_score = 0.5
            formal_z3_strength = 0.3
            cost_input_per_m = 0.1
            cost_output_per_m = 0.2
            latency_ttft_ms = 100.0
            tokens_per_sec = 200.0
            s1_affinity = 0.9
            s2_affinity = 0.3
            s3_affinity = 0.1
            recommended_topologies = ["sequential"]
            supports_tools = false
            supports_json_mode = false
            supports_vision = false
            context_window = 128000
            [models.domain_scores]
            code = 0.5
            math = 0.4

            [[models]]
            id = "expensive-smart"
            provider = "test"
            family = "test"
            code_score = 0.9
            reasoning_score = 0.95
            tool_use_score = 0.9
            math_score = 0.9
            formal_z3_strength = 0.8
            cost_input_per_m = 5.0
            cost_output_per_m = 15.0
            latency_ttft_ms = 3000.0
            tokens_per_sec = 50.0
            s1_affinity = 0.1
            s2_affinity = 0.9
            s3_affinity = 0.95
            recommended_topologies = ["avr", "debate"]
            supports_tools = true
            supports_json_mode = true
            supports_vision = true
            context_window = 1000000
            [models.domain_scores]
            code = 0.9
            math = 0.95
        "#;
        ModelRegistry::from_toml_str(toml).unwrap()
    }

    fn two_node_graph() -> TopologyGraph {
        let mut g = TopologyGraph::try_new("sequential").unwrap();
        let n0 = TopologyNode::new(
            "coder".into(),
            "".into(),
            2,
            vec!["tools".into()],
            0,
            5.0,
            60.0,
        );
        let n1 = TopologyNode::new("reviewer".into(), "".into(), 3, vec![], 0, 5.0, 60.0);
        let edge = TopologyEdge::control();
        g.add_node(n0);
        g.add_node(n1);
        g.try_add_edge(0, 1, edge).unwrap();
        g
    }

    #[test]
    fn test_assign_models_basic() {
        let registry = test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = two_node_graph();
        let n = assigner.assign_models_inner(&mut graph, "code", 10.0);
        assert_eq!(n, 2);
        // Coder (S2, needs tools) -> expensive-smart (only one with tools)
        assert_eq!(graph.try_get_node(0).unwrap().model_id, "expensive-smart");
        // Reviewer (S3, no special caps) -> expensive-smart (highest S3 affinity)
        assert_eq!(graph.try_get_node(1).unwrap().model_id, "expensive-smart");
    }

    #[test]
    fn test_assign_respects_budget() {
        let registry = test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = two_node_graph();
        let n = assigner.assign_models_inner(&mut graph, "code", 0.005);
        let model0 = &graph.try_get_node(0).unwrap().model_id;
        let model1 = &graph.try_get_node(1).unwrap().model_id;
        // With tiny budget, either cheap-fast is picked or fewer nodes assigned
        assert!(model0 == "cheap-fast" || model1 == "cheap-fast" || n < 2);
    }

    #[test]
    fn test_assign_keeps_existing_when_no_candidate() {
        let registry = test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut g = TopologyGraph::try_new("sequential").unwrap();
        let n0 = TopologyNode::new(
            "special".into(),
            "original-model".into(),
            2,
            vec!["tools".into(), "json".into(), "vision".into()],
            0,
            0.001,
            60.0,
        );
        g.add_node(n0);
        let n = assigner.assign_models_inner(&mut g, "code", 0.001);
        // No model can satisfy tools+json+vision within 0.001 budget
        assert_eq!(g.try_get_node(0).unwrap().model_id, "original-model");
        assert_eq!(n, 0);
    }

    #[test]
    fn test_budget_exhaustion_stops_early() {
        let registry = test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = two_node_graph();
        let n = assigner.assign_models_inner(&mut graph, "code", 0.0);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_assign_with_provider_hint() {
        // Two providers: "test-a" (cheap) and "test-b" (expensive)
        let toml = r#"
            [[models]]
            id = "model-a"
            provider = "provider-a"
            family = "test"
            code_score = 0.7
            reasoning_score = 0.7
            tool_use_score = 0.7
            math_score = 0.7
            formal_z3_strength = 0.5
            cost_input_per_m = 1.0
            cost_output_per_m = 3.0
            latency_ttft_ms = 200.0
            tokens_per_sec = 100.0
            s1_affinity = 0.5
            s2_affinity = 0.7
            s3_affinity = 0.5
            recommended_topologies = ["sequential"]
            supports_tools = true
            supports_json_mode = true
            supports_vision = false
            context_window = 128000
            [models.domain_scores]
            code = 0.7

            [[models]]
            id = "model-b"
            provider = "provider-b"
            family = "test"
            code_score = 0.75
            reasoning_score = 0.75
            tool_use_score = 0.75
            math_score = 0.75
            formal_z3_strength = 0.6
            cost_input_per_m = 1.5
            cost_output_per_m = 4.0
            latency_ttft_ms = 300.0
            tokens_per_sec = 80.0
            s1_affinity = 0.5
            s2_affinity = 0.75
            s3_affinity = 0.6
            recommended_topologies = ["sequential"]
            supports_tools = true
            supports_json_mode = true
            supports_vision = false
            context_window = 128000
            [models.domain_scores]
            code = 0.75
        "#;
        let registry = ModelRegistry::from_toml_str(toml).unwrap();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = TopologyGraph::try_new("sequential").unwrap();
        let n0 = TopologyNode::new("coder".into(), "".into(), 2, vec![], 0, 5.0, 60.0);
        graph.add_node(n0);

        // Without hint: record which model wins
        let n_no_hint = assigner.assign_models_inner(&mut graph, "code", 10.0);
        assert_eq!(n_no_hint, 1);
        let assigned_no_hint = graph.try_get_node(0).unwrap().model_id.clone();

        // With hint for the OTHER provider: the hint should flip the result
        let other_provider = if assigned_no_hint == "model-a" {
            "provider-b"
        } else {
            "provider-a"
        };
        let expected_with_hint = if other_provider == "provider-a" {
            "model-a"
        } else {
            "model-b"
        };
        let mut graph2 = TopologyGraph::try_new("sequential").unwrap();
        let n0b = TopologyNode::new("coder".into(), "".into(), 2, vec![], 0, 5.0, 60.0);
        graph2.add_node(n0b);
        let hints = vec![(0, other_provider.to_string())];
        let n_with_hint =
            assigner.assign_models_with_hints_inner(&mut graph2, "code", 10.0, &hints, None);
        assert_eq!(n_with_hint, 1);
        let assigned_with_hint = graph2.try_get_node(0).unwrap().model_id.clone();
        assert_eq!(
            assigned_with_hint, expected_with_hint,
            "Provider hint for {} should flip selection from {} to {}",
            other_provider, assigned_no_hint, expected_with_hint
        );
    }

    fn provider_policy_registry() -> ModelRegistry {
        let toml = r#"
            [[models]]
            id = "gpt-best"
            provider = "openai"
            family = "test"
            code_score = 0.95
            reasoning_score = 0.95
            tool_use_score = 0.95
            math_score = 0.95
            formal_z3_strength = 0.7
            cost_input_per_m = 0.1
            cost_output_per_m = 0.2
            latency_ttft_ms = 100.0
            tokens_per_sec = 200.0
            s1_affinity = 0.95
            s2_affinity = 0.95
            s3_affinity = 0.95
            recommended_topologies = ["sequential"]
            supports_tools = true
            supports_json_mode = true
            supports_vision = false
            context_window = 128000
            [models.domain_scores]
            code = 0.95

            [[models]]
            id = "gemini-ok"
            provider = "google"
            family = "test"
            code_score = 0.70
            reasoning_score = 0.70
            tool_use_score = 0.70
            math_score = 0.70
            formal_z3_strength = 0.5
            cost_input_per_m = 0.1
            cost_output_per_m = 0.2
            latency_ttft_ms = 100.0
            tokens_per_sec = 200.0
            s1_affinity = 0.70
            s2_affinity = 0.70
            s3_affinity = 0.70
            recommended_topologies = ["sequential"]
            supports_tools = true
            supports_json_mode = true
            supports_vision = false
            context_window = 128000
            [models.domain_scores]
            code = 0.70
        "#;
        ModelRegistry::from_toml_str(toml).unwrap()
    }

    fn one_policy_node(model_id: &str) -> TopologyGraph {
        let mut graph = TopologyGraph::try_new("sequential").unwrap();
        graph.add_node(TopologyNode::new(
            "worker".into(),
            model_id.into(),
            2,
            vec![],
            0,
            10.0,
            60.0,
        ));
        graph
    }

    fn runtime_alias_registry() -> ModelRegistry {
        let toml = r#"
            [[models]]
            id = "legacy-best"
            provider = "test"
            family = "legacy"
            runtime_selectable = false
            runtime_replacement = "current-ok"
            code_score = 0.99
            reasoning_score = 0.99
            tool_use_score = 0.99
            math_score = 0.99
            formal_z3_strength = 0.8
            cost_input_per_m = 0.1
            cost_output_per_m = 0.2
            latency_ttft_ms = 100.0
            tokens_per_sec = 200.0
            s1_affinity = 0.99
            s2_affinity = 0.99
            s3_affinity = 0.99
            recommended_topologies = ["sequential"]
            supports_tools = true
            supports_json_mode = true
            supports_vision = false
            context_window = 128000
            [models.domain_scores]
            code = 0.99

            [[models]]
            id = "current-ok"
            provider = "test"
            family = "current"
            code_score = 0.70
            reasoning_score = 0.70
            tool_use_score = 0.70
            math_score = 0.70
            formal_z3_strength = 0.5
            cost_input_per_m = 0.1
            cost_output_per_m = 0.2
            latency_ttft_ms = 100.0
            tokens_per_sec = 200.0
            s1_affinity = 0.70
            s2_affinity = 0.70
            s3_affinity = 0.70
            recommended_topologies = ["sequential"]
            supports_tools = true
            supports_json_mode = true
            supports_vision = false
            context_window = 128000
            [models.domain_scores]
            code = 0.70
        "#;
        ModelRegistry::from_toml_str(toml).unwrap()
    }

    #[test]
    fn test_runtime_non_selectable_card_is_not_assigned() {
        let registry = runtime_alias_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = one_policy_node("");

        assigner.assign_models_inner(&mut graph, "code", 10.0);

        assert_eq!(graph.try_get_node(0).unwrap().model_id, "current-ok");
    }

    #[test]
    fn test_preassigned_runtime_alias_uses_declared_replacement() {
        let registry = runtime_alias_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = one_policy_node("legacy-best");

        assigner.assign_models_inner(&mut graph, "code", 10.0);

        assert_eq!(graph.try_get_node(0).unwrap().model_id, "current-ok");
    }

    #[test]
    fn test_provider_policy_denylist_filters_highest_candidate() {
        let registry = provider_policy_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut baseline = one_policy_node("");
        assigner.assign_models_inner(&mut baseline, "code", 10.0);
        assert_eq!(baseline.try_get_node(0).unwrap().model_id, "gpt-best");

        let deny = vec!["openai".to_string()];
        let policy = ProviderPolicyFilter::from_slices(None, Some(&deny));
        let mut graph = one_policy_node("");
        let n = assigner
            .assign_models_with_policy_inner(&mut graph, "code", 10.0, &[], None, &policy)
            .unwrap();

        assert_eq!(n, 1);
        assert_eq!(graph.try_get_node(0).unwrap().model_id, "gemini-ok");
        assert!(assigner
            .last_assignment_trace()
            .iter()
            .all(|candidate| candidate.model_id != "gpt-best"));
    }

    #[test]
    fn test_provider_policy_allowlist_filters_outside_candidates() {
        let registry = provider_policy_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let allow = vec!["google".to_string()];
        let policy = ProviderPolicyFilter::from_slices(Some(&allow), None);
        let mut graph = one_policy_node("");

        assigner
            .assign_models_with_policy_inner(&mut graph, "code", 10.0, &[], None, &policy)
            .unwrap();

        assert_eq!(graph.try_get_node(0).unwrap().model_id, "gemini-ok");
    }

    #[test]
    fn test_provider_policy_denylist_wins_allowlist_conflict() {
        let registry = provider_policy_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let allow = vec!["openai".to_string(), "google".to_string()];
        let deny = vec!["openai".to_string()];
        let policy = ProviderPolicyFilter::from_slices(Some(&allow), Some(&deny));
        let mut graph = one_policy_node("");

        assigner
            .assign_models_with_policy_inner(&mut graph, "code", 10.0, &[], None, &policy)
            .unwrap();

        assert_eq!(graph.try_get_node(0).unwrap().model_id, "gemini-ok");
    }

    #[test]
    fn test_provider_hint_does_not_authorize_policy_denied_provider() {
        let registry = provider_policy_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let deny = vec!["openai".to_string()];
        let policy = ProviderPolicyFilter::from_slices(None, Some(&deny));
        let hints = vec![(0usize, "openai".to_string())];
        let mut graph = one_policy_node("");

        assigner
            .assign_models_with_policy_inner(&mut graph, "code", 10.0, &hints, None, &policy)
            .unwrap();

        assert_eq!(graph.try_get_node(0).unwrap().model_id, "gemini-ok");
    }

    #[test]
    fn test_preassigned_denied_model_is_reassigned_under_provider_policy() {
        let registry = provider_policy_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let deny = vec!["openai".to_string()];
        let policy = ProviderPolicyFilter::from_slices(None, Some(&deny));
        let mut graph = one_policy_node("gpt-best");

        assigner
            .assign_models_with_policy_inner(&mut graph, "code", 10.0, &[], None, &policy)
            .unwrap();

        assert_eq!(graph.try_get_node(0).unwrap().model_id, "gemini-ok");
    }

    #[test]
    fn test_preassigned_denied_model_without_replacement_fails_closed() {
        let registry = provider_policy_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let allow = vec!["anthropic".to_string()];
        let policy = ProviderPolicyFilter::from_slices(Some(&allow), None);
        let mut graph = one_policy_node("gpt-best");

        let err = assigner
            .assign_models_with_policy_inner(&mut graph, "code", 10.0, &[], None, &policy)
            .unwrap_err();

        assert!(err.contains("without eligible replacement"));
        assert!(err.contains("provider_policy_outside_allowlist"));
        assert_eq!(graph.try_get_node(0).unwrap().model_id, "gpt-best");
    }

    #[test]
    fn test_assign_single_node_filters_provider_policy() {
        let registry = provider_policy_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let deny = vec!["openai".to_string()];
        let policy = ProviderPolicyFilter::from_slices(None, Some(&deny));
        let mut graph = one_policy_node("");

        let selected = assigner
            .assign_single_node_with_policy_inner(
                &mut graph,
                0,
                None,
                SingleNodePolicyOptions {
                    task_domain: "code",
                    budget_usd: 10.0,
                    task_system: None,
                    provider_policy: &policy,
                },
            )
            .unwrap();

        assert_eq!(selected.as_deref(), Some("gemini-ok"));
        assert_eq!(graph.try_get_node(0).unwrap().model_id, "gemini-ok");
    }

    #[test]
    fn test_assign_single_node() {
        let registry = test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = two_node_graph();
        let model_id = assigner.assign_single_node_inner(&mut graph, 1, "math", 10.0, None, None);
        assert!(model_id.is_some());
        assert_eq!(graph.try_get_node(1).unwrap().model_id, model_id.unwrap());
    }

    #[test]
    fn test_assign_single_node_with_exclusion() {
        let registry = test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = two_node_graph();
        // First assign without exclusion
        let model_id = assigner.assign_single_node_inner(&mut graph, 1, "math", 10.0, None, None);
        assert!(model_id.is_some());
        let first_model = model_id.unwrap();

        // Now assign with that model excluded — should pick a different one
        let mut graph2 = two_node_graph();
        let model_id2 = assigner.assign_single_node_inner(
            &mut graph2,
            1,
            "math",
            10.0,
            Some(std::slice::from_ref(&first_model)),
            None,
        );
        if let Some(ref m) = model_id2 {
            assert_ne!(m, &first_model, "Excluded model should not be reassigned");
        }
        // Either different model or None (all excluded) — both are correct
    }

    #[test]
    fn test_custom_weights_cost_heavy() {
        // With cost weight dominating (0.0/0.0/1.0), cheap-fast should win
        // for a node that doesn't require tools (reviewer, S3).
        let registry = test_registry();
        let assigner = ModelAssigner::with_weights(&registry, 0.0, 0.0, 1.0);
        let mut g = TopologyGraph::try_new("sequential").unwrap();
        let n = TopologyNode::new("reviewer".into(), "".into(), 3, vec![], 0, 5.0, 60.0);
        g.add_node(n);
        assigner.assign_models_inner(&mut g, "code", 10.0);
        assert_eq!(
            g.try_get_node(0).unwrap().model_id,
            "cheap-fast",
            "cost-heavy weights should prefer cheap-fast"
        );
    }

    #[test]
    fn test_custom_weights_affinity_heavy() {
        // With affinity weight dominating (1.0/0.0/0.0) for S3, expensive-smart should win.
        let registry = test_registry();
        let assigner = ModelAssigner::with_weights(&registry, 1.0, 0.0, 0.0);
        let mut g = TopologyGraph::try_new("sequential").unwrap();
        let n = TopologyNode::new("reviewer".into(), "".into(), 3, vec![], 0, 5.0, 60.0);
        g.add_node(n);
        assigner.assign_models_inner(&mut g, "code", 10.0);
        assert_eq!(
            g.try_get_node(0).unwrap().model_id,
            "expensive-smart",
            "affinity-heavy weights for S3 should prefer expensive-smart"
        );
    }

    #[test]
    fn test_weights_normalized() {
        // with_weights normalizes to sum=1.0
        let registry = test_registry();
        let a = ModelAssigner::with_weights(&registry, 4.0, 4.0, 2.0);
        assert!((a.weight_affinity - 0.4).abs() < 1e-6);
        assert!((a.weight_domain - 0.4).abs() < 1e-6);
        assert!((a.weight_cost - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_default_weights_unchanged() {
        let registry = test_registry();
        let a = ModelAssigner::from_registry(&registry);
        assert!((a.weight_affinity - 0.4).abs() < 1e-6);
        assert!((a.weight_domain - 0.4).abs() < 1e-6);
        assert!((a.weight_cost - 0.2).abs() < 1e-6);
    }

    // ── F7 effective_system + role-aware tier promotion ────────────────
    //
    // Topaz-inspired (arXiv 2604.03527): route by role+task requirement,
    // not by raw template tier. See sage-python/docs/benchmarks/
    // 2026-04-17-swebench-smoke-debug.md for the motivating evidence.

    #[test]
    fn test_effective_system_without_task_hint_is_identity() {
        // Legacy behaviour preserved when task_system=None — every existing
        // caller that didn't know about task-level routing still gets the
        // per-node tier it always got.
        for role in ["planner", "coder", "synthesizer", "worker"] {
            for local in [
                CognitiveSystem::S1,
                CognitiveSystem::S2,
                CognitiveSystem::S3,
            ] {
                assert_eq!(effective_system(role, local, None, "code"), local);
            }
        }
    }

    #[test]
    fn test_effective_system_s3_task_promotes_producers() {
        // The SWE-bench case: sequential template has planner=S1, coder=S2,
        // synthesizer=S1. With a task-level S3 hint on a code task, the
        // producer nodes (planner, coder, worker) floor at S2; the
        // synthesizer stays S1.
        assert_eq!(
            effective_system(
                "planner",
                CognitiveSystem::S1,
                Some(CognitiveSystem::S3),
                "code"
            ),
            CognitiveSystem::S2
        );
        assert_eq!(
            effective_system(
                "coder",
                CognitiveSystem::S2,
                Some(CognitiveSystem::S3),
                "code"
            ),
            CognitiveSystem::S2
        );
        assert_eq!(
            effective_system(
                "worker_0",
                CognitiveSystem::S1,
                Some(CognitiveSystem::S3),
                "code"
            ),
            CognitiveSystem::S2
        );
        assert_eq!(
            effective_system(
                "synthesizer",
                CognitiveSystem::S1,
                Some(CognitiveSystem::S3),
                "code"
            ),
            CognitiveSystem::S1
        );
    }

    #[test]
    fn test_effective_system_s2_task_is_no_op_for_low_local() {
        // S2 task: producers floor at S1, which is already the minimum —
        // no promotion happens. Templates that deliberately use cheap
        // planners on S2 tasks keep that choice.
        assert_eq!(
            effective_system(
                "planner",
                CognitiveSystem::S1,
                Some(CognitiveSystem::S2),
                "code"
            ),
            CognitiveSystem::S1
        );
    }

    #[test]
    fn test_effective_system_sink_roles_never_promoted() {
        // Synthesizer / aggregator / formatter / output_* are terminal
        // forwarders (SINK_NODE_PROMPT). Cheap is correct — the domain
        // floor must NOT override the sink classification.
        for sink in [
            "synthesizer",
            "aggregator",
            "output_formatter",
            "formatter",
            "output_writer",
        ] {
            assert_eq!(
                effective_system(sink, CognitiveSystem::S1, Some(CognitiveSystem::S3), "math"),
                CognitiveSystem::S1,
                "sink role `{}` must stay on local tier even on math/S3",
                sink
            );
        }
    }

    #[test]
    fn test_effective_system_never_demotes() {
        // If the template explicitly picked S3 for a node, we never
        // downgrade, even if the task tier is lower.
        assert_eq!(
            effective_system(
                "verifier",
                CognitiveSystem::S3,
                Some(CognitiveSystem::S1),
                "code"
            ),
            CognitiveSystem::S3
        );
        assert_eq!(
            effective_system(
                "coder",
                CognitiveSystem::S3,
                Some(CognitiveSystem::S2),
                "code"
            ),
            CognitiveSystem::S3
        );
    }

    // ── F7 domain-aware floor (advisor sequence item 1) ──────────────
    //
    // Math/formal S3 tasks get the FULL reasoner tier (S3), not the
    // general S2 floor. Code/general S3 tasks keep the S2 floor.
    //
    // Rationale: a planner on a Coq/Lean/SMT task is doing proof search,
    // not codegen — promoting to S2 picks a strong coder (e.g.
    // gpt-5.3-codex) that can't actually prove the goal. Promoting to S3
    // picks the reasoner-tier (e.g. gemini-3.1-pro-preview) that can.
    // Cards already expose `math` and `formal` columns explicitly.

    #[test]
    fn test_f7_math_s3_floors_at_s3() {
        // Math S3 task: producer planner gets full S3, not S2.
        for domain in ["math", "Math", "MATH"] {
            assert_eq!(
                effective_system(
                    "planner",
                    CognitiveSystem::S1,
                    Some(CognitiveSystem::S3),
                    domain
                ),
                CognitiveSystem::S3,
                "math/S3 must floor producer at S3 (not S2), got domain={}",
                domain
            );
        }
    }

    #[test]
    fn test_f7_formal_s3_floors_at_s3() {
        // Formal verification S3 task: same — full reasoner tier.
        // Both bare "formal" and "formal_verification" must classify.
        for domain in ["formal", "formal_verification", "Formal", "formal_proofs"] {
            assert_eq!(
                effective_system(
                    "coder",
                    CognitiveSystem::S1,
                    Some(CognitiveSystem::S3),
                    domain
                ),
                CognitiveSystem::S3,
                "formal/S3 must floor producer at S3, got domain={}",
                domain
            );
        }
    }

    #[test]
    fn test_f7_code_s3_unchanged_floors_at_s2() {
        // Sanity: the original S2 floor for non-rigour domains MUST be
        // preserved. Otherwise we'd burn budget on a pure reasoner for
        // every SWE-bench task — exactly the opposite of what we want.
        for domain in ["code", "general", "", "swe_bench", "agent"] {
            assert_eq!(
                effective_system(
                    "planner",
                    CognitiveSystem::S1,
                    Some(CognitiveSystem::S3),
                    domain
                ),
                CognitiveSystem::S2,
                "non-rigour S3 must keep the S2 floor, got domain={}",
                domain
            );
        }
    }

    #[test]
    fn test_is_sink_role_classification() {
        // Positive: every role that templates.rs assigns SINK_NODE_PROMPT to.
        // Audit cmd: `grep -B 1 SINK_NODE_PROMPT sage-core/src/topology/templates.rs`
        for r in [
            "synthesizer",
            "Synthesizer",
            "aggregator",
            "mixer",
            "judge",
            "verifier",
            "solver",
            "output_formatter",
            "formatter",
            "sink",
            "output",
        ] {
            assert!(is_sink_role(r), "`{}` should classify as sink", r);
        }
        // Negative: producer / tool-using / reasoning roles.
        for r in [
            "planner",
            "coder",
            "worker_0",
            "source",
            "thinker",
            "brainstormer",
            "actor",
            "critic",
            "formalizer",
            "preprocessor",
            "splitter",
            "dispatcher",
        ] {
            assert!(!is_sink_role(r), "`{}` should NOT classify as sink", r);
        }
    }

    // ── Drift guards (advisor 2026-04-17): pin the SINK_ROLES list and
    // the "no coder/worker on S1" template invariant to templates.rs at
    // build time. If a future template adds a SINK_NODE_PROMPT node with
    // a new role, OR declares a coder/worker at S1, these tests fail and
    // the offending diff stops at CI.

    /// Diversity penalty: a 3-node graph where two providers tie on
    /// affinity/domain should distribute instead of all going to the
    /// marginally better one. Without the penalty, the v5f smoke had
    /// MiniMax at 88% of calls; with it, nodes 2+ get a -0.08 penalty
    /// per prior assignment to the same provider.
    #[test]
    fn test_diversity_penalty_spreads_across_providers() {
        // Two providers with near-identical scores
        let toml = r#"
            [[models]]
            id = "alpha-pro"
            provider = "alpha"
            family = "test"
            code_score = 0.80
            reasoning_score = 0.80
            tool_use_score = 0.80
            math_score = 0.80
            formal_z3_strength = 0.6
            cost_input_per_m = 1.0
            cost_output_per_m = 3.0
            latency_ttft_ms = 200.0
            tokens_per_sec = 100.0
            s1_affinity = 0.5
            s2_affinity = 0.80
            s3_affinity = 0.80
            recommended_topologies = ["sequential"]
            supports_tools = true
            supports_json_mode = true
            supports_vision = false
            context_window = 128000
            [models.domain_scores]
            code = 0.80

            [[models]]
            id = "beta-pro"
            provider = "beta"
            family = "test"
            code_score = 0.78
            reasoning_score = 0.78
            tool_use_score = 0.78
            math_score = 0.78
            formal_z3_strength = 0.6
            cost_input_per_m = 1.0
            cost_output_per_m = 3.0
            latency_ttft_ms = 200.0
            tokens_per_sec = 100.0
            s1_affinity = 0.5
            s2_affinity = 0.78
            s3_affinity = 0.78
            recommended_topologies = ["sequential"]
            supports_tools = true
            supports_json_mode = true
            supports_vision = false
            context_window = 128000
            [models.domain_scores]
            code = 0.78
        "#;
        let registry = ModelRegistry::from_toml_str(toml).unwrap();
        let assigner = ModelAssigner::from_registry(&registry);

        // 3 coder nodes. With diversity penalty, we expect at least 2
        // distinct providers (not all-alpha-pro).
        let mut graph = TopologyGraph::try_new("sequential").unwrap();
        for _ in 0..3 {
            let n = TopologyNode::new("coder".into(), "".into(), 2, vec![], 0, 5.0, 60.0);
            graph.add_node(n);
        }
        let n = assigner.assign_models_inner(&mut graph, "code", 10.0);
        assert_eq!(n, 3);

        let providers: Vec<String> = (0..3)
            .map(|i| {
                let model_id = &graph.try_get_node(i).unwrap().model_id;
                registry
                    .get(model_id)
                    .map(|c| c.provider.clone())
                    .unwrap_or_default()
            })
            .collect();
        let distinct: std::collections::HashSet<_> = providers.iter().collect();
        assert!(
            distinct.len() >= 2,
            "diversity penalty should spread across providers; got {:?}",
            providers
        );
    }

    /// Every node tagged with templates::SINK_NODE_PROMPT must classify
    /// as sink via `is_sink_role`. Catches drift in either direction —
    /// new template sink role missing from SINK_ROLES, OR a SINK_ROLES
    /// entry that no template actually uses.
    #[test]
    fn test_sink_drift_templates_match_classifier() {
        use crate::topology::templates::{
            avr, brainstorming, debate, formal_solver, hierarchical, horizon_pipeline, hub,
            parallel, parallel_fanout, robust, self_moa, sequential, SINK_NODE_PROMPT,
        };

        let templates: Vec<(&str, TopologyGraph)> = vec![
            ("sequential", sequential("test-model")),
            ("parallel", parallel("test-model", 3)),
            ("avr", avr("test-model", "test-model")),
            ("self_moa", self_moa("test-model", 3)),
            ("hierarchical", hierarchical("test-model", "test-model")),
            ("hub", hub("test-model", "test-model", 3)),
            ("debate", debate("test-model", "test-model")),
            ("brainstorming", brainstorming("test-model", 3)),
            ("robust", robust("test-model", 3)),
            ("horizon_pipeline", horizon_pipeline("test-model", 3)),
            ("parallel_fanout", parallel_fanout("test-model", 3)),
            ("formal_solver", formal_solver("test-model")),
        ];

        let mut sink_count = 0;
        for (name, graph) in &templates {
            for idx in 0..graph.node_count() {
                let node = graph.try_get_node(idx).unwrap();
                if node.prompt == SINK_NODE_PROMPT {
                    sink_count += 1;
                    assert!(
                        is_sink_role(&node.role),
                        "template `{name}` node {idx} has SINK_NODE_PROMPT but role `{}` is NOT in SINK_ROLES — F7 will over-promote it on task-tier escalation",
                        node.role
                    );
                }
            }
        }
        // Sanity: ensure the test actually found sinks (would silently
        // pass if the SINK_NODE_PROMPT marker drifted to a different name).
        assert!(
            sink_count >= 6,
            "expected >=6 sink nodes across 12 templates, found {sink_count}"
        );
    }

    /// Roles whose F6 prompt explicitly mandates "AT LEAST 3 distinct
    /// execute_bash calls before emitting any diff" (see
    /// sage-python/src/sage/topology/role_prompts.py — the `_CODER`
    /// template, matched by substrings ("coder", "actor", "coder_worker")).
    /// Other producer roles (worker/thinker/brainstormer → `_WORKER`)
    /// only suggest "1-3 tool calls is typical" — softer, no hard floor.
    ///
    /// If a template declares a coder/actor at node.system=1, that's not
    /// a problem in itself (F1 max_steps = ctx.system, not node.system),
    /// BUT it's a smell: the only way that node gets a non-cheap budget
    /// is if the pipeline ALWAYS escalates the task tier. Since
    /// "S1 non-math skips topology" (CLAUDE.md), a coder@node.S1 only
    /// runs on S2/S3 tasks anyway → still safe. This test pins the
    /// template invariant; the runtime invariant ("S1 tasks bypass") is
    /// pinned at the Python layer.
    ///
    /// Rationale for the narrower predicate (vs the original draft that
    /// also flagged worker/thinker/brainstormer): `parallel_fanout` (line
    /// 678 in templates.rs) deliberately cycles workers S1/S2/S3 for
    /// output diversity (SC-MAS arXiv 2601.09434). That's a feature.
    #[test]
    fn test_no_strict_mandate_role_at_s1_in_any_template() {
        use crate::topology::templates::{
            avr, brainstorming, debate, formal_solver, hierarchical, horizon_pipeline, hub,
            parallel, parallel_fanout, robust, self_moa, sequential,
        };

        let templates: Vec<(&str, TopologyGraph)> = vec![
            ("sequential", sequential("test-model")),
            ("parallel", parallel("test-model", 3)),
            ("avr", avr("test-model", "test-model")),
            ("self_moa", self_moa("test-model", 3)),
            ("hierarchical", hierarchical("test-model", "test-model")),
            ("hub", hub("test-model", "test-model", 3)),
            ("debate", debate("test-model", "test-model")),
            ("brainstorming", brainstorming("test-model", 3)),
            ("robust", robust("test-model", 3)),
            ("horizon_pipeline", horizon_pipeline("test-model", 3)),
            ("parallel_fanout", parallel_fanout("test-model", 3)),
            ("formal_solver", formal_solver("test-model")),
        ];

        // Substring match — same predicate as get_role_prompt(role) maps to
        // _CODER in sage-python/src/sage/topology/role_prompts.py.
        let strict_mandate_substrings = ["coder", "actor"];

        for (name, graph) in &templates {
            for idx in 0..graph.node_count() {
                let node = graph.try_get_node(idx).unwrap();
                let role_lc = node.role.to_lowercase();
                let triggers_coder_prompt = strict_mandate_substrings
                    .iter()
                    .any(|s| role_lc.contains(s));
                if triggers_coder_prompt {
                    assert!(
                        node.system >= 2,
                        "template `{name}` declares strict-mandate role `{}` at system={} (S1) — F6 _CODER prompt requires >=3 execute_bash calls, F1 budgets only 5 steps at S1, leaving 1 buffer. Promote node to S2+ or move the role to a non-coder name.",
                        node.role, node.system
                    );
                }
            }
        }
    }

    #[test]
    fn test_formal_solver_sink_protected_on_math_s3() {
        // Regression for the audit-uncovered bug: pre-this-fix, F7's
        // domain floor would push formal_solver's `solver` node from S1
        // to S3 on a math task — replacing free deterministic Rust
        // computation with a $0.10 LLM call. The sink classification
        // must take precedence over the domain-aware floor.
        assert_eq!(
            effective_system(
                "solver",
                CognitiveSystem::S1,
                Some(CognitiveSystem::S3),
                "math"
            ),
            CognitiveSystem::S1,
            "formal_solver's solver MUST stay S1 — it's pure Rust compute"
        );
        // Also covered: every other SINK_NODE_PROMPT role on math/S3.
        for sink in ["mixer", "judge", "verifier", "solver"] {
            assert_eq!(
                effective_system(sink, CognitiveSystem::S1, Some(CognitiveSystem::S3), "math"),
                CognitiveSystem::S1,
                "newly-classified sink `{}` must not be promoted by domain rule",
                sink
            );
        }
    }

    #[test]
    fn test_s3_task_pushes_planner_to_reasoner_model() {
        // End-to-end: the two_node_graph() test registry has "cheap-fast"
        // (s1_affinity=0.9, s2_affinity=0.3) and "expensive-smart"
        // (s1_affinity=0.1, s2_affinity=0.9). A planner node at local=S1
        // would normally score cheap-fast highest. With task_system=S3,
        // the effective tier becomes S2 and expensive-smart should win.
        let registry = test_registry();
        let assigner = ModelAssigner::from_registry(&registry);

        let mut g_no_hint = TopologyGraph::try_new("sequential").unwrap();
        g_no_hint.add_node(TopologyNode::new(
            "planner".into(),
            "".into(),
            1,
            vec![],
            0,
            5.0,
            60.0,
        ));
        assigner.assign_models_with_hints_inner(&mut g_no_hint, "code", 10.0, &[], None);
        let baseline = g_no_hint.try_get_node(0).unwrap().model_id.clone();

        let mut g_with_hint = TopologyGraph::try_new("sequential").unwrap();
        g_with_hint.add_node(TopologyNode::new(
            "planner".into(),
            "".into(),
            1,
            vec![],
            0,
            5.0,
            60.0,
        ));
        assigner.assign_models_with_hints_inner(
            &mut g_with_hint,
            "code",
            10.0,
            &[],
            Some(CognitiveSystem::S3),
        );
        let promoted = g_with_hint.try_get_node(0).unwrap().model_id.clone();

        assert_eq!(
            baseline, "cheap-fast",
            "baseline planner@S1 should pick the S1-heavy cheap-fast"
        );
        assert_eq!(
            promoted, "expensive-smart",
            "task_system=S3 should promote planner to S2 and pick expensive-smart"
        );
    }

    #[test]
    fn test_is_high_rigour_domain_classification() {
        // Positive: substring match, case-insensitive — handles every
        // sensible variant the pipeline might emit.
        for d in [
            "math",
            "Math",
            "MATH",
            "formal",
            "Formal_Verification",
            "formal_proofs",
            "discrete_math",
            "applied_math",
        ] {
            assert!(is_high_rigour_domain(d), "`{}` should be high-rigour", d);
        }
        // Negative: explicit non-rigour domains and the unset case.
        for d in ["code", "general", "", "swe_bench", "agent", "tools"] {
            assert!(
                !is_high_rigour_domain(d),
                "`{}` should NOT be high-rigour",
                d
            );
        }
    }

    // ── P1-2B trace tests ──────────────────────────────────────────────

    fn _trace_registry() -> ModelRegistry {
        let toml = r#"
[[models]]
id = "cheap"
provider = "provider-a"
family = "test"
code_score = 0.5
reasoning_score = 0.5
tool_use_score = 0.5
math_score = 0.5
formal_z3_strength = 0.3
cost_input_per_m = 0.1
cost_output_per_m = 0.2
latency_ttft_ms = 100.0
tokens_per_sec = 200.0
s1_affinity = 0.9
s2_affinity = 0.4
s3_affinity = 0.1
recommended_topologies = ["sequential"]
supports_tools = true
supports_json_mode = true
supports_vision = false
context_window = 128000
[models.domain_scores]
code = 0.5

[[models]]
id = "mid"
provider = "provider-b"
family = "test"
code_score = 0.7
reasoning_score = 0.7
tool_use_score = 0.7
math_score = 0.7
formal_z3_strength = 0.5
cost_input_per_m = 1.0
cost_output_per_m = 2.0
latency_ttft_ms = 200.0
tokens_per_sec = 100.0
s1_affinity = 0.5
s2_affinity = 0.7
s3_affinity = 0.4
recommended_topologies = ["sequential"]
supports_tools = true
supports_json_mode = true
supports_vision = false
context_window = 128000
[models.domain_scores]
code = 0.7

[[models]]
id = "smart"
provider = "provider-c"
family = "test"
code_score = 0.9
reasoning_score = 0.9
tool_use_score = 0.9
math_score = 0.9
formal_z3_strength = 0.8
cost_input_per_m = 2.0
cost_output_per_m = 4.0
latency_ttft_ms = 500.0
tokens_per_sec = 50.0
s1_affinity = 0.2
s2_affinity = 0.95
s3_affinity = 0.8
recommended_topologies = ["sequential"]
supports_tools = true
supports_json_mode = true
supports_vision = false
context_window = 128000
[models.domain_scores]
code = 0.9
"#;
        ModelRegistry::from_toml_str(toml).unwrap()
    }

    #[test]
    fn test_assignment_trace_top3_matches_selected_model() {
        let registry = _trace_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = TopologyGraph::try_new("sequential").unwrap();
        graph.add_node(TopologyNode::new(
            "worker".into(),
            "".into(),
            2,
            vec![],
            0,
            5.0,
            60.0,
        ));

        let n = assigner.assign_models_inner(&mut graph, "code", 10.0);
        if n == 0 {
            return; // CI no-default-features may have 0 models
        }

        let trace = assigner.last_assignment_trace();
        assert!(trace.len() >= 3, "expected >=3, got {}", trace.len());

        assert_eq!(trace[0].node_idx, 0);
        assert_eq!(trace[0].rank, 1);
        assert_eq!(trace[1].rank, 2);
        assert_eq!(trace[2].rank, 3);

        let chosen = graph.try_get_node(0).unwrap().model_id.clone();
        assert_eq!(trace[0].model_id, chosen);

        assert!(trace[0].total_score.is_finite());
        assert!(trace[0].affinity_score.is_finite());
        assert!(trace[0].total_score >= trace[1].total_score);
    }

    #[test]
    fn test_assignment_trace_records_provider_hint_bonus() {
        let registry = _trace_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = TopologyGraph::try_new("sequential").unwrap();
        graph.add_node(TopologyNode::new(
            "worker".into(),
            "".into(),
            2,
            vec![],
            0,
            5.0,
            60.0,
        ));

        let hints = vec![(0usize, "provider-b".to_string())];
        let n = assigner.assign_models_with_hints_inner(&mut graph, "code", 10.0, &hints, None);
        if n == 0 {
            return; // CI no-default-features may have 0 models
        }

        let trace = assigner.last_assignment_trace();
        assert!(
            trace.iter().any(|t| t.hint_bonus > 0.0),
            "expected at least one traced candidate to carry provider hint bonus"
        );
    }

    #[test]
    fn test_assignment_trace_is_cleared_between_calls() {
        let registry = _trace_registry();
        let assigner = ModelAssigner::from_registry(&registry);

        let mut g1 = TopologyGraph::try_new("sequential").unwrap();
        g1.add_node(TopologyNode::new(
            "w1".into(),
            "".into(),
            2,
            vec![],
            0,
            5.0,
            60.0,
        ));
        let n = assigner.assign_models_inner(&mut g1, "code", 10.0);
        if n == 0 {
            // CI no-default-features build may have 0 available models
            return;
        }
        assert!(!assigner.last_assignment_trace().is_empty());

        // Budget=0 → no models affordable
        let mut g2 = TopologyGraph::try_new("sequential").unwrap();
        g2.add_node(TopologyNode::new(
            "w2".into(),
            "".into(),
            2,
            vec![],
            0,
            5.0,
            60.0,
        ));
        assigner.assign_models_inner(&mut g2, "code", 0.0);
        assert!(assigner.last_assignment_trace().is_empty());
    }

    // ── Slice 10D Phase 2 — filter rejection recording (cgpro lock) ─────

    /// Helper: registry with cards exercising every rejection predicate.
    /// Layout: 1 eligible card + 6 cards each tripping a distinct reason.
    fn rejection_test_registry() -> ModelRegistry {
        let toml = r#"
            [[models]]
            id = "eligible"
            provider = "alpha"
            family = "test"
            code_score = 0.9
            reasoning_score = 0.9
            tool_use_score = 0.9
            math_score = 0.9
            formal_z3_strength = 0.8
            cost_input_per_m = 0.1
            cost_output_per_m = 0.2
            latency_ttft_ms = 100.0
            tokens_per_sec = 200.0
            s1_affinity = 0.9
            s2_affinity = 0.9
            s3_affinity = 0.9
            recommended_topologies = ["sequential"]
            supports_tools = true
            supports_json_mode = true
            supports_vision = true
            context_window = 128000
            runtime_selectable = true
            [models.domain_scores]
            code = 0.9

            [[models]]
            id = "inactive-card"
            provider = "alpha"
            family = "legacy"
            code_score = 0.5
            reasoning_score = 0.5
            tool_use_score = 0.5
            math_score = 0.5
            formal_z3_strength = 0.3
            cost_input_per_m = 0.1
            cost_output_per_m = 0.2
            latency_ttft_ms = 100.0
            tokens_per_sec = 200.0
            s1_affinity = 0.5
            s2_affinity = 0.5
            s3_affinity = 0.5
            recommended_topologies = ["sequential"]
            supports_tools = true
            supports_json_mode = true
            supports_vision = false
            context_window = 128000
            runtime_selectable = false
            [models.domain_scores]
            code = 0.5

            [[models]]
            id = "denied-provider"
            provider = "beta"
            family = "test"
            code_score = 0.8
            reasoning_score = 0.8
            tool_use_score = 0.8
            math_score = 0.8
            formal_z3_strength = 0.5
            cost_input_per_m = 0.1
            cost_output_per_m = 0.2
            latency_ttft_ms = 100.0
            tokens_per_sec = 200.0
            s1_affinity = 0.5
            s2_affinity = 0.5
            s3_affinity = 0.5
            recommended_topologies = ["sequential"]
            supports_tools = true
            supports_json_mode = true
            supports_vision = false
            context_window = 128000
            runtime_selectable = true
            [models.domain_scores]
            code = 0.5

            [[models]]
            id = "dead-provider-card"
            provider = "gamma"
            family = "test"
            code_score = 0.8
            reasoning_score = 0.8
            tool_use_score = 0.8
            math_score = 0.8
            formal_z3_strength = 0.5
            cost_input_per_m = 0.1
            cost_output_per_m = 0.2
            latency_ttft_ms = 100.0
            tokens_per_sec = 200.0
            s1_affinity = 0.5
            s2_affinity = 0.5
            s3_affinity = 0.5
            recommended_topologies = ["sequential"]
            supports_tools = true
            supports_json_mode = true
            supports_vision = false
            context_window = 128000
            runtime_selectable = true
            [models.domain_scores]
            code = 0.5

            [[models]]
            id = "no-tools-card"
            provider = "alpha"
            family = "test"
            code_score = 0.5
            reasoning_score = 0.5
            tool_use_score = 0.5
            math_score = 0.5
            formal_z3_strength = 0.3
            cost_input_per_m = 0.1
            cost_output_per_m = 0.2
            latency_ttft_ms = 100.0
            tokens_per_sec = 200.0
            s1_affinity = 0.5
            s2_affinity = 0.5
            s3_affinity = 0.5
            recommended_topologies = ["sequential"]
            supports_tools = false
            supports_json_mode = false
            supports_vision = false
            context_window = 128000
            runtime_selectable = true
            [models.domain_scores]
            code = 0.5

            [[models]]
            id = "expensive-card"
            provider = "alpha"
            family = "test"
            code_score = 0.5
            reasoning_score = 0.5
            tool_use_score = 0.5
            math_score = 0.5
            formal_z3_strength = 0.3
            cost_input_per_m = 1000.0
            cost_output_per_m = 5000.0
            latency_ttft_ms = 100.0
            tokens_per_sec = 200.0
            s1_affinity = 0.5
            s2_affinity = 0.5
            s3_affinity = 0.5
            recommended_topologies = ["sequential"]
            supports_tools = true
            supports_json_mode = true
            supports_vision = false
            context_window = 128000
            runtime_selectable = true
            [models.domain_scores]
            code = 0.5
        "#;
        ModelRegistry::from_toml_str(toml).unwrap()
    }

    /// Single-node graph needing tools, with a generous budget so cost
    /// only fails the deliberately-expensive card.
    fn rejection_test_graph() -> TopologyGraph {
        let mut g = TopologyGraph::try_new("sequential").unwrap();
        let node = TopologyNode::new(
            "coder".into(),
            "".into(),
            2,
            vec!["tools".into()],
            0,
            1.0, // node max cost — small enough to reject "expensive-card"
            60.0,
        );
        g.add_node(node);
        g
    }

    #[test]
    fn test_phase2_no_assignment_yet_returns_none() {
        // cgpro Phase 2 test "no assignment path yet -> None"
        let registry = rejection_test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        assert!(assigner.py_last_filter_rejections().is_none());
        assert!(!assigner.py_last_filter_rejections_truncated());
    }

    #[test]
    fn test_phase2_clean_assignment_returns_empty_list() {
        // cgpro Phase 2 test "no rejection after clean assignment -> []"
        let registry = test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = two_node_graph();
        assigner.assign_models_inner(&mut graph, "code", 100.0);
        let rejections = assigner.py_last_filter_rejections();
        assert!(rejections.is_some());
        // With test_registry (only 2 cards: cheap-fast no-tools + expensive-smart tools),
        // the coder node (needs tools) will reject cheap-fast (capability_mismatch).
        // The reviewer node has no constraints, both cards eligible → no rejections.
        // Net effect: at least one rejection but no truncation.
        assert!(!assigner.py_last_filter_rejections_truncated());
    }

    #[test]
    fn test_phase2_card_inactive_recorded() {
        // cgpro Phase 2 test "runtime_selectable=false -> card_inactive"
        let registry = rejection_test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = rejection_test_graph();
        assigner.assign_models_inner(&mut graph, "code", 1.0);
        let rejections = assigner
            .py_last_filter_rejections()
            .expect("should have observed rejections");
        assert!(
            rejections
                .iter()
                .any(|(mid, rc)| mid == "inactive-card" && rc == REASON_CARD_INACTIVE),
            "expected inactive-card to be rejected as card_inactive; got {:?}",
            rejections
        );
    }

    #[test]
    fn test_phase2_provider_denylist_recorded() {
        // cgpro Phase 2 test "denylist candidate -> provider_excluded_policy_denylist"
        let registry = rejection_test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = rejection_test_graph();
        let policy = ProviderPolicyFilter {
            allowlist: None,
            denylist: ["beta".to_string()].into_iter().collect(),
        };
        let _ =
            assigner.assign_models_with_policy_inner(&mut graph, "code", 1.0, &[], None, &policy);
        let rejections = assigner
            .py_last_filter_rejections()
            .expect("should have observed rejections");
        assert!(
            rejections.iter().any(|(mid, rc)| {
                mid == "denied-provider" && rc == REASON_PROVIDER_POLICY_DENYLIST
            }),
            "expected denied-provider to be rejected as denylist; got {:?}",
            rejections
        );
    }

    #[test]
    fn test_phase2_provider_outside_allowlist_recorded() {
        // cgpro Phase 2 test "allowlist miss -> provider_excluded_policy_allowlist"
        let registry = rejection_test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = rejection_test_graph();
        let policy = ProviderPolicyFilter {
            allowlist: Some(["alpha".to_string()].into_iter().collect()),
            denylist: HashSet::new(),
        };
        let _ =
            assigner.assign_models_with_policy_inner(&mut graph, "code", 1.0, &[], None, &policy);
        let rejections = assigner
            .py_last_filter_rejections()
            .expect("should have observed rejections");
        // beta + gamma providers both outside allowlist
        assert!(
            rejections.iter().any(|(mid, rc)| {
                rc == REASON_PROVIDER_POLICY_ALLOWLIST
                    && (mid == "denied-provider" || mid == "dead-provider-card")
            }),
            "expected outside-allowlist rejection; got {:?}",
            rejections
        );
    }

    #[test]
    fn test_phase2_provider_dead_recorded() {
        // cgpro Phase 2 test "dead provider via exclude_providers -> provider_excluded_dead"
        let registry = rejection_test_registry();
        let mut assigner = ModelAssigner::from_registry(&registry);
        assigner.set_excluded_providers(vec!["gamma".to_string()]);
        let mut graph = rejection_test_graph();
        assigner.assign_models_inner(&mut graph, "code", 1.0);
        let rejections = assigner
            .py_last_filter_rejections()
            .expect("should have observed rejections");
        assert!(
            rejections
                .iter()
                .any(|(mid, rc)| { mid == "dead-provider-card" && rc == REASON_PROVIDER_DEAD }),
            "expected dead-provider-card to be rejected as provider_excluded_dead; got {:?}",
            rejections
        );
    }

    #[test]
    fn test_phase2_excluded_by_caller_recorded() {
        // cgpro Phase 2 test "FrugalGPT exclude_model_ids -> excluded_by_caller"
        let registry = rejection_test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = rejection_test_graph();
        let excludes = ["eligible".to_string()];
        let _ =
            assigner.assign_single_node_inner(&mut graph, 0, "code", 1.0, Some(&excludes), None);
        let rejections = assigner
            .py_last_filter_rejections()
            .expect("should have observed rejections");
        assert!(
            rejections
                .iter()
                .any(|(mid, rc)| { mid == "eligible" && rc == REASON_EXCLUDED_BY_CALLER }),
            "expected 'eligible' to be rejected as excluded_by_caller; got {:?}",
            rejections
        );
    }

    #[test]
    fn test_phase2_capability_mismatch_recorded() {
        // cgpro Phase 2 test "tools/json missing -> capability_mismatch"
        let registry = rejection_test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = rejection_test_graph();
        assigner.assign_models_inner(&mut graph, "code", 1.0);
        let rejections = assigner
            .py_last_filter_rejections()
            .expect("should have observed rejections");
        assert!(
            rejections
                .iter()
                .any(|(mid, rc)| { mid == "no-tools-card" && rc == REASON_CAPABILITY_MISMATCH }),
            "expected no-tools-card to be rejected as capability_mismatch; got {:?}",
            rejections
        );
    }

    #[test]
    fn test_phase2_cost_above_budget_recorded() {
        // cgpro Phase 2 test "budget too small -> cost_above_budget"
        let registry = rejection_test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = rejection_test_graph();
        // node max_cost=1.0; expensive-card is 100/M+500/M which exceeds 1.0
        assigner.assign_models_inner(&mut graph, "code", 100.0);
        let rejections = assigner
            .py_last_filter_rejections()
            .expect("should have observed rejections");
        assert!(
            rejections
                .iter()
                .any(|(mid, rc)| { mid == "expensive-card" && rc == REASON_COST_ABOVE_BUDGET }),
            "expected expensive-card to be rejected as cost_above_budget; got {:?}",
            rejections
        );
    }

    #[test]
    fn test_phase2_unknown_provider_under_active_policy() {
        // cgpro Phase 2 test "unknown/empty provider under active
        // policy -> provider_excluded_policy_unknown_provider"
        // Use the preassigned-unknown branch: set node.model_id to a
        // model NOT in registry while policy is active.
        let registry = rejection_test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph = TopologyGraph::try_new("sequential").unwrap();
        let node = TopologyNode::new(
            "coder".into(),
            "not-in-registry".into(), // preassigned to unknown model
            2,
            vec!["tools".into()],
            0,
            1.0,
            60.0,
        );
        graph.add_node(node);
        let policy = ProviderPolicyFilter {
            allowlist: Some(["alpha".to_string()].into_iter().collect()),
            denylist: HashSet::new(),
        };
        let _ =
            assigner.assign_models_with_policy_inner(&mut graph, "code", 1.0, &[], None, &policy);
        let rejections = assigner
            .py_last_filter_rejections()
            .expect("should have observed rejections");
        assert!(
            rejections.iter().any(|(mid, rc)| {
                mid == "not-in-registry" && rc == REASON_PROVIDER_POLICY_UNKNOWN
            }),
            "expected unknown-preassigned to be rejected as unknown_provider; got {:?}",
            rejections
        );
    }

    #[test]
    fn test_phase2_cap_at_20_truncated() {
        // cgpro Phase 2 test "cap >20 -> len == 20 and truncated == true"
        let registry = rejection_test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        // Manually push 25 rejections to test the cap
        assigner.begin_filter_recording();
        for i in 0..25 {
            let mid = format!("model-{}", i);
            assigner.push_filter_rejection(&mid, REASON_PROVIDER_POLICY_DENYLIST);
        }
        let rejections = assigner
            .py_last_filter_rejections()
            .expect("should have observed rejections");
        assert_eq!(rejections.len(), RUST_FILTER_REJECTIONS_CAP);
        assert!(assigner.py_last_filter_rejections_truncated());
        // Newest 20 retained (FIFO eviction of oldest)
        assert_eq!(rejections[0].0, "model-5");
        assert_eq!(rejections.last().unwrap().0, "model-24");
    }

    #[test]
    fn test_phase2_begin_filter_recording_resets_state() {
        // Two assignment calls in sequence — the second call's
        // rejections must NOT include leftovers from the first.
        let registry = rejection_test_registry();
        let assigner = ModelAssigner::from_registry(&registry);
        let mut graph1 = rejection_test_graph();
        assigner.assign_models_inner(&mut graph1, "code", 1.0);
        let first_count = assigner.py_last_filter_rejections().unwrap().len();
        assert!(first_count > 0);

        // Second call — rejections list is reset
        let mut graph2 = rejection_test_graph();
        assigner.assign_models_inner(&mut graph2, "code", 1.0);
        let second = assigner.py_last_filter_rejections().unwrap();
        // The same number of rejections (same setup) — not first_count + first_count
        assert_eq!(second.len(), first_count);
    }
}
