//! PyO3 wrappers for TopologyEngine and TopologyExecutor.
//!
//! Thin Python-facing wrappers following the same pattern as `PyMultiViewMMU`
//! in `memory/smmu.rs`. Each wrapper owns its inner Rust type and delegates
//! all methods through to it.

use pyo3::prelude::*;

use super::engine::{GenerateResult, TopologyEngine};
use super::executor::{ExecutionMode, TopologyExecutor};
use super::topology_graph::TopologyGraph;
use crate::memory::smmu::MultiViewMMU;

// ── PyMutationStats ─────────────────────────────────────────────────────────

/// Python-facing snapshot of the Thompson-sampling posterior for a single
/// mutation operator. Immutable view — a fresh instance is returned per
/// `PyTopologyEngine.mutation_stats_snapshot()` call.
///
/// The Rust `MutationStats` struct (mutations.rs) keeps posteriors for all 7
/// operators in two `[f64; 7]` arrays. We don't `#[pyclass]` it directly
/// because that would force Python ABI constraints on a struct already used
/// widely in Rust code. Instead we expose one wrapper per operator.
#[pyclass(name = "MutationStats", frozen)]
pub struct PyMutationStats {
    op_name: String,
    attempts: u32,
    successes: u32,
    alpha: f64,
    beta: f64,
}

#[pymethods]
impl PyMutationStats {
    /// Operator name (`add_node`, `remove_node`, `swap_model`, `rewire_edge`,
    /// `split_node`, `merge_nodes`, `mutate_prompt`).
    #[getter]
    pub fn op_name(&self) -> &str {
        &self.op_name
    }

    /// Number of times this operator was selected (Thompson-sampled) since
    /// the engine was created. Derived from `alpha + beta - 2` (drops the
    /// Beta(1,1) prior).
    #[getter]
    pub fn attempts(&self) -> u32 {
        self.attempts
    }

    /// Number of successful applications (produced a verifier-valid graph).
    #[getter]
    pub fn successes(&self) -> u32 {
        self.successes
    }

    /// Raw Beta posterior alpha parameter (successes + 1).
    #[getter]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Raw Beta posterior beta parameter (failures + 1).
    #[getter]
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Expected success rate = alpha / (alpha + beta).
    #[getter]
    pub fn success_rate(&self) -> f64 {
        let denom = self.alpha + self.beta;
        if denom <= 0.0 {
            0.5
        } else {
            self.alpha / denom
        }
    }

    pub fn __repr__(&self) -> String {
        format!(
            "MutationStats(op='{}', attempts={}, successes={}, alpha={:.1}, beta={:.1}, success_rate={:.3})",
            self.op_name,
            self.attempts,
            self.successes,
            self.alpha,
            self.beta,
            self.success_rate(),
        )
    }
}

// ── PyGenerateResult ────────────────────────────────────────────────────────

/// Python-facing wrapper for `GenerateResult`.
#[pyclass(name = "GenerateResult")]
pub struct PyGenerateResult {
    topology: TopologyGraph,
    source: String,
    confidence: f32,
}

impl PyGenerateResult {
    /// Create from a Rust `GenerateResult`.
    pub fn from_inner(result: GenerateResult) -> Self {
        Self {
            topology: result.topology,
            source: result.source.as_str().to_string(),
            confidence: result.confidence,
        }
    }
}

#[pymethods]
impl PyGenerateResult {
    /// The generated topology graph.
    #[getter]
    pub fn topology(&self) -> TopologyGraph {
        self.topology.clone()
    }

    /// How the topology was obtained (e.g. "smmu_hit", "archive_hit", "template_fallback").
    #[getter]
    pub fn source(&self) -> String {
        self.source.clone()
    }

    /// Confidence in the topology quality [0.0, 1.0].
    #[getter]
    pub fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Stable opaque string identifier for this topology.
    ///
    /// Derived from the template type, node count, and first 8 characters of the
    /// topology ULID.  Suitable as a cache key or logging token without exposing
    /// the full mutable internal state.  The existing `topology()` getter remains
    /// for callers that need the full graph.
    pub fn topology_id(&self) -> String {
        let node_count = self.topology.node_count();
        let id_prefix = &self.topology.id[..self.topology.id.len().min(8)];
        format!(
            "{}:n{}:{}",
            self.topology.template_type, node_count, id_prefix
        )
    }

    pub fn __repr__(&self) -> String {
        format!(
            "GenerateResult(source='{}', confidence={:.2}, template='{}')",
            self.source, self.confidence, self.topology.template_type
        )
    }
}

// ── PyTopologyEngine ────────────────────────────────────────────────────────

/// Python-facing wrapper for `TopologyEngine`.
///
/// Owns a `MultiViewMMU` internally to avoid PyO3 dual-mutable-reference
/// complexity. Python callers interact with a single unified object.
#[pyclass(name = "TopologyEngine")]
pub struct PyTopologyEngine {
    inner: TopologyEngine,
    smmu: MultiViewMMU,
}

impl Default for PyTopologyEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl PyTopologyEngine {
    /// Create a new engine with default settings and an empty S-MMU.
    #[new]
    pub fn new() -> Self {
        Self {
            inner: TopologyEngine::new(),
            smmu: MultiViewMMU::new(),
        }
    }

    /// Generate a topology for a task using the 6-path strategy.
    ///
    /// Args:
    ///     task_description: Human-readable task description.
    ///     task_embedding: Optional 768-dim embedding for semantic search.
    ///     system: Cognitive system tier (1=S1, 2=S2, 3=S3).
    ///     exploration_budget: 0.0 = pure exploit, 1.0 = pure explore.
    ///
    /// Returns:
    ///     GenerateResult with topology, source, and confidence.
    #[pyo3(signature = (task_description, task_embedding=None, system=2, exploration_budget=0.5))]
    pub fn generate(
        &mut self,
        task_description: &str,
        task_embedding: Option<Vec<f32>>,
        system: u8,
        exploration_budget: f32,
    ) -> PyResult<PyGenerateResult> {
        let result = self.inner.generate(
            &mut self.smmu,
            task_description,
            task_embedding,
            system,
            exploration_budget,
        );
        Ok(PyGenerateResult::from_inner(result))
    }

    /// Generate with per-path toggles for deterministic testing (Phase 2).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        task_description,
        task_embedding=None,
        system=2,
        exploration_budget=0.5,
        allow_smmu=true,
        allow_archive=true,
        allow_mutation=true,
        allow_mcts=true,
        allow_template=true
    ))]
    pub fn generate_with_options(
        &mut self,
        task_description: &str,
        task_embedding: Option<Vec<f32>>,
        system: u8,
        exploration_budget: f32,
        allow_smmu: bool,
        allow_archive: bool,
        allow_mutation: bool,
        allow_mcts: bool,
        allow_template: bool,
    ) -> PyResult<PyGenerateResult> {
        let options = crate::topology::engine::GenerationOptions {
            allow_smmu,
            allow_archive,
            allow_mutation,
            allow_mcts,
            allow_template,
        };
        let result = self.inner.generate_with_options(
            &mut self.smmu,
            task_description,
            task_embedding,
            system,
            exploration_budget,
            &options,
        );
        Ok(PyGenerateResult::from_inner(result))
    }

    /// Seed the archive with a single topology outcome for testing.
    #[pyo3(signature = (system=2, quality=0.85, task_summary="seed archive"))]
    pub fn seed_archive_outcome(
        &mut self,
        system: u8,
        quality: f32,
        task_summary: &str,
    ) -> String {
        self.inner.seed_archive_outcome(&mut self.smmu, system, quality, task_summary)
    }

    /// Seed the archive with diverse outcomes. Returns count inserted.
    pub fn seed_archive_diversity(&mut self, count: usize) -> usize {
        self.inner.seed_archive_diversity(&mut self.smmu, count)
    }

    /// Record the outcome of a topology execution.
    ///
    /// Feeds into S-MMU bridge, MAP-Elites archive, and contextual bandit.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (topology_id, task_summary, keywords, task_embedding=None, quality=0.5, cost=0.0, latency_ms=0.0))]
    pub fn record_outcome(
        &mut self,
        topology_id: &str,
        task_summary: &str,
        keywords: Vec<String>,
        task_embedding: Option<Vec<f32>>,
        quality: f32,
        cost: f32,
        latency_ms: f32,
    ) {
        self.inner.record_outcome(
            &mut self.smmu,
            topology_id,
            task_summary,
            keywords,
            task_embedding,
            quality,
            cost,
            latency_ms,
        );
    }

    /// Determine whether online evolution should run.
    ///
    /// Returns True when the archive has enough data and enough new outcomes
    /// have been recorded since the last evolution pass.
    ///
    /// Args:
    ///     min_outcomes: Minimum archive entries before first evolution (default 5).
    ///     cooldown_outcomes: Minimum new outcomes between evolution runs (default 3).
    #[pyo3(signature = (min_outcomes=5, cooldown_outcomes=3))]
    pub fn should_evolve(&self, min_outcomes: usize, cooldown_outcomes: usize) -> bool {
        self.inner.should_evolve(min_outcomes, cooldown_outcomes)
    }

    /// Run a synchronous evolution loop on the MAP-Elites archive.
    ///
    /// Args:
    ///     pop_size: Number of candidates per generation.
    ///     generations: Number of evolution rounds.
    #[pyo3(signature = (pop_size=10, generations=5))]
    pub fn evolve(&mut self, pop_size: usize, generations: usize) {
        self.inner.evolve(pop_size, generations);
    }

    /// Take ownership of the list of mutation operator names applied during
    /// the most recent `evolve()` call, clearing the internal buffer.
    ///
    /// Order: generation-major, candidate-major — one entry per inner
    /// per-child mutation attempt (1..=3 per candidate). Attempts that were
    /// Thompson-sampled but rejected by the verifier are also listed.
    ///
    /// Returns:
    ///     List of operator names. Empty between `evolve()` calls or when
    ///     `evolve()` was a no-op (empty archive).
    pub fn drain_last_applied_ops(&self) -> Vec<String> {
        self.inner.drain_last_applied_ops()
    }

    /// Read-only snapshot of the per-operator Thompson-sampling posteriors.
    ///
    /// Returns:
    ///     List of `MutationStats` — one per operator, in the canonical
    ///     `OPERATOR_NAMES` order (`add_node`, `remove_node`, `swap_model`,
    ///     `rewire_edge`, `split_node`, `merge_nodes`, `mutate_prompt`).
    pub fn mutation_stats_snapshot(&self) -> Vec<PyMutationStats> {
        self.inner
            .mutation_stats_snapshot()
            .into_iter()
            .map(
                |(op_name, attempts, successes, alpha, beta)| PyMutationStats {
                    op_name,
                    attempts,
                    successes,
                    alpha,
                    beta,
                },
            )
            .collect()
    }

    /// Store a topology in the cache by its id.
    pub fn cache_topology(&mut self, graph: &TopologyGraph) {
        self.inner.cache_topology(graph.clone());
    }

    /// Lazy-load a topology from cache by ULID id.
    pub fn get_topology(&self, id: &str) -> Option<TopologyGraph> {
        self.inner.get_topology(id).cloned()
    }

    /// Number of cached topologies.
    pub fn topology_count(&self) -> usize {
        self.inner.topology_count()
    }

    /// Number of occupied cells in the MAP-Elites archive.
    pub fn archive_cell_count(&self) -> usize {
        self.inner.archive().cell_count()
    }

    /// Fraction of possible archive cells occupied (total = 108).
    pub fn archive_coverage(&self) -> f32 {
        self.inner.archive().coverage()
    }

    /// Number of chunks registered in the internal S-MMU.
    pub fn smmu_chunk_count(&self) -> usize {
        self.smmu.chunk_count()
    }

    /// Save bandit posteriors + MAP-Elites archive to a directory.
    ///
    /// Creates `{dir}/bandit_state.db` and `{dir}/archive_state.db`.
    /// Requires the `cognitive` feature (SQLite persistence).
    #[cfg(feature = "cognitive")]
    pub fn save_state(&self, dir: &str) -> PyResult<()> {
        self.inner
            .save_state(dir)
            .map_err(pyo3::exceptions::PyIOError::new_err)
    }

    /// Load bandit posteriors + MAP-Elites archive from a directory.
    ///
    /// Looks for `{dir}/bandit_state.db` and `{dir}/archive_state.db`.
    /// Missing files are silently skipped (cold start).
    /// Returns `(bandit_arms, archive_cells)` loaded.
    #[cfg(feature = "cognitive")]
    pub fn load_state(&mut self, dir: &str) -> PyResult<(usize, usize)> {
        self.inner
            .load_state(dir)
            .map_err(pyo3::exceptions::PyIOError::new_err)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "TopologyEngine(cached={}, archive_cells={}, smmu_chunks={})",
            self.inner.topology_count(),
            self.inner.archive().cell_count(),
            self.smmu.chunk_count()
        )
    }
}

// ── PyTopologyExecutor ──────────────────────────────────────────────────────

/// Python-facing wrapper for `TopologyExecutor`.
///
/// Manages node scheduling for both static (DAG-based) and dynamic (gate-based)
/// execution of topology graphs.
#[pyclass(name = "TopologyExecutor")]
pub struct PyTopologyExecutor {
    inner: TopologyExecutor,
}

#[pymethods]
impl PyTopologyExecutor {
    /// Create a new executor for the given graph.
    ///
    /// Auto-selects execution mode based on the graph's template type.
    #[new]
    pub fn new(graph: &TopologyGraph) -> Self {
        Self {
            inner: TopologyExecutor::new(graph),
        }
    }

    /// Return the next wave of ready node indices.
    ///
    /// Returns an empty list when done or when max_iterations is exceeded.
    pub fn next_ready(&mut self, graph: &TopologyGraph) -> Vec<usize> {
        self.inner.next_ready(graph)
    }

    /// Mark a node as Completed.
    pub fn mark_completed(&mut self, node_index: usize) {
        self.inner.mark_completed(node_index);
    }

    /// Mark a node as Running.
    pub fn mark_running(&mut self, node_index: usize) {
        self.inner.mark_running(node_index);
    }

    /// Mark a node as Skipped (for closed-gate branches).
    pub fn mark_skipped(&mut self, node_index: usize) {
        self.inner.mark_skipped(node_index);
    }

    /// Reset a single node to Pending, enabling re-execution in multi-turn loops.
    pub fn reset_node(&mut self, node_index: usize) {
        self.inner.reset_node(node_index);
    }

    /// Open a gate on a control edge between two nodes.
    pub fn open_gate(&self, graph: &mut TopologyGraph, from: usize, to: usize) {
        self.inner.open_gate(graph, from, to);
    }

    /// Close a gate on a control edge between two nodes.
    pub fn close_gate(&self, graph: &mut TopologyGraph, from: usize, to: usize) {
        self.inner.close_gate(graph, from, to);
    }

    /// True if all nodes are either Completed or Skipped.
    pub fn is_done(&self) -> bool {
        self.inner.is_done()
    }

    /// Return the execution mode: "static" or "dynamic".
    pub fn mode(&self) -> String {
        match self.inner.mode() {
            ExecutionMode::Static => "static".to_string(),
            ExecutionMode::Dynamic => "dynamic".to_string(),
        }
    }

    pub fn __repr__(&self) -> String {
        let mode_str = match self.inner.mode() {
            ExecutionMode::Static => "static",
            ExecutionMode::Dynamic => "dynamic",
        };
        format!(
            "TopologyExecutor(mode='{}', done={})",
            mode_str,
            self.inner.is_done()
        )
    }
}

// ── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::templates;

    #[test]
    fn test_engine_new_defaults() {
        let engine = PyTopologyEngine::new();
        assert_eq!(engine.topology_count(), 0);
        assert_eq!(engine.archive_cell_count(), 0);
        assert!((engine.archive_coverage() - 0.0).abs() < f32::EPSILON);
        assert_eq!(engine.smmu_chunk_count(), 0);
    }

    #[test]
    fn test_engine_generate_template_fallback() {
        let mut engine = PyTopologyEngine::new();
        let result = engine
            .generate("Write a sorting function", None, 2, 0.5)
            .unwrap();
        assert_eq!(result.source, "template_fallback");
        assert_eq!(result.topology.template_type, "avr");
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    }

    #[test]
    fn test_engine_cache_and_get() {
        let mut engine = PyTopologyEngine::new();
        let graph = templates::sequential("test-model");
        let id = graph.id.clone();

        engine.cache_topology(&graph);
        assert_eq!(engine.topology_count(), 1);

        let retrieved = engine.get_topology(&id);
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.template_type, "sequential");
        assert_eq!(retrieved.id, id);

        // Non-existent ID returns None.
        assert!(engine.get_topology("nonexistent").is_none());
    }

    #[test]
    fn test_engine_evolve_empty_archive() {
        let mut engine = PyTopologyEngine::new();
        // evolve on empty archive should be a no-op (no panic).
        engine.evolve(10, 5);
        assert_eq!(engine.archive_cell_count(), 0);
    }

    #[test]
    fn test_executor_new_sequential() {
        let graph = templates::sequential("test-model");
        let exec = PyTopologyExecutor::new(&graph);
        assert_eq!(exec.mode(), "static");
        assert!(!exec.is_done());
    }

    #[test]
    fn test_executor_static_flow() {
        let graph = templates::sequential("test-model");
        let mut exec = PyTopologyExecutor::new(&graph);

        // Wave 1: only entry node (0).
        let wave1 = exec.next_ready(&graph);
        assert_eq!(wave1, vec![0]);

        exec.mark_running(0);
        exec.mark_completed(0);

        // Wave 2: node 1.
        let wave2 = exec.next_ready(&graph);
        assert_eq!(wave2, vec![1]);

        exec.mark_running(1);
        exec.mark_completed(1);

        // Wave 3: node 2.
        let wave3 = exec.next_ready(&graph);
        assert_eq!(wave3, vec![2]);

        exec.mark_running(2);
        exec.mark_completed(2);

        assert!(exec.is_done());
        let wave4 = exec.next_ready(&graph);
        assert!(wave4.is_empty());
    }

    #[test]
    fn test_generate_result_fields() {
        let mut engine = PyTopologyEngine::new();
        let result = engine.generate("test task", None, 1, 0.0).unwrap();

        // Check all getter fields are accessible.
        let _topology = result.topology();
        let source = result.source();
        let confidence = result.confidence();

        assert_eq!(source, "template_fallback");
        assert!((0.0..=1.0).contains(&confidence));

        // __repr__ should contain the source and template.
        let repr = result.__repr__();
        assert!(repr.contains("template_fallback"));
        assert!(repr.contains("sequential"));
    }

    #[test]
    fn test_topology_id_stable_format() {
        let mut engine = PyTopologyEngine::new();
        let result = engine.generate("sort a list", None, 1, 0.0).unwrap();

        let tid = result.topology_id();
        // Format: "<template>:n<node_count>:<8-char-ulid-prefix>"
        let parts: Vec<&str> = tid.splitn(3, ':').collect();
        assert_eq!(
            parts.len(),
            3,
            "topology_id must have 3 colon-separated parts"
        );
        assert_eq!(parts[0], "sequential");
        assert!(parts[1].starts_with('n'), "second part must start with 'n'");
        let node_count: usize = parts[1][1..].parse().expect("node count must be numeric");
        assert!(node_count > 0, "node count must be positive");
        assert_eq!(parts[2].len(), 8, "ULID prefix must be 8 chars");

        // topology() getter still works (backward compat).
        let _graph = result.topology();
    }

    #[test]
    fn test_engine_repr() {
        let engine = PyTopologyEngine::new();
        let repr = engine.__repr__();
        assert!(repr.contains("TopologyEngine"));
        assert!(repr.contains("cached=0"));
        assert!(repr.contains("archive_cells=0"));
        assert!(repr.contains("smmu_chunks=0"));
    }

    #[test]
    fn test_executor_repr() {
        let graph = templates::sequential("test-model");
        let exec = PyTopologyExecutor::new(&graph);
        let repr = exec.__repr__();
        assert!(repr.contains("TopologyExecutor"));
        assert!(repr.contains("static"));
        assert!(repr.contains("done=false"));
    }

    #[test]
    fn test_executor_dynamic_mode() {
        let graph = templates::avr("model-a", "model-b");
        let exec = PyTopologyExecutor::new(&graph);
        assert_eq!(exec.mode(), "dynamic");
    }

    #[test]
    fn test_executor_mark_skipped() {
        let graph = templates::sequential("test-model");
        let mut exec = PyTopologyExecutor::new(&graph);

        // Skip all nodes.
        exec.mark_skipped(0);
        exec.mark_skipped(1);
        exec.mark_skipped(2);

        assert!(exec.is_done());
    }

    #[test]
    fn test_engine_generate_all_systems() {
        let mut engine = PyTopologyEngine::new();

        // S1 -> sequential
        let r1 = engine.generate("hello", None, 1, 0.0).unwrap();
        assert_eq!(r1.topology.template_type, "sequential");

        // S2 -> avr
        let r2 = engine.generate("code review", None, 2, 0.0).unwrap();
        assert_eq!(r2.topology.template_type, "avr");

        // S3 -> debate
        let r3 = engine.generate("prove theorem", None, 3, 0.0).unwrap();
        assert_eq!(r3.topology.template_type, "debate");
    }

    #[test]
    fn test_engine_record_outcome_and_smmu() {
        let mut engine = PyTopologyEngine::new();

        // Generate a topology to get an ID, then cache it.
        let result = engine.generate("sorting", None, 2, 0.0).unwrap();
        let topology = result.topology();
        let id = topology.id.clone();
        engine.cache_topology(&topology);

        // Record an outcome.
        engine.record_outcome(
            &id,
            "sorting task completed",
            vec!["sort".to_string(), "algorithm".to_string()],
            None,
            0.9,
            0.01,
            150.0,
        );

        // S-MMU should have gained a chunk from the bridge.
        assert!(engine.smmu_chunk_count() > 0);
    }

    #[test]
    fn test_mutation_stats_snapshot_default_prior() {
        // Fresh engine — no mutations fired yet. Every operator should return
        // a Beta(1,1) prior with 0 attempts/successes and 50% success rate.
        let engine = PyTopologyEngine::new();
        let snapshot = engine.mutation_stats_snapshot();

        assert_eq!(
            snapshot.len(),
            7,
            "snapshot should expose all 7 mutation operators"
        );

        let expected_names = [
            "add_node",
            "remove_node",
            "swap_model",
            "rewire_edge",
            "split_node",
            "merge_nodes",
            "mutate_prompt",
        ];
        for (i, stat) in snapshot.iter().enumerate() {
            assert_eq!(stat.op_name(), expected_names[i]);
            assert_eq!(stat.attempts(), 0);
            assert_eq!(stat.successes(), 0);
            assert!((stat.alpha() - 1.0).abs() < f64::EPSILON);
            assert!((stat.beta() - 1.0).abs() < f64::EPSILON);
            assert!((stat.success_rate() - 0.5).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_drain_last_applied_ops_empty_on_fresh_engine() {
        let engine = PyTopologyEngine::new();
        assert!(engine.drain_last_applied_ops().is_empty());
    }

    #[test]
    fn test_drain_last_applied_ops_after_evolve_with_seeded_archive() {
        let mut engine = PyTopologyEngine::new();

        // Seed the archive via the public record_outcome() path so evolve()
        // has a candidate to mutate. record_outcome() reads the cached
        // topology, computes a BehaviorDescriptor, and inserts into the
        // MAP-Elites archive.
        let result = engine.generate("sorting", None, 2, 0.0).unwrap();
        let topology = result.topology();
        let id = topology.id.clone();
        engine.cache_topology(&topology);
        engine.record_outcome(
            &id,
            "seed outcome for mutation op tests",
            vec!["sort".to_string()],
            None,
            0.85,
            0.05,
            50.0,
        );
        assert!(
            engine.archive_cell_count() > 0,
            "seed record_outcome must populate at least one archive cell"
        );

        // Run evolve — this should push at least one op-name into the buffer.
        engine.evolve(2, 1);

        let ops = engine.drain_last_applied_ops();
        assert!(
            !ops.is_empty(),
            "evolve() over a seeded archive must record at least one applied op"
        );

        // Every recorded op must be one of the 7 canonical names.
        let valid_names: [&str; 7] = [
            "add_node",
            "remove_node",
            "swap_model",
            "rewire_edge",
            "split_node",
            "merge_nodes",
            "mutate_prompt",
        ];
        for op in &ops {
            assert!(
                valid_names.contains(&op.as_str()),
                "op '{}' is not a known operator name",
                op
            );
        }

        // Buffer should be drained — subsequent call returns empty.
        assert!(engine.drain_last_applied_ops().is_empty());
    }

    #[test]
    fn test_mutation_stats_snapshot_updates_after_evolve() {
        let mut engine = PyTopologyEngine::new();

        // Seed a single archive cell via record_outcome().
        let result = engine.generate("parallel map", None, 2, 0.0).unwrap();
        let topology = result.topology();
        let id = topology.id.clone();
        engine.cache_topology(&topology);
        engine.record_outcome(
            &id,
            "seeded",
            vec!["parallel".to_string()],
            None,
            0.8,
            0.05,
            50.0,
        );

        engine.evolve(3, 1);

        let snapshot = engine.mutation_stats_snapshot();
        let total_attempts: u32 = snapshot.iter().map(|s| s.attempts()).sum();
        assert!(
            total_attempts > 0,
            "total attempts across operators must grow after evolve()"
        );
    }

    #[test]
    fn test_py_mutation_stats_repr() {
        let stat = PyMutationStats {
            op_name: "swap_model".to_string(),
            attempts: 4,
            successes: 3,
            alpha: 4.0,
            beta: 2.0,
        };
        let repr = stat.__repr__();
        assert!(repr.contains("swap_model"));
        assert!(repr.contains("attempts=4"));
        assert!(repr.contains("successes=3"));
    }
}
