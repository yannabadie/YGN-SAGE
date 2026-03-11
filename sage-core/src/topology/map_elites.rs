//! MAP-Elites archive — N-dimensional grid archive for topology quality-diversity search.
//!
//! The archive stores elite topologies indexed by a 4-dimensional behavior descriptor:
//! agent count, max depth, cost range, and model diversity. Each cell holds the
//! best-known topology for that behavioral niche.
//!
//! Insertion uses Pareto dominance: a new entry replaces an existing one only if it
//! has strictly higher quality AND strictly lower cost. All topologies are verified
//! via `HybridVerifier` before insertion — invalid topologies are rejected.

use std::collections::HashMap;
use tracing::{debug, info, info_span};

use crate::topology::topology_graph::*;
use crate::topology::verifier::HybridVerifier;

use petgraph::visit::EdgeRef;

// ── BehaviorDescriptor ──────────────────────────────────────────────────────

/// 4-dimensional behavior descriptor bucketing a topology into a grid cell.
///
/// The 4 bucket values form a composite key into the MAP-Elites grid archive.
/// Total possible cells = 4 * 3 * 3 * 3 = 108.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BehaviorDescriptor {
    /// Bucket index for number of agents (1=solo, 2=pair, 3=small team 3-5, 4=large 6+).
    pub agent_count_bucket: u8,
    /// Bucket index for graph depth (1=shallow 1-2, 2=medium 3-4, 3=deep 5+).
    pub max_depth_bucket: u8,
    /// Bucket for cost range (1=cheap <$0.01, 2=moderate $0.01-$0.10, 3=expensive >$0.10).
    pub cost_bucket: u8,
    /// Bucket for model diversity (1=homogeneous <0.3, 2=mixed 0.3-0.7, 3=diverse >0.7).
    pub model_diversity_bucket: u8,
}

impl BehaviorDescriptor {
    /// Total number of possible cells in the grid (4 * 3 * 3 * 3).
    pub const TOTAL_CELLS: usize = 4 * 3 * 3 * 3;

    /// Extract a behavior descriptor from a `TopologyGraph` and measured cost.
    ///
    /// Computes agent count, max depth (longest path in the DAG via control edges),
    /// and model diversity (fraction of unique models) from the graph structure.
    pub fn from_topology(graph: &TopologyGraph, total_cost: f32) -> Self {
        let agent_count = graph.node_count() as u32;
        let max_depth = compute_max_depth(graph);
        let model_diversity = compute_model_diversity(graph);

        Self::from_raw(agent_count, max_depth, total_cost, model_diversity)
    }

    /// Create a descriptor from raw feature values, applying bucketing logic.
    pub fn from_raw(agent_count: u32, max_depth: u32, cost: f32, model_diversity: f32) -> Self {
        Self {
            agent_count_bucket: bucket_agent_count(agent_count),
            max_depth_bucket: bucket_max_depth(max_depth),
            cost_bucket: bucket_cost(cost),
            model_diversity_bucket: bucket_model_diversity(model_diversity),
        }
    }

    /// Return the composite key tuple for HashMap lookups.
    pub fn key(&self) -> (u8, u8, u8, u8) {
        (
            self.agent_count_bucket,
            self.max_depth_bucket,
            self.cost_bucket,
            self.model_diversity_bucket,
        )
    }
}

// ── Bucketing functions ─────────────────────────────────────────────────────

/// Bucket agent count: 1=solo, 2=pair, 3=small team (3-5), 4=large (6+).
fn bucket_agent_count(count: u32) -> u8 {
    match count {
        0..=1 => 1,
        2 => 2,
        3..=5 => 3,
        _ => 4,
    }
}

/// Bucket max depth: 1=shallow (1-2), 2=medium (3-4), 3=deep (5+).
fn bucket_max_depth(depth: u32) -> u8 {
    match depth {
        0..=2 => 1,
        3..=4 => 2,
        _ => 3,
    }
}

/// Bucket cost: 1=cheap (<$0.01), 2=moderate ($0.01-$0.10), 3=expensive (>$0.10).
/// NaN falls through to bucket 3 (expensive) since NaN < x is always false.
fn bucket_cost(cost: f32) -> u8 {
    if cost < 0.01 {
        1
    } else if cost <= 0.10 {
        2
    } else {
        3
    }
}

/// Bucket model diversity: 1=homogeneous (<0.3), 2=mixed (0.3-0.7), 3=diverse (>0.7).
fn bucket_model_diversity(diversity: f32) -> u8 {
    if diversity < 0.3 {
        1
    } else if diversity <= 0.7 {
        2
    } else {
        3
    }
}

// ── Graph feature extraction ────────────────────────────────────────────────

/// Compute the longest path (max depth) in a topology graph.
///
/// Traverses ALL edge types (control, message, state) since pipeline depth
/// reflects the full data flow, not just control ordering. For cyclic graphs
/// (e.g., AVR with back-edges), falls back to node count as an upper bound.
fn compute_max_depth(graph: &TopologyGraph) -> u32 {
    let inner = graph.inner_graph();
    if inner.node_count() == 0 {
        return 0;
    }

    // Build a control-only DAG (open-gate control edges only).
    let entry_nodes = graph.entry_nodes();
    if entry_nodes.is_empty() {
        // No entry nodes => cyclic or disconnected, use node count as upper bound.
        return inner.node_count() as u32;
    }

    // BFS-based longest-path from entry nodes via all edges.
    let mut max_depth: u32 = 0;
    let mut depths = vec![0u32; inner.node_count()];
    let mut visited = vec![false; inner.node_count()];

    // Topological order if acyclic; otherwise fall back to node count.
    let topo_order = match graph.try_topological_sort() {
        Ok(order) => order,
        Err(_) => return inner.node_count() as u32,
    };

    // Mark entry nodes as depth 0.
    for &entry in &entry_nodes {
        visited[entry] = true;
        depths[entry] = 0;
    }

    // Process nodes in topological order to compute longest path.
    for &node_idx in &topo_order {
        visited[node_idx] = true;
        let current_depth = depths[node_idx];
        if current_depth > max_depth {
            max_depth = current_depth;
        }

        // Propagate to neighbors via control edges.
        for edge_ref in inner.edges_directed(
            petgraph::graph::NodeIndex::new(node_idx),
            petgraph::Direction::Outgoing,
        ) {
            let target = edge_ref.target().index();
            let new_depth = current_depth + 1;
            if new_depth > depths[target] {
                depths[target] = new_depth;
            }
        }
    }

    max_depth
}

/// Compute model diversity as the fraction of unique model IDs among all nodes.
///
/// Returns 0.0 for empty graphs, 1.0 / count for single-model graphs,
/// and up to 1.0 for fully diverse graphs.
fn compute_model_diversity(graph: &TopologyGraph) -> f32 {
    let inner = graph.inner_graph();
    let count = inner.node_count();
    if count == 0 {
        return 0.0;
    }

    let unique_models: std::collections::HashSet<&str> =
        inner.node_weights().map(|n| n.model_id.as_str()).collect();

    unique_models.len() as f32 / count as f32
}

// ── EliteEntry ──────────────────────────────────────────────────────────────

/// An elite entry in the MAP-Elites archive — a topology with its measured performance.
#[derive(Debug, Clone)]
pub struct EliteEntry {
    /// The topology graph itself.
    pub graph: TopologyGraph,
    /// Measured quality score [0, 1].
    pub quality: f32,
    /// Measured cost in USD.
    pub cost: f32,
    /// Measured latency in milliseconds.
    pub latency_ms: f32,
    /// How many times this entry has been evaluated.
    pub evaluation_count: u32,
}

// ── Serializable proxy for TopologyGraph ────────────────────────────────────

/// Serializable representation of a TopologyGraph for SQLite persistence.
///
/// petgraph's DiGraph doesn't have serde enabled in this project, so we
/// serialize the graph as a list of nodes + edges + metadata.
#[cfg(feature = "cognitive")]
mod serde_proxy {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize)]
    pub struct SerializableNode {
        pub node_id: String,
        pub role: String,
        pub model_id: String,
        pub system: u8,
        pub required_capabilities: Vec<String>,
        pub security_label: u8,
        pub max_cost_usd: f32,
        pub max_wall_time_s: f32,
    }

    #[derive(Serialize, Deserialize)]
    pub struct SerializableEdge {
        pub from: usize,
        pub to: usize,
        pub edge_type: String,
        pub field_mapping: Option<HashMap<String, String>>,
        pub gate: String,
        pub condition: Option<String>,
        pub weight: f32,
    }

    #[derive(Serialize, Deserialize)]
    pub struct SerializableGraph {
        pub id: String,
        pub template_type: String,
        pub nodes: Vec<SerializableNode>,
        pub edges: Vec<SerializableEdge>,
    }

    impl SerializableGraph {
        pub fn from_topology(graph: &TopologyGraph) -> Self {
            let inner = graph.inner_graph();
            let nodes: Vec<SerializableNode> = inner
                .node_weights()
                .map(|n| SerializableNode {
                    node_id: n.node_id.clone(),
                    role: n.role.clone(),
                    model_id: n.model_id.clone(),
                    system: n.system,
                    required_capabilities: n.required_capabilities.clone(),
                    security_label: n.security_label,
                    max_cost_usd: n.max_cost_usd,
                    max_wall_time_s: n.max_wall_time_s,
                })
                .collect();

            let edges: Vec<SerializableEdge> = inner
                .edge_references()
                .map(|e| SerializableEdge {
                    from: e.source().index(),
                    to: e.target().index(),
                    edge_type: e.weight().edge_type.clone(),
                    field_mapping: e.weight().field_mapping.clone(),
                    gate: e.weight().gate.clone(),
                    condition: e.weight().condition.clone(),
                    weight: e.weight().weight,
                })
                .collect();

            SerializableGraph {
                id: graph.id.clone(),
                template_type: graph.template_type.clone(),
                nodes,
                edges,
            }
        }

        pub fn to_topology(&self) -> Result<TopologyGraph, String> {
            let mut graph = TopologyGraph::try_new(&self.template_type)?;
            // Override the auto-generated id with the stored one.
            graph.id = self.id.clone();

            for node in &self.nodes {
                graph.add_node(TopologyNode::new(
                    node.role.clone(),
                    node.model_id.clone(),
                    node.system,
                    node.required_capabilities.clone(),
                    node.security_label,
                    node.max_cost_usd,
                    node.max_wall_time_s,
                ));
            }

            for edge in &self.edges {
                let e = TopologyEdge::try_new(
                    edge.edge_type.clone(),
                    edge.field_mapping.clone(),
                    edge.gate.clone(),
                    edge.condition.clone(),
                    edge.weight,
                )?;
                graph.try_add_edge(edge.from, edge.to, e)?;
            }

            Ok(graph)
        }
    }
}

// ── MapElitesArchive ────────────────────────────────────────────────────────

/// N-dimensional grid archive for topology quality-diversity search.
///
/// Stores elite topologies indexed by 4-dimensional behavior descriptors.
/// Each cell holds the best-known topology for that behavioral niche.
/// All insertions are verified via `HybridVerifier`.
pub struct MapElitesArchive {
    /// Grid cells: (agent_count_bucket, max_depth_bucket, cost_bucket, model_diversity_bucket) -> elite.
    cells: HashMap<(u8, u8, u8, u8), EliteEntry>,
    /// Verifier for topology validation.
    verifier: HybridVerifier,
}

impl Default for MapElitesArchive {
    fn default() -> Self {
        Self::new()
    }
}

impl MapElitesArchive {
    /// Create a new empty archive with default verifier.
    pub fn new() -> Self {
        Self {
            cells: HashMap::new(),
            verifier: HybridVerifier::new(),
        }
    }

    /// Insert a topology into the archive at the given descriptor's cell.
    ///
    /// Inserts if the cell is empty OR the new entry Pareto-dominates the existing
    /// one (strictly higher quality AND strictly lower cost). Returns `true` if
    /// inserted, `false` if rejected.
    ///
    /// Invalid topologies (per `HybridVerifier`) are always rejected.
    pub fn insert(
        &mut self,
        descriptor: &BehaviorDescriptor,
        graph: TopologyGraph,
        quality: f32,
        cost: f32,
        latency_ms: f32,
    ) -> bool {
        let _span = info_span!(
            "map_elites.insert",
            agent_bucket = descriptor.agent_count_bucket,
            depth_bucket = descriptor.max_depth_bucket,
            cost_bucket = descriptor.cost_bucket,
            diversity_bucket = descriptor.model_diversity_bucket,
            quality = quality,
            cost = cost,
        )
        .entered();

        // Validate topology before insertion.
        let result = self.verifier.verify(&graph);
        if !result.valid {
            debug!(
                errors = ?result.errors,
                "map_elites_insert_rejected_invalid"
            );
            return false;
        }

        let key = descriptor.key();

        // Check if cell is empty or new entry Pareto-dominates.
        if let Some(existing) = self.cells.get(&key) {
            // Pareto dominance: strictly higher quality AND strictly lower cost.
            if quality > existing.quality && cost < existing.cost {
                debug!(
                    old_quality = existing.quality,
                    new_quality = quality,
                    old_cost = existing.cost,
                    new_cost = cost,
                    "map_elites_pareto_replacement"
                );
            } else {
                debug!(
                    existing_quality = existing.quality,
                    existing_cost = existing.cost,
                    "map_elites_insert_rejected_not_dominating"
                );
                return false;
            }
        }

        self.cells.insert(
            key,
            EliteEntry {
                graph,
                quality,
                cost,
                latency_ms,
                evaluation_count: 1,
            },
        );

        info!(
            cell = ?key,
            quality = quality,
            cost = cost,
            total_cells = self.cells.len(),
            "map_elites_entry_inserted"
        );

        true
    }

    /// Look up an elite entry by behavior descriptor.
    pub fn get(&self, descriptor: &BehaviorDescriptor) -> Option<&EliteEntry> {
        self.cells.get(&descriptor.key())
    }

    /// Find the entry with the highest quality across all cells.
    pub fn best_by_quality(&self) -> Option<&EliteEntry> {
        self.cells.values().max_by(|a, b| {
            a.quality
                .partial_cmp(&b.quality)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Iterate all entries as (key tuple, &EliteEntry) pairs.
    #[allow(clippy::type_complexity)]
    pub fn all_entries(&self) -> Vec<((u8, u8, u8, u8), &EliteEntry)> {
        self.cells.iter().map(|(&k, v)| (k, v)).collect()
    }

    /// Number of occupied cells.
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Fraction of possible cells occupied (total possible = 108).
    pub fn coverage(&self) -> f32 {
        self.cells.len() as f32 / BehaviorDescriptor::TOTAL_CELLS as f32
    }

    /// Persist the archive to a SQLite database.
    #[cfg(feature = "cognitive")]
    pub fn save_to_sqlite(&self, path: &str) -> Result<(), String> {
        use rusqlite::{params, Connection};

        let conn = Connection::open(path).map_err(|e| format!("SQLite open: {}", e))?;
        conn.execute_batch("PRAGMA journal_mode=WAL;")
            .map_err(|e| format!("WAL: {}", e))?;
        conn.execute_batch(CREATE_TABLE)
            .map_err(|e| format!("Create table: {}", e))?;

        // Clear existing entries before saving (full replace).
        conn.execute("DELETE FROM map_elites_entries", [])
            .map_err(|e| format!("Delete: {}", e))?;

        let mut stmt = conn
            .prepare(
                "INSERT INTO map_elites_entries \
                 (agent_count_bucket, max_depth_bucket, cost_bucket, model_diversity_bucket, \
                  topology_json, quality, cost, latency_ms, evaluation_count) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            )
            .map_err(|e| format!("Prepare: {}", e))?;

        for (&(acb, mdb, cb, dvb), entry) in &self.cells {
            let proxy = serde_proxy::SerializableGraph::from_topology(&entry.graph);
            let json =
                serde_json::to_string(&proxy).map_err(|e| format!("Serialize topology: {}", e))?;

            stmt.execute(params![
                acb as i32,
                mdb as i32,
                cb as i32,
                dvb as i32,
                json,
                entry.quality as f64,
                entry.cost as f64,
                entry.latency_ms as f64,
                entry.evaluation_count,
            ])
            .map_err(|e| format!("Insert: {}", e))?;
        }

        info!(
            path = path,
            cell_count = self.cells.len(),
            "map_elites_saved_to_sqlite"
        );

        Ok(())
    }

    /// Load an archive from a SQLite database.
    ///
    /// Note: loaded entries bypass `HybridVerifier` since they were validated
    /// at insertion time. If verifier rules change between save and load,
    /// call `verify_all()` after loading to re-validate.
    #[cfg(feature = "cognitive")]
    pub fn load_from_sqlite(path: &str) -> Result<Self, String> {
        use rusqlite::Connection;

        let conn = Connection::open(path).map_err(|e| format!("SQLite open: {}", e))?;
        conn.execute_batch("PRAGMA journal_mode=WAL;")
            .map_err(|e| format!("WAL: {}", e))?;
        conn.execute_batch(CREATE_TABLE)
            .map_err(|e| format!("Create table: {}", e))?;

        let mut stmt = conn
            .prepare(
                "SELECT agent_count_bucket, max_depth_bucket, cost_bucket, \
                 model_diversity_bucket, topology_json, quality, cost, \
                 latency_ms, evaluation_count \
                 FROM map_elites_entries",
            )
            .map_err(|e| format!("Prepare: {}", e))?;

        let mut archive = Self::new();

        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i32>(0)? as u8,  // agent_count_bucket
                    row.get::<_, i32>(1)? as u8,  // max_depth_bucket
                    row.get::<_, i32>(2)? as u8,  // cost_bucket
                    row.get::<_, i32>(3)? as u8,  // model_diversity_bucket
                    row.get::<_, String>(4)?,     // topology_json
                    row.get::<_, f64>(5)? as f32, // quality
                    row.get::<_, f64>(6)? as f32, // cost
                    row.get::<_, f64>(7)? as f32, // latency_ms
                    row.get::<_, u32>(8)?,        // evaluation_count
                ))
            })
            .map_err(|e| format!("Query: {}", e))?;

        for row_result in rows {
            let (acb, mdb, cb, dvb, json, quality, cost, latency_ms, eval_count) =
                row_result.map_err(|e| format!("Row: {}", e))?;

            let proxy: serde_proxy::SerializableGraph =
                serde_json::from_str(&json).map_err(|e| format!("Deserialize topology: {}", e))?;
            let graph = proxy.to_topology()?;

            archive.cells.insert(
                (acb, mdb, cb, dvb),
                EliteEntry {
                    graph,
                    quality,
                    cost,
                    latency_ms,
                    evaluation_count: eval_count,
                },
            );
        }

        info!(
            path = path,
            cell_count = archive.cells.len(),
            "map_elites_loaded_from_sqlite"
        );

        Ok(archive)
    }
}

/// SQLite schema for MAP-Elites archive persistence.
#[cfg(feature = "cognitive")]
const CREATE_TABLE: &str = r#"
    CREATE TABLE IF NOT EXISTS map_elites_entries (
        agent_count_bucket INTEGER NOT NULL,
        max_depth_bucket INTEGER NOT NULL,
        cost_bucket INTEGER NOT NULL,
        model_diversity_bucket INTEGER NOT NULL,
        topology_json TEXT NOT NULL,
        quality REAL NOT NULL,
        cost REAL NOT NULL,
        latency_ms REAL NOT NULL,
        evaluation_count INTEGER NOT NULL,
        PRIMARY KEY (agent_count_bucket, max_depth_bucket, cost_bucket, model_diversity_bucket)
    )
"#;

// ── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::templates;

    // -- Helper: build a valid topology graph --------------------------------

    fn make_valid_graph(template: &str, model: &str) -> TopologyGraph {
        templates::TemplateStore::create(template, model).unwrap()
    }

    fn make_multi_model_graph() -> TopologyGraph {
        let mut g = TopologyGraph::try_new("sequential").unwrap();
        let n0 = TopologyNode::new(
            "coder".into(),
            "gemini-flash".into(),
            1,
            vec!["text".into()],
            0,
            1.0,
            60.0,
        );
        let n1 = TopologyNode::new(
            "reviewer".into(),
            "gpt-5".into(),
            2,
            vec!["reasoning".into()],
            0,
            2.0,
            120.0,
        );
        let n2 = TopologyNode::new(
            "formatter".into(),
            "claude-4".into(),
            1,
            vec!["text".into()],
            0,
            1.0,
            60.0,
        );
        let i0 = g.add_node(n0);
        let i1 = g.add_node(n1);
        let i2 = g.add_node(n2);
        g.try_add_edge(i0, i1, TopologyEdge::control()).unwrap();
        g.try_add_edge(i1, i2, TopologyEdge::control()).unwrap();
        g
    }

    // -- Test 1: Empty archive -----------------------------------------------

    #[test]
    fn test_empty_archive_has_zero_cells() {
        let archive = MapElitesArchive::new();
        assert_eq!(archive.cell_count(), 0);
        assert!(archive.best_by_quality().is_none());
        assert!(archive.all_entries().is_empty());
        assert!((archive.coverage() - 0.0).abs() < f32::EPSILON);
    }

    // -- Test 2: Insert into empty cell succeeds -----------------------------

    #[test]
    fn test_insert_into_empty_cell_succeeds() {
        let mut archive = MapElitesArchive::new();
        let graph = make_valid_graph("sequential", "model-a");
        let desc = BehaviorDescriptor::from_raw(3, 2, 0.05, 0.5);

        let inserted = archive.insert(&desc, graph, 0.9, 0.05, 100.0);
        assert!(inserted);
        assert_eq!(archive.cell_count(), 1);

        let entry = archive.get(&desc).unwrap();
        assert!((entry.quality - 0.9).abs() < f32::EPSILON);
        assert!((entry.cost - 0.05).abs() < f32::EPSILON);
        assert_eq!(entry.evaluation_count, 1);
    }

    // -- Test 3: Insert invalid topology rejected ----------------------------

    #[test]
    fn test_insert_invalid_topology_rejected() {
        let mut archive = MapElitesArchive::new();

        // Empty graph with a node that has no model_id (fails capability check).
        let mut graph = TopologyGraph::try_new("sequential").unwrap();
        let node = TopologyNode::new(
            "broken".into(),
            "".into(), // empty model_id
            1,
            vec![], // empty capabilities too
            0,
            1.0,
            60.0,
        );
        graph.add_node(node);

        let desc = BehaviorDescriptor::from_raw(1, 1, 0.005, 0.0);
        let inserted = archive.insert(&desc, graph, 0.8, 0.005, 50.0);
        assert!(!inserted, "Invalid topology should be rejected");
        assert_eq!(archive.cell_count(), 0);
    }

    // -- Test 4: Pareto domination replaces inferior entry --------------------

    #[test]
    fn test_pareto_domination_replaces_inferior() {
        let mut archive = MapElitesArchive::new();
        let desc = BehaviorDescriptor::from_raw(3, 2, 0.05, 0.5);

        // Insert initial entry.
        let graph1 = make_valid_graph("sequential", "model-a");
        assert!(archive.insert(&desc, graph1, 0.7, 0.08, 200.0));

        // Insert Pareto-dominating entry (higher quality AND lower cost).
        let graph2 = make_valid_graph("sequential", "model-b");
        assert!(archive.insert(&desc, graph2, 0.9, 0.03, 100.0));

        // Verify replacement.
        assert_eq!(archive.cell_count(), 1);
        let entry = archive.get(&desc).unwrap();
        assert!((entry.quality - 0.9).abs() < f32::EPSILON);
        assert!((entry.cost - 0.03).abs() < f32::EPSILON);
    }

    // -- Test 5: Non-dominating insert rejected ------------------------------

    #[test]
    fn test_non_dominating_insert_rejected() {
        let mut archive = MapElitesArchive::new();
        let desc = BehaviorDescriptor::from_raw(3, 2, 0.05, 0.5);

        // Insert initial entry with good quality and low cost.
        let graph1 = make_valid_graph("sequential", "model-a");
        assert!(archive.insert(&desc, graph1, 0.9, 0.03, 100.0));

        // Try to insert with lower quality AND higher cost (non-dominating).
        let graph2 = make_valid_graph("sequential", "model-b");
        assert!(!archive.insert(&desc, graph2, 0.7, 0.08, 200.0));

        // Also rejected: higher quality but also higher cost.
        let graph3 = make_valid_graph("sequential", "model-c");
        assert!(!archive.insert(&desc, graph3, 0.95, 0.05, 150.0));

        // Also rejected: lower cost but also lower quality.
        let graph4 = make_valid_graph("sequential", "model-d");
        assert!(!archive.insert(&desc, graph4, 0.8, 0.01, 80.0));

        // Verify original entry unchanged.
        let entry = archive.get(&desc).unwrap();
        assert!((entry.quality - 0.9).abs() < f32::EPSILON);
        assert!((entry.cost - 0.03).abs() < f32::EPSILON);
    }

    // -- Test 6: best_by_quality finds highest -------------------------------

    #[test]
    fn test_best_by_quality_finds_highest() {
        let mut archive = MapElitesArchive::new();

        // Insert three entries in different cells.
        let desc1 = BehaviorDescriptor::from_raw(1, 1, 0.005, 0.1);
        let desc2 = BehaviorDescriptor::from_raw(3, 2, 0.05, 0.5);
        let desc3 = BehaviorDescriptor::from_raw(6, 5, 0.20, 0.9);

        archive.insert(
            &desc1,
            make_valid_graph("sequential", "m"),
            0.7,
            0.005,
            50.0,
        );
        archive.insert(
            &desc2,
            make_valid_graph("sequential", "m"),
            0.95,
            0.05,
            100.0,
        );
        archive.insert(
            &desc3,
            make_valid_graph("sequential", "m"),
            0.8,
            0.20,
            200.0,
        );

        let best = archive.best_by_quality().unwrap();
        assert!((best.quality - 0.95).abs() < f32::EPSILON);
    }

    // -- Test 7: all_entries returns correct count ----------------------------

    #[test]
    fn test_all_entries_returns_correct_count() {
        let mut archive = MapElitesArchive::new();

        let descriptors = [
            BehaviorDescriptor::from_raw(1, 1, 0.005, 0.1),
            BehaviorDescriptor::from_raw(2, 3, 0.05, 0.5),
            BehaviorDescriptor::from_raw(4, 5, 0.20, 0.9),
        ];

        for (i, desc) in descriptors.iter().enumerate() {
            archive.insert(
                desc,
                make_valid_graph("sequential", "m"),
                0.5 + i as f32 * 0.1,
                0.005 + i as f32 * 0.05,
                50.0 + i as f32 * 50.0,
            );
        }

        let entries = archive.all_entries();
        assert_eq!(entries.len(), 3);
        assert_eq!(archive.cell_count(), 3);
    }

    // -- Test 8: Coverage calculation ----------------------------------------

    #[test]
    fn test_coverage_calculation() {
        let mut archive = MapElitesArchive::new();
        assert!((archive.coverage() - 0.0).abs() < f32::EPSILON);

        // Insert one entry.
        let desc = BehaviorDescriptor::from_raw(1, 1, 0.005, 0.1);
        archive.insert(&desc, make_valid_graph("sequential", "m"), 0.8, 0.005, 50.0);

        // Coverage should be 1/108.
        let expected = 1.0 / 108.0;
        assert!(
            (archive.coverage() - expected).abs() < 1e-5,
            "Expected coverage ~{}, got {}",
            expected,
            archive.coverage()
        );

        assert_eq!(BehaviorDescriptor::TOTAL_CELLS, 108);
    }

    // -- Test 9: BehaviorDescriptor bucketing logic --------------------------

    #[test]
    fn test_behavior_descriptor_bucketing() {
        // Agent count buckets.
        assert_eq!(bucket_agent_count(0), 1);
        assert_eq!(bucket_agent_count(1), 1);
        assert_eq!(bucket_agent_count(2), 2);
        assert_eq!(bucket_agent_count(3), 3);
        assert_eq!(bucket_agent_count(5), 3);
        assert_eq!(bucket_agent_count(6), 4);
        assert_eq!(bucket_agent_count(100), 4);

        // Max depth buckets.
        assert_eq!(bucket_max_depth(0), 1);
        assert_eq!(bucket_max_depth(1), 1);
        assert_eq!(bucket_max_depth(2), 1);
        assert_eq!(bucket_max_depth(3), 2);
        assert_eq!(bucket_max_depth(4), 2);
        assert_eq!(bucket_max_depth(5), 3);
        assert_eq!(bucket_max_depth(10), 3);

        // Cost buckets.
        assert_eq!(bucket_cost(0.0), 1);
        assert_eq!(bucket_cost(0.005), 1);
        assert_eq!(bucket_cost(0.009), 1);
        assert_eq!(bucket_cost(0.01), 2);
        assert_eq!(bucket_cost(0.05), 2);
        assert_eq!(bucket_cost(0.10), 2);
        assert_eq!(bucket_cost(0.11), 3);
        assert_eq!(bucket_cost(1.0), 3);

        // Model diversity buckets.
        assert_eq!(bucket_model_diversity(0.0), 1);
        assert_eq!(bucket_model_diversity(0.29), 1);
        assert_eq!(bucket_model_diversity(0.3), 2);
        assert_eq!(bucket_model_diversity(0.5), 2);
        assert_eq!(bucket_model_diversity(0.7), 2);
        assert_eq!(bucket_model_diversity(0.71), 3);
        assert_eq!(bucket_model_diversity(1.0), 3);

        // Full descriptor from raw values.
        let desc = BehaviorDescriptor::from_raw(4, 3, 0.05, 0.5);
        assert_eq!(desc.agent_count_bucket, 3); // 3-5 => bucket 3
        assert_eq!(desc.max_depth_bucket, 2); // 3-4 => bucket 2
        assert_eq!(desc.cost_bucket, 2); // $0.01-$0.10 => bucket 2
        assert_eq!(desc.model_diversity_bucket, 2); // 0.3-0.7 => bucket 2
        assert_eq!(desc.key(), (3, 2, 2, 2));
    }

    // -- Test 10: from_topology extracts correct features --------------------

    #[test]
    fn test_from_topology_extracts_features() {
        // Sequential template: 3 nodes, depth 2, single model => diversity = 1/3.
        let graph = make_valid_graph("sequential", "same-model");
        let desc = BehaviorDescriptor::from_topology(&graph, 0.005);

        assert_eq!(graph.node_count(), 3);
        assert_eq!(desc.agent_count_bucket, 3); // 3 agents => bucket 3 (small team)
        assert_eq!(desc.max_depth_bucket, 1); // depth 2 => bucket 1 (shallow)
        assert_eq!(desc.cost_bucket, 1); // $0.005 => bucket 1 (cheap)
                                         // 1 unique model out of 3 nodes = 1/3 = 0.333 => bucket 2 (0.3-0.7).
        assert_eq!(desc.model_diversity_bucket, 2);

        // Multi-model graph: 3 nodes, 3 different models => diversity = 1.0.
        let multi = make_multi_model_graph();
        let desc2 = BehaviorDescriptor::from_topology(&multi, 0.20);

        assert_eq!(multi.node_count(), 3);
        assert_eq!(desc2.agent_count_bucket, 3); // 3 agents => bucket 3
        assert_eq!(desc2.cost_bucket, 3); // $0.20 => bucket 3 (expensive)
        assert_eq!(desc2.model_diversity_bucket, 3); // 3/3 = 1.0 => bucket 3 (diverse)
    }

    // -- Test 11: Default trait implementation --------------------------------

    #[test]
    fn test_default_archive() {
        let archive = MapElitesArchive::default();
        assert_eq!(archive.cell_count(), 0);
    }

    // -- Test 12: Multiple cells populated correctly -------------------------

    #[test]
    fn test_multiple_distinct_cells() {
        let mut archive = MapElitesArchive::new();

        // Populate 5 distinct cells.
        let configs: Vec<(u32, u32, f32, f32)> = vec![
            (1, 1, 0.005, 0.1), // solo, shallow, cheap, homogeneous
            (2, 2, 0.01, 0.3),  // pair, shallow, moderate, mixed
            (4, 3, 0.05, 0.5),  // small team, medium, moderate, mixed
            (6, 5, 0.10, 0.7),  // large, deep, moderate, mixed
            (8, 6, 0.20, 0.9),  // large, deep, expensive, diverse
        ];

        for (i, &(ac, md, cost, div)) in configs.iter().enumerate() {
            let desc = BehaviorDescriptor::from_raw(ac, md, cost, div);
            let graph = make_valid_graph("sequential", &format!("model-{}", i));
            archive.insert(
                &desc,
                graph,
                0.5 + i as f32 * 0.1,
                cost,
                50.0 + i as f32 * 50.0,
            );
        }

        assert_eq!(archive.cell_count(), 5);
        assert!((archive.coverage() - 5.0 / 108.0).abs() < 1e-5);
    }

    // -- Test 13: Empty graph depth and diversity ----------------------------

    #[test]
    fn test_empty_graph_features() {
        let graph = TopologyGraph::try_new("sequential").unwrap();
        assert_eq!(compute_max_depth(&graph), 0);
        assert!((compute_model_diversity(&graph) - 0.0).abs() < f32::EPSILON);
    }

    // -- Test 14: Parallel topology feature extraction -----------------------

    #[test]
    fn test_parallel_topology_features() {
        // parallel(model, 3) => 5 nodes (1 source + 3 workers + 1 aggregator).
        let graph = make_valid_graph("parallel", "model-x");
        assert_eq!(graph.node_count(), 5);

        let depth = compute_max_depth(&graph);
        // source -> worker -> aggregator (via message edges too).
        // All edges count for depth: source->worker (control), worker->agg (message).
        assert!(
            depth >= 2,
            "Parallel graph depth should be >= 2, got {}",
            depth
        );

        let diversity = compute_model_diversity(&graph);
        // Single model => 1/5 = 0.2.
        assert!((diversity - 0.2).abs() < f32::EPSILON);
    }
}
