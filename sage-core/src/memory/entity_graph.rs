//! RESERVED FOR PHASE C: RustEntityGraph — native entity-relation graph for Tier 2 Semantic Memory.
//! Currently exported to Python but not wired. To be integrated in Phase C for fast BFS/neighborhood
//! queries in the adaptation runtime controller.
//! See: docs/superpowers/specs/2026-03-14-cognitive-orchestration-pipeline-design.md
//!
//! Unified entity graph merging semantic and causal memory.
//!
//! Replaces the Python `semantic.py` (entity-relation graph) and `causal.py`
//! (causal memory) with a single petgraph-backed structure exposed to Python
//! via PyO3.
//!
//! Edge types:
//! - **Semantic**: labelled relation from semantic memory (e.g. "works_at")
//! - **Causal**: directed cause→effect edge with confidence (from causal memory)
//! - **Temporal**: reserved for future temporal ordering

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet, VecDeque};
use tracing::instrument;

// ---------------------------------------------------------------------------
// Internal types (not PyO3-exported)
// ---------------------------------------------------------------------------

/// Edge kinds in the unified entity graph.
#[derive(Debug, Clone, PartialEq)]
enum EdgeKind {
    /// Semantic relation label (e.g. "works_at", "is_a").
    Semantic(String),
    /// Causal directed edge: (relation_label, confidence ∈ [0,1]).
    Causal(String, f32),
    /// Temporal ordering (reserved for future use).
    Temporal,
}

/// Node data for an entity in the graph.
#[derive(Debug, Clone)]
struct EntityNode {
    name: String,
    entity_type: String,
    metadata: HashMap<String, String>,
}

// ---------------------------------------------------------------------------
// Public PyO3 class
// ---------------------------------------------------------------------------

/// Unified entity graph merging semantic + causal memories.
///
/// Replaces Python `semantic.py` + `causal.py`.
/// Note: SQLite persistence is a separate task — this struct is in-memory only.
#[pyclass]
pub struct RustEntityGraph {
    graph: DiGraph<EntityNode, EdgeKind>,
    name_to_idx: HashMap<String, NodeIndex>,
}

#[pymethods]
impl RustEntityGraph {
    #[new]
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            name_to_idx: HashMap::new(),
        }
    }

    /// Add an entity (node). Returns `True` if the entity is new, `False` if it
    /// already exists (idempotent — existing node is not modified).
    #[instrument(skip(self))]
    #[pyo3(signature = (name, entity_type = "entity", metadata = None))]
    pub fn add_entity(
        &mut self,
        name: &str,
        entity_type: &str,
        metadata: Option<HashMap<String, String>>,
    ) -> bool {
        if self.name_to_idx.contains_key(name) {
            return false;
        }
        let node = EntityNode {
            name: name.to_string(),
            entity_type: entity_type.to_string(),
            metadata: metadata.unwrap_or_default(),
        };
        let idx = self.graph.add_node(node);
        self.name_to_idx.insert(name.to_string(), idx);
        true
    }

    /// Add a semantic (labelled) relation between two entities.
    ///
    /// Returns `True` on success, `False` if either entity does not exist.
    #[instrument(skip(self))]
    pub fn add_relation(
        &mut self,
        from_entity: &str,
        to_entity: &str,
        relation_type: &str,
    ) -> bool {
        let from_idx = match self.name_to_idx.get(from_entity) {
            Some(&idx) => idx,
            None => return false,
        };
        let to_idx = match self.name_to_idx.get(to_entity) {
            Some(&idx) => idx,
            None => return false,
        };
        self.graph
            .add_edge(from_idx, to_idx, EdgeKind::Semantic(relation_type.to_string()));
        true
    }

    /// Add a causal relation (directed cause→effect) with an optional confidence
    /// score in [0, 1].
    ///
    /// Returns `True` on success, `False` if either entity does not exist.
    #[instrument(skip(self))]
    #[pyo3(signature = (cause, effect, relation, confidence = 1.0))]
    pub fn add_causal_relation(
        &mut self,
        cause: &str,
        effect: &str,
        relation: &str,
        confidence: f32,
    ) -> bool {
        let from_idx = match self.name_to_idx.get(cause) {
            Some(&idx) => idx,
            None => return false,
        };
        let to_idx = match self.name_to_idx.get(effect) {
            Some(&idx) => idx,
            None => return false,
        };
        self.graph.add_edge(
            from_idx,
            to_idx,
            EdgeKind::Causal(relation.to_string(), confidence),
        );
        true
    }

    /// Get context for a task by finding matching entities and BFS-expanding
    /// neighbours up to `max_depth` hops.
    ///
    /// Matching is substring/word-overlap based (case-insensitive). Returns a
    /// newline-separated string of relation triplets. Returns an empty string
    /// when no entities match the task.
    #[instrument(skip(self))]
    #[pyo3(signature = (task, max_depth = 2))]
    pub fn get_context_for(&self, task: &str, max_depth: usize) -> String {
        let task_lower = task.to_lowercase();
        // Only use words of length >= 3 to avoid spurious matches on
        // short words ("a", "I", "to", etc.).
        let task_words: HashSet<&str> = task_lower
            .split_whitespace()
            .filter(|w| w.len() >= 3)
            .collect();

        // Seed entities whose name appears in the task (or vice-versa).
        let mut seeds: Vec<NodeIndex> = Vec::new();
        for (name, &idx) in &self.name_to_idx {
            let name_lower = name.to_lowercase();
            // Only match entity names that are at least 3 chars long.
            if name_lower.len() < 3 {
                continue;
            }
            let matches = task_words
                .iter()
                .any(|w| name_lower.contains(*w))
                || task_lower.contains(&name_lower);
            if matches {
                seeds.push(idx);
            }
        }

        if seeds.is_empty() {
            return String::new();
        }

        // BFS from seeds, collecting relation strings.
        let mut visited: HashSet<NodeIndex> = HashSet::new();
        let mut queue: VecDeque<(NodeIndex, usize)> = VecDeque::new();
        let mut context_parts: Vec<String> = Vec::new();

        for seed in &seeds {
            if visited.insert(*seed) {
                queue.push_back((*seed, 0));
            }
        }

        while let Some((node_idx, depth)) = queue.pop_front() {
            let node = &self.graph[node_idx];

            for edge in self.graph.edges(node_idx) {
                let target = edge.target();
                let target_node = &self.graph[target];

                let line = match edge.weight() {
                    EdgeKind::Semantic(rel) => {
                        format!("{} --[{}]--> {}", node.name, rel, target_node.name)
                    }
                    EdgeKind::Causal(rel, conf) => format!(
                        "{} ==[{} ({:.0}%)]=> {}",
                        node.name,
                        rel,
                        conf * 100.0,
                        target_node.name
                    ),
                    EdgeKind::Temporal => {
                        format!("{} ~> {}", node.name, target_node.name)
                    }
                };
                context_parts.push(line);

                if depth < max_depth && visited.insert(target) {
                    queue.push_back((target, depth + 1));
                }
            }
        }

        context_parts.join("\n")
    }

    /// Get the causal chain forward from `entity` (BFS on causal edges only),
    /// up to `max_depth` hops.
    ///
    /// Returns a list of `(cause, relation, effect, confidence)` tuples.
    #[instrument(skip(self))]
    #[pyo3(signature = (entity, max_depth = 5))]
    pub fn get_causal_chain(
        &self,
        entity: &str,
        max_depth: usize,
    ) -> Vec<(String, String, String, f32)> {
        let start = match self.name_to_idx.get(entity) {
            Some(&idx) => idx,
            None => return Vec::new(),
        };

        let mut visited: HashSet<NodeIndex> = HashSet::new();
        let mut queue: VecDeque<(NodeIndex, usize)> = VecDeque::new();
        let mut chain: Vec<(String, String, String, f32)> = Vec::new();

        visited.insert(start);
        queue.push_back((start, 0));

        while let Some((node_idx, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }
            let node = &self.graph[node_idx];
            for edge in self.graph.edges(node_idx) {
                if let EdgeKind::Causal(rel, conf) = edge.weight() {
                    let target = edge.target();
                    let target_node = &self.graph[target];
                    chain.push((
                        node.name.clone(),
                        rel.clone(),
                        target_node.name.clone(),
                        *conf,
                    ));
                    if visited.insert(target) {
                        queue.push_back((target, depth + 1));
                    }
                }
            }
        }

        chain
    }

    /// List all entities, optionally filtered by `entity_type`.
    ///
    /// Returns a list of `(name, entity_type)` tuples.
    #[pyo3(signature = (entity_type = None))]
    pub fn get_entities(&self, entity_type: Option<&str>) -> Vec<(String, String)> {
        self.graph
            .node_weights()
            .filter(|n| entity_type.is_none() || entity_type == Some(n.entity_type.as_str()))
            .map(|n| (n.name.clone(), n.entity_type.clone()))
            .collect()
    }

    /// Number of entity nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges (relations) in the graph.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }
}

impl Default for RustEntityGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_entity() {
        let mut g = RustEntityGraph::new();
        assert!(g.add_entity("Alice", "person", None));
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn test_add_duplicate_entity() {
        let mut g = RustEntityGraph::new();
        assert!(g.add_entity("Alice", "person", None));
        // Second call with same name is idempotent and returns false.
        assert!(!g.add_entity("Alice", "person", None));
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn test_add_semantic_relation() {
        let mut g = RustEntityGraph::new();
        g.add_entity("Alice", "person", None);
        g.add_entity("Acme", "company", None);
        assert!(g.add_relation("Alice", "Acme", "works_at"));
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_add_semantic_relation_missing_entity() {
        let mut g = RustEntityGraph::new();
        g.add_entity("Alice", "person", None);
        // "Ghost" does not exist — relation should fail.
        assert!(!g.add_relation("Alice", "Ghost", "knows"));
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_add_causal_relation() {
        let mut g = RustEntityGraph::new();
        g.add_entity("Rain", "event", None);
        g.add_entity("Flood", "event", None);
        assert!(g.add_causal_relation("Rain", "Flood", "causes", 0.9));
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_add_causal_relation_default_confidence() {
        let mut g = RustEntityGraph::new();
        g.add_entity("X", "event", None);
        g.add_entity("Y", "event", None);
        // Default confidence = 1.0
        assert!(g.add_causal_relation("X", "Y", "leads_to", 1.0));
        let chain = g.get_causal_chain("X", 1);
        assert_eq!(chain.len(), 1);
        assert!((chain[0].3 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_get_context_for() {
        let mut g = RustEntityGraph::new();
        g.add_entity("Python", "language", None);
        g.add_entity("Rust", "language", None);
        g.add_relation("Python", "Rust", "transpiles_to");

        let ctx = g.get_context_for("I want to learn Python programming", 2);
        assert!(!ctx.is_empty(), "expected non-empty context for matching task");
        assert!(
            ctx.contains("Python"),
            "context should mention Python: {ctx}"
        );
    }

    #[test]
    fn test_context_empty_for_no_match() {
        let mut g = RustEntityGraph::new();
        g.add_entity("Mars", "planet", None);
        g.add_entity("Jupiter", "planet", None);
        g.add_relation("Mars", "Jupiter", "orbits_near");

        let ctx = g.get_context_for("write a Python sorting algorithm", 2);
        assert!(
            ctx.is_empty(),
            "expected empty context when no entities match: {ctx}"
        );
    }

    #[test]
    fn test_get_causal_chain() {
        let mut g = RustEntityGraph::new();
        g.add_entity("Spark", "event", None);
        g.add_entity("Fire", "event", None);
        g.add_entity("Smoke", "event", None);
        g.add_causal_relation("Spark", "Fire", "ignites", 0.95);
        g.add_causal_relation("Fire", "Smoke", "produces", 0.8);

        let chain = g.get_causal_chain("Spark", 5);
        assert_eq!(chain.len(), 2, "chain should have 2 steps: {chain:?}");

        let causes: Vec<&str> = chain.iter().map(|(c, _, _, _)| c.as_str()).collect();
        let effects: Vec<&str> = chain.iter().map(|(_, _, e, _)| e.as_str()).collect();
        assert!(causes.contains(&"Spark"));
        assert!(effects.contains(&"Fire"));
        assert!(causes.contains(&"Fire"));
        assert!(effects.contains(&"Smoke"));
    }

    #[test]
    fn test_get_causal_chain_max_depth() {
        let mut g = RustEntityGraph::new();
        // A → B → C → D (3 hops)
        for name in &["A", "B", "C", "D"] {
            g.add_entity(name, "node", None);
        }
        g.add_causal_relation("A", "B", "next", 1.0);
        g.add_causal_relation("B", "C", "next", 1.0);
        g.add_causal_relation("C", "D", "next", 1.0);

        // max_depth=2 should only follow A→B and B→C, not C→D
        let chain = g.get_causal_chain("A", 2);
        assert_eq!(chain.len(), 2, "depth 2 should yield 2 hops: {chain:?}");
    }

    #[test]
    fn test_get_entities_filtered() {
        let mut g = RustEntityGraph::new();
        g.add_entity("Alice", "person", None);
        g.add_entity("Bob", "person", None);
        g.add_entity("Acme", "company", None);

        let persons = g.get_entities(Some("person"));
        assert_eq!(persons.len(), 2);

        let companies = g.get_entities(Some("company"));
        assert_eq!(companies.len(), 1);
        assert_eq!(companies[0].0, "Acme");

        let all = g.get_entities(None);
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_mixed_edge_types() {
        let mut g = RustEntityGraph::new();
        g.add_entity("Climate", "phenomenon", None);
        g.add_entity("Storm", "event", None);
        g.add_entity("Damage", "event", None);

        // Semantic relation
        g.add_relation("Climate", "Storm", "produces");
        // Causal relation
        g.add_causal_relation("Storm", "Damage", "causes", 0.7);

        assert_eq!(g.edge_count(), 2);

        let chain = g.get_causal_chain("Storm", 3);
        assert_eq!(chain.len(), 1);
        assert_eq!(chain[0].0, "Storm");
        assert_eq!(chain[0].2, "Damage");
    }
}
