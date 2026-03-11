//! TopologyGraph: unified intermediate representation for multi-agent topologies.
//!
//! Three-flow edge model:
//! - **Control**: execution ordering (who runs after whom)
//! - **Message**: data flow (what output feeds into which input)
//! - **State**: shared state synchronization (memory/context propagation)

use petgraph::algo::{is_cyclic_directed, toposort};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use ulid::Ulid;

// ---------------------------------------------------------------------------
// Internal enums (not PyO3-exported)
// ---------------------------------------------------------------------------

/// The three edge flow types in a topology graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EdgeType {
    Control,
    Message,
    State,
}

impl EdgeType {
    /// Parse from string (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "control" => Some(Self::Control),
            "message" => Some(Self::Message),
            "state" => Some(Self::State),
            _ => None,
        }
    }

    /// Canonical string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Control => "control",
            Self::Message => "message",
            Self::State => "state",
        }
    }
}

/// Gate state for an edge — whether data flows or is blocked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Gate {
    Open,
    Closed,
}

impl Gate {
    /// Parse from string (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "open" => Some(Self::Open),
            "closed" => Some(Self::Closed),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Closed => "closed",
        }
    }
}

/// Topology template — the 8 built-in multi-agent patterns.
/// Custom(Ulid) is reserved for Phase 4.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TopologyTemplate {
    Sequential,
    Parallel,
    AVR,
    SelfMoA,
    Hierarchical,
    Hub,
    Debate,
    Brainstorming,
}

impl TopologyTemplate {
    /// Parse a template name (case-insensitive).
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "sequential" => Some(Self::Sequential),
            "parallel" => Some(Self::Parallel),
            "avr" => Some(Self::AVR),
            "selfmoa" | "self_moa" | "self-moa" => Some(Self::SelfMoA),
            "hierarchical" => Some(Self::Hierarchical),
            "hub" => Some(Self::Hub),
            "debate" => Some(Self::Debate),
            "brainstorming" => Some(Self::Brainstorming),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Sequential => "sequential",
            Self::Parallel => "parallel",
            Self::AVR => "avr",
            Self::SelfMoA => "selfmoa",
            Self::Hierarchical => "hierarchical",
            Self::Hub => "hub",
            Self::Debate => "debate",
            Self::Brainstorming => "brainstorming",
        }
    }
}

// ---------------------------------------------------------------------------
// TopologyNode
// ---------------------------------------------------------------------------

/// A node in the topology graph — represents one agent slot.
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyNode {
    #[pyo3(get)]
    pub node_id: String,
    /// Role name: "coder", "reviewer", "reasoner", etc.
    #[pyo3(get)]
    pub role: String,
    /// Model identifier: "gemini-2.5-flash", etc.
    #[pyo3(get)]
    pub model_id: String,
    /// Cognitive system tier: 1=S1, 2=S2, 3=S3.
    #[pyo3(get)]
    pub system: u8,
    /// Capabilities this node requires from its assigned model.
    #[pyo3(get)]
    pub required_capabilities: Vec<String>,
    /// Security label: 0=public, 1=internal, 2=confidential, 3=restricted.
    #[pyo3(get)]
    pub security_label: u8,
    /// Maximum cost budget in USD for this node.
    #[pyo3(get)]
    pub max_cost_usd: f32,
    /// Maximum wall-clock time in seconds for this node.
    #[pyo3(get)]
    pub max_wall_time_s: f32,
}

#[pymethods]
impl TopologyNode {
    #[new]
    #[pyo3(signature = (role, model_id, system=1, required_capabilities=vec![], security_label=0, max_cost_usd=1.0, max_wall_time_s=60.0))]
    pub fn py_new(
        role: String,
        model_id: String,
        system: u8,
        required_capabilities: Vec<String>,
        security_label: u8,
        max_cost_usd: f32,
        max_wall_time_s: f32,
    ) -> Self {
        Self::new(
            role,
            model_id,
            system,
            required_capabilities,
            security_label,
            max_cost_usd,
            max_wall_time_s,
        )
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }
}

impl std::fmt::Display for TopologyNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TopologyNode(role='{}', model='{}', S{}, label={}, budget=${:.2}, timeout={:.0}s)",
            self.role,
            self.model_id,
            self.system,
            self.security_label,
            self.max_cost_usd,
            self.max_wall_time_s
        )
    }
}

impl TopologyNode {
    /// Full constructor.
    pub fn new(
        role: String,
        model_id: String,
        system: u8,
        required_capabilities: Vec<String>,
        security_label: u8,
        max_cost_usd: f32,
        max_wall_time_s: f32,
    ) -> Self {
        Self {
            node_id: Ulid::new().to_string(),
            role,
            model_id,
            system,
            required_capabilities,
            security_label,
            max_cost_usd,
            max_wall_time_s,
        }
    }

    /// Deterministic constructor with explicit ID (for tests and internal use).
    pub fn with_id(node_id: String, role: String, model_id: String) -> Self {
        Self {
            node_id,
            role,
            model_id,
            system: 1,
            required_capabilities: Vec::new(),
            security_label: 0,
            max_cost_usd: 1.0,
            max_wall_time_s: 60.0,
        }
    }
}

// ---------------------------------------------------------------------------
// TopologyEdge
// ---------------------------------------------------------------------------

/// An edge in the topology graph — one of three flow types.
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologyEdge {
    /// Edge flow type: "control", "message", or "state".
    #[pyo3(get)]
    pub edge_type: String,
    /// Optional field-level routing for message edges.
    #[pyo3(get)]
    pub field_mapping: Option<HashMap<String, String>>,
    /// Gate state: "open" or "closed".
    #[pyo3(get)]
    pub gate: String,
    /// Optional condition expression (for switch/conditional edges).
    #[pyo3(get)]
    pub condition: Option<String>,
    /// Edge weight (default 1.0).
    #[pyo3(get)]
    pub weight: f32,

    // Internal typed fields (not exposed to Python directly).
    #[serde(skip)]
    edge_type_enum: Option<EdgeType>,
    #[serde(skip)]
    gate_enum: Option<Gate>,
}

#[pymethods]
impl TopologyEdge {
    #[new]
    #[pyo3(signature = (edge_type, field_mapping=None, gate="open".to_string(), condition=None, weight=1.0))]
    pub fn py_new(
        edge_type: String,
        field_mapping: Option<HashMap<String, String>>,
        gate: String,
        condition: Option<String>,
        weight: f32,
    ) -> PyResult<Self> {
        Self::try_new(edge_type, field_mapping, gate, condition, weight)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    fn __repr__(&self) -> String {
        format!(
            "TopologyEdge(type='{}', gate='{}', weight={:.1})",
            self.edge_type, self.gate, self.weight
        )
    }
}

impl TopologyEdge {
    /// Validated constructor (pure Rust, no Python dependency).
    pub fn try_new(
        edge_type: String,
        field_mapping: Option<HashMap<String, String>>,
        gate: String,
        condition: Option<String>,
        weight: f32,
    ) -> Result<Self, String> {
        let edge_type_enum = EdgeType::parse(&edge_type).ok_or_else(|| {
            format!(
                "Invalid edge type '{}'. Expected: control, message, state",
                edge_type
            )
        })?;
        let gate_enum = Gate::parse(&gate)
            .ok_or_else(|| format!("Invalid gate '{}'. Expected: open, closed", gate))?;
        Ok(Self {
            edge_type: edge_type_enum.as_str().to_string(),
            field_mapping,
            gate: gate_enum.as_str().to_string(),
            condition,
            weight,
            edge_type_enum: Some(edge_type_enum),
            gate_enum: Some(gate_enum),
        })
    }

    /// Get the typed EdgeType enum.
    pub fn typed_edge_type(&self) -> EdgeType {
        self.edge_type_enum
            .unwrap_or_else(|| EdgeType::parse(&self.edge_type).expect("valid edge_type"))
    }

    /// Get the typed Gate enum.
    pub fn typed_gate(&self) -> Gate {
        self.gate_enum
            .unwrap_or_else(|| Gate::parse(&self.gate).expect("valid gate"))
    }

    /// Convenience constructor: control edge.
    pub fn control() -> Self {
        Self {
            edge_type: "control".to_string(),
            field_mapping: None,
            gate: "open".to_string(),
            condition: None,
            weight: 1.0,
            edge_type_enum: Some(EdgeType::Control),
            gate_enum: Some(Gate::Open),
        }
    }

    /// Convenience: message edge with optional field mapping.
    pub fn message(field_mapping: Option<HashMap<String, String>>) -> Self {
        Self {
            edge_type: "message".to_string(),
            field_mapping,
            gate: "open".to_string(),
            condition: None,
            weight: 1.0,
            edge_type_enum: Some(EdgeType::Message),
            gate_enum: Some(Gate::Open),
        }
    }

    /// Convenience: state edge.
    pub fn state() -> Self {
        Self {
            edge_type: "state".to_string(),
            field_mapping: None,
            gate: "open".to_string(),
            condition: None,
            weight: 1.0,
            edge_type_enum: Some(EdgeType::State),
            gate_enum: Some(Gate::Open),
        }
    }

    /// Convenience: set gate on this edge (builder pattern).
    pub fn with_gate(mut self, gate: Gate) -> Self {
        self.gate = gate.as_str().to_string();
        self.gate_enum = Some(gate);
        self
    }

    /// Mutate the gate state in-place (for executor gate manipulation).
    pub fn set_gate(&mut self, gate: Gate) {
        self.gate = gate.as_str().to_string();
        self.gate_enum = Some(gate);
    }

    /// Convenience: set a condition on this edge (builder pattern).
    pub fn with_condition(mut self, condition: String) -> Self {
        self.condition = Some(condition);
        self
    }
}

// ---------------------------------------------------------------------------
// TopologyGraph
// ---------------------------------------------------------------------------

/// The unified topology intermediate representation.
/// Wraps a petgraph DiGraph with typed nodes and three-flow edges.
#[pyclass]
#[derive(Debug, Clone)]
pub struct TopologyGraph {
    graph: DiGraph<TopologyNode, TopologyEdge>,
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub template_type: String,
    template: TopologyTemplate,
}

#[pymethods]
impl TopologyGraph {
    /// Create an empty topology with the given template type (Python entry point).
    #[new]
    pub fn py_new(template_type: &str) -> PyResult<Self> {
        Self::try_new(template_type).map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    /// Add a node to the graph. Returns the node index.
    #[pyo3(name = "add_node")]
    pub fn py_add_node(&mut self, node: TopologyNode) -> usize {
        self.add_node(node)
    }

    /// Add an edge between two nodes (by index).
    #[pyo3(name = "add_edge")]
    pub fn py_add_edge(&mut self, from: usize, to: usize, edge: TopologyEdge) -> PyResult<()> {
        self.try_add_edge(from, to, edge)
            .map_err(|e| pyo3::exceptions::PyIndexError::new_err(e))
    }

    /// Number of nodes.
    #[pyo3(name = "node_count")]
    pub fn py_node_count(&self) -> usize {
        self.node_count()
    }

    /// Number of edges.
    #[pyo3(name = "edge_count")]
    pub fn py_edge_count(&self) -> usize {
        self.edge_count()
    }

    /// Check if the graph is acyclic (all edges considered).
    #[pyo3(name = "is_acyclic")]
    pub fn py_is_acyclic(&self) -> bool {
        self.is_acyclic()
    }

    /// Get all node IDs.
    #[pyo3(name = "node_ids")]
    pub fn py_node_ids(&self) -> Vec<String> {
        self.node_ids()
    }

    /// Get a node by index.
    #[pyo3(name = "get_node")]
    pub fn py_get_node(&self, index: usize) -> PyResult<TopologyNode> {
        self.try_get_node(index)
            .map_err(|e| pyo3::exceptions::PyIndexError::new_err(e))
    }

    /// Get topological ordering of nodes.
    /// Returns node indices in topological order. Errors if graph has cycles.
    #[pyo3(name = "topological_sort")]
    pub fn py_topological_sort(&self) -> PyResult<Vec<usize>> {
        self.try_topological_sort()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }
}

impl std::fmt::Display for TopologyGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TopologyGraph(id='{}', template='{}', nodes={}, edges={})",
            self.id,
            self.template_type,
            self.graph.node_count(),
            self.graph.edge_count()
        )
    }
}

// ---------------------------------------------------------------------------
// Pure Rust methods (no Python dependency)
// ---------------------------------------------------------------------------

impl TopologyGraph {
    /// Create an empty topology with the given template type.
    pub fn try_new(template_type: &str) -> Result<Self, String> {
        let template = TopologyTemplate::parse(template_type).ok_or_else(|| {
            format!(
                "Unknown topology template '{}'. Valid: sequential, parallel, avr, selfmoa, hierarchical, hub, debate, brainstorming",
                template_type
            )
        })?;
        Ok(Self {
            graph: DiGraph::new(),
            id: Ulid::new().to_string(),
            template_type: template.as_str().to_string(),
            template,
        })
    }

    /// Add a node to the graph. Returns the node index.
    pub fn add_node(&mut self, node: TopologyNode) -> usize {
        self.graph.add_node(node).index()
    }

    /// Add an edge between two nodes (by index). Returns error if indices are out of range.
    pub fn try_add_edge(
        &mut self,
        from: usize,
        to: usize,
        edge: TopologyEdge,
    ) -> Result<(), String> {
        let node_count = self.graph.node_count();
        if from >= node_count {
            return Err(format!(
                "Source node index {} out of range (graph has {} nodes)",
                from, node_count
            ));
        }
        if to >= node_count {
            return Err(format!(
                "Target node index {} out of range (graph has {} nodes)",
                to, node_count
            ));
        }
        self.graph
            .add_edge(NodeIndex::new(from), NodeIndex::new(to), edge);
        Ok(())
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Check if the graph is acyclic (all edges considered).
    pub fn is_acyclic(&self) -> bool {
        !is_cyclic_directed(&self.graph)
    }

    /// Get all node IDs.
    pub fn node_ids(&self) -> Vec<String> {
        self.graph
            .node_weights()
            .map(|n| n.node_id.clone())
            .collect()
    }

    /// Get a node by index. Returns error if index is out of range.
    pub fn try_get_node(&self, index: usize) -> Result<TopologyNode, String> {
        self.graph
            .node_weight(NodeIndex::new(index))
            .cloned()
            .ok_or_else(|| {
                format!(
                    "Node index {} out of range (graph has {} nodes)",
                    index,
                    self.graph.node_count()
                )
            })
    }

    /// Topological sort. Returns error if graph has cycles.
    pub fn try_topological_sort(&self) -> Result<Vec<usize>, String> {
        toposort(&self.graph, None)
            .map(|indices| indices.iter().map(|idx| idx.index()).collect())
            .map_err(|cycle| {
                let node = &self.graph[cycle.node_id()];
                format!(
                    "Graph has a cycle involving node '{}' (role='{}')",
                    node.node_id, node.role
                )
            })
    }

    /// Get the internal template enum.
    pub fn template(&self) -> TopologyTemplate {
        self.template
    }

    /// Check if graph has cycles (using petgraph is_cyclic_directed).
    pub fn has_cycles(&self) -> bool {
        is_cyclic_directed(&self.graph)
    }

    /// Get node indices that have no incoming edges (entry points).
    pub fn entry_nodes(&self) -> Vec<usize> {
        self.graph
            .node_indices()
            .filter(|&idx| {
                self.graph
                    .neighbors_directed(idx, petgraph::Direction::Incoming)
                    .next()
                    .is_none()
            })
            .map(|idx| idx.index())
            .collect()
    }

    /// Get node indices that have no outgoing edges (exit points).
    pub fn exit_nodes(&self) -> Vec<usize> {
        self.graph
            .node_indices()
            .filter(|&idx| {
                self.graph
                    .neighbors_directed(idx, petgraph::Direction::Outgoing)
                    .next()
                    .is_none()
            })
            .map(|idx| idx.index())
            .collect()
    }

    /// Get edges of a specific type. Returns (from_index, to_index, &edge).
    pub fn edges_of_type(&self, edge_type: EdgeType) -> Vec<(usize, usize, &TopologyEdge)> {
        self.graph
            .edge_references()
            .filter(|e| e.weight().typed_edge_type() == edge_type)
            .map(|e| (e.source().index(), e.target().index(), e.weight()))
            .collect()
    }

    /// Get a reference to the underlying petgraph.
    pub fn inner_graph(&self) -> &DiGraph<TopologyNode, TopologyEdge> {
        &self.graph
    }

    /// Mutable access to inner graph (for mutations).
    pub fn inner_graph_mut(&mut self) -> &mut DiGraph<TopologyNode, TopologyEdge> {
        &mut self.graph
    }

    /// Parse template name string to enum.
    pub fn parse_template(name: &str) -> Option<TopologyTemplate> {
        TopologyTemplate::parse(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_type_roundtrip() {
        for et in [EdgeType::Control, EdgeType::Message, EdgeType::State] {
            let s = et.as_str();
            assert_eq!(EdgeType::parse(s), Some(et));
        }
    }

    #[test]
    fn test_gate_roundtrip() {
        for g in [Gate::Open, Gate::Closed] {
            let s = g.as_str();
            assert_eq!(Gate::parse(s), Some(g));
        }
    }

    #[test]
    fn test_template_roundtrip() {
        let templates = [
            TopologyTemplate::Sequential,
            TopologyTemplate::Parallel,
            TopologyTemplate::AVR,
            TopologyTemplate::SelfMoA,
            TopologyTemplate::Hierarchical,
            TopologyTemplate::Hub,
            TopologyTemplate::Debate,
            TopologyTemplate::Brainstorming,
        ];
        for t in templates {
            let s = t.as_str();
            assert_eq!(TopologyTemplate::parse(s), Some(t));
        }
    }

    #[test]
    fn test_unknown_template() {
        assert_eq!(TopologyTemplate::parse("unknown"), None);
    }
}
