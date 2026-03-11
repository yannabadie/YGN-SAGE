//! MCTS (Monte Carlo Tree Search) topology searcher.
//!
//! Uses UCB1 selection, random mutation expansion, heuristic rollout scoring,
//! and backpropagation to search the topology space starting from a root graph.
//! No LLM calls — purely structural search using the mutation operators and verifier.

use std::time::Instant;

use rand::Rng;
use tracing::{debug, info_span};

use crate::topology::mutations::apply_random_mutation;
use crate::topology::topology_graph::TopologyGraph;
use crate::topology::verifier::HybridVerifier;

// ── MctsNode ────────────────────────────────────────────────────────────────

/// A node in the MCTS search tree.
struct MctsNode {
    topology: TopologyGraph,
    visit_count: u32,
    total_quality: f64,
    children: Vec<MctsNode>,
}

impl MctsNode {
    fn new(topology: TopologyGraph) -> Self {
        Self {
            topology,
            visit_count: 0,
            total_quality: 0.0,
            children: vec![],
        }
    }

    /// UCB1 selection score.
    fn ucb1(&self, parent_visits: u32, c: f64) -> f64 {
        if self.visit_count == 0 {
            return f64::INFINITY;
        }
        let exploit = self.total_quality / self.visit_count as f64;
        let explore = c * ((parent_visits as f64).ln() / self.visit_count as f64).sqrt();
        exploit + explore
    }

    /// Select child index with highest UCB1.
    fn best_child_ucb1(&self, c: f64) -> Option<usize> {
        if self.children.is_empty() {
            return None;
        }
        let pv = self.visit_count;
        self.children
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.ucb1(pv, c)
                    .partial_cmp(&b.ucb1(pv, c))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }

    /// Select child with highest average quality (for final selection).
    fn best_child_quality(&self) -> Option<&MctsNode> {
        self.children
            .iter()
            .filter(|c| c.visit_count > 0)
            .max_by(|a, b| {
                let qa = a.total_quality / a.visit_count as f64;
                let qb = b.total_quality / b.visit_count as f64;
                qa.partial_cmp(&qb).unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

// ── Rollout heuristic ───────────────────────────────────────────────────────

/// Score a topology using a fast heuristic (no LLM).
///
/// Base score from verifier validity + bonus for reasonable node count.
fn rollout_score(topology: &TopologyGraph) -> f64 {
    let verifier = HybridVerifier::new();
    let result = verifier.verify(topology);
    let base = if result.errors.is_empty() { 0.6 } else { 0.1 };
    let node_bonus = (topology.node_count() as f64).min(5.0) / 10.0;
    (base + node_bonus).min(1.0)
}

// ── MctsSearcher ────────────────────────────────────────────────────────────

/// Monte Carlo Tree Search over the topology mutation space.
///
/// Starting from a root topology, iteratively selects promising subtrees (UCB1),
/// expands via random mutation, scores with a heuristic rollout, and backpropagates.
pub struct MctsSearcher {
    max_simulations: u32,
    max_time_ms: u64,
    exploration_constant: f64,
}

impl MctsSearcher {
    /// Create a new MCTS searcher.
    ///
    /// # Arguments
    /// - `max_simulations`: Maximum number of simulation iterations.
    /// - `max_time_ms`: Maximum wall-clock time in milliseconds.
    pub fn new(max_simulations: u32, max_time_ms: u64) -> Self {
        Self {
            max_simulations,
            max_time_ms,
            exploration_constant: 1.41,
        }
    }

    /// Run MCTS search starting from root topology. Returns best topology found.
    pub fn search(&self, root: TopologyGraph) -> Option<TopologyGraph> {
        let _span = info_span!(
            "mcts.search",
            max_simulations = self.max_simulations,
            max_time_ms = self.max_time_ms,
        )
        .entered();

        let start = Instant::now();
        let mut tree = MctsNode::new(root);
        let mut rng = rand::rng();
        let mut completed_sims = 0u32;

        for _ in 0..self.max_simulations {
            if start.elapsed().as_millis() as u64 > self.max_time_ms {
                break;
            }

            // 1. SELECT: walk down tree following best UCB1 child until
            //    reaching a node with fewer than 3 children (expansion candidate).
            let path = self.select(&tree);

            // 2. EXPAND: at the selected leaf, mutate to create a new child and score it.
            let leaf_quality = self.expand_and_rollout(&mut tree, &path, &mut rng);

            // 3. BACKPROPAGATE: update visit counts and qualities up the path.
            self.backpropagate(&mut tree, &path, leaf_quality);

            completed_sims += 1;
        }

        debug!(
            completed_sims = completed_sims,
            elapsed_ms = start.elapsed().as_millis() as u64,
            children = tree.children.len(),
            "mcts_search_complete"
        );

        tree.best_child_quality().map(|c| c.topology.clone())
    }

    /// SELECT phase: walk down the tree via UCB1 until finding an expandable node
    /// (one with fewer than 3 children). Returns the path of child indices.
    fn select(&self, tree: &MctsNode) -> Vec<usize> {
        let mut path = Vec::new();
        let mut current = tree;

        loop {
            // If this node has fewer than 3 children, it's expandable — stop here.
            if current.children.len() < 3 {
                break;
            }

            // Otherwise, follow the best UCB1 child.
            match current.best_child_ucb1(self.exploration_constant) {
                Some(idx) => {
                    path.push(idx);
                    current = &current.children[idx];
                }
                None => break,
            }
        }

        path
    }

    /// EXPAND + ROLLOUT: navigate to the node at `path`, apply a random mutation
    /// to create a new child, score it with the rollout heuristic.
    fn expand_and_rollout<R: Rng>(
        &self,
        tree: &mut MctsNode,
        path: &[usize],
        rng: &mut R,
    ) -> f64 {
        // Navigate to the leaf node.
        let mut current = tree;
        for &idx in path {
            current = &mut current.children[idx];
        }

        // Apply a random mutation to generate a new child topology.
        let parent_topology = current.topology.clone();
        let mutated = apply_random_mutation(parent_topology, rng);

        match mutated {
            crate::topology::mutations::MutationResult::Success(new_topology) => {
                let score = rollout_score(&new_topology);
                current.children.push(MctsNode::new(new_topology));
                score
            }
            crate::topology::mutations::MutationResult::Invalid(_reason) => {
                // Mutation failed — still count as a visit with low score.
                // Don't add a child node for invalid mutations.
                0.1
            }
        }
    }

    /// BACKPROPAGATE: walk back up the path, incrementing visit_count and
    /// adding quality to total_quality at each node.
    fn backpropagate(&self, tree: &mut MctsNode, path: &[usize], quality: f64) {
        // Update root.
        tree.visit_count += 1;
        tree.total_quality += quality;

        // Update each node along the path.
        let mut current = tree;
        for &idx in path {
            current = &mut current.children[idx];
            current.visit_count += 1;
            current.total_quality += quality;
        }
    }
}

// ── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::templates;

    #[test]
    fn test_mcts_node_ucb1_unvisited() {
        let g = templates::sequential("gemini-2.5-flash");
        let node = MctsNode::new(g);
        assert_eq!(node.ucb1(100, 1.41), f64::INFINITY);
    }

    #[test]
    fn test_mcts_node_ucb1_visited() {
        let g = templates::sequential("gemini-2.5-flash");
        let mut node = MctsNode::new(g);
        node.visit_count = 10;
        node.total_quality = 7.0;
        let ucb = node.ucb1(100, 1.41);
        // exploit = 0.7, explore = 1.41 * sqrt(ln(100)/10) ≈ 1.41 * 0.678 ≈ 0.956
        // total ≈ 1.656
        assert!(ucb > 0.7 && ucb < 2.5, "UCB1 was {}", ucb);
    }

    #[test]
    fn test_mcts_search_returns_topology() {
        let root = templates::sequential("gemini-2.5-flash");
        let searcher = MctsSearcher::new(20, 500);
        let result = searcher.search(root);
        assert!(result.is_some());
    }

    #[test]
    fn test_mcts_search_time_bounded() {
        let root = templates::parallel("gemini-2.5-flash", 3);
        let searcher = MctsSearcher::new(10000, 50); // high sim count but low time
        let start = Instant::now();
        let _result = searcher.search(root);
        assert!(
            start.elapsed().as_millis() < 200,
            "Search took {}ms, should stop well before 200ms",
            start.elapsed().as_millis()
        );
    }

    #[test]
    fn test_mcts_search_empty_graph() {
        let g = TopologyGraph::try_new("sequential").unwrap();
        let searcher = MctsSearcher::new(10, 100);
        // Empty graph may or may not produce result - just shouldn't panic
        let _ = searcher.search(g);
    }

    #[test]
    fn test_rollout_score_valid_topology() {
        let g = templates::sequential("gemini-2.5-flash");
        let score = rollout_score(&g);
        // Valid 3-node graph: base 0.6 + min(3, 5)/10 = 0.6 + 0.3 = 0.9
        assert!(
            (score - 0.9).abs() < 0.01,
            "Expected ~0.9, got {}",
            score
        );
    }

    #[test]
    fn test_rollout_score_empty_graph() {
        let g = TopologyGraph::try_new("sequential").unwrap();
        let score = rollout_score(&g);
        // Empty graph is valid (verifier passes), base 0.6 + 0/10 = 0.6
        assert!(
            (score - 0.6).abs() < 0.01,
            "Expected ~0.6, got {}",
            score
        );
    }

    #[test]
    fn test_mcts_best_child_quality_empty() {
        let g = templates::sequential("gemini-2.5-flash");
        let node = MctsNode::new(g);
        assert!(node.best_child_quality().is_none());
    }

    #[test]
    fn test_mcts_best_child_ucb1_empty() {
        let g = templates::sequential("gemini-2.5-flash");
        let node = MctsNode::new(g);
        assert!(node.best_child_ucb1(1.41).is_none());
    }
}
