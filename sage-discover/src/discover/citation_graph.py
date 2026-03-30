"""src/discover/citation_graph.py — Citation graph via NetworkX."""
from __future__ import annotations

import logging
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


class CitationGraphBuilder:
    """Builds and analyzes a citation graph using NetworkX DiGraph."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_paper(self, paper_id: str, **attrs: Any) -> None:
        self.graph.add_node(paper_id, **attrs)

    def add_citation(self, citing: str, cited: str) -> None:
        if citing not in self.graph:
            self.graph.add_node(citing)
        if cited not in self.graph:
            self.graph.add_node(cited)
        self.graph.add_edge(citing, cited, relation="cites")

    def pagerank(self, alpha: float = 0.85) -> dict[str, float]:
        if self.graph.number_of_nodes() == 0:
            return {}
        return nx.pagerank(self.graph, alpha=alpha)

    def communities(self) -> list[set[str]]:
        if self.graph.number_of_nodes() < 2:
            return [set(self.graph.nodes)]
        undirected = self.graph.to_undirected()
        return list(nx.community.louvain_communities(undirected))

    def bridges(self) -> dict[str, float]:
        if self.graph.number_of_nodes() < 2:
            return {n: 0.0 for n in self.graph.nodes}
        return nx.betweenness_centrality(self.graph)

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    def neighbors(self, paper_id: str, direction: str = "both") -> list[str]:
        result = []
        if direction in ("citing", "both"):
            result.extend(self.graph.predecessors(paper_id))
        if direction in ("cited", "both"):
            result.extend(self.graph.successors(paper_id))
        return list(set(result))
