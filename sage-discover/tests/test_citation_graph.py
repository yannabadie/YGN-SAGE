"""tests/test_citation_graph.py — Citation graph tests."""
from __future__ import annotations

import pytest

from discover.citation_graph import CitationGraphBuilder


@pytest.fixture
def builder():
    return CitationGraphBuilder()


def test_add_paper_node(builder):
    builder.add_paper("p1", title="Paper A", year=2025, citation_count=10)
    assert "p1" in builder.graph
    assert builder.graph.nodes["p1"]["title"] == "Paper A"


def test_add_citation_edge(builder):
    builder.add_paper("p1", title="A", year=2025, citation_count=0)
    builder.add_paper("p2", title="B", year=2025, citation_count=0)
    builder.add_citation("p1", "p2")
    assert builder.graph.has_edge("p1", "p2")


def test_pagerank(builder):
    builder.add_paper("p1", title="A", year=2025, citation_count=0)
    builder.add_paper("p2", title="B", year=2025, citation_count=0)
    builder.add_paper("p3", title="C", year=2025, citation_count=0)
    builder.add_citation("p1", "p3")
    builder.add_citation("p2", "p3")
    ranks = builder.pagerank()
    assert ranks["p3"] > ranks["p1"]


def test_communities(builder):
    for i in range(5):
        builder.add_paper(f"a{i}", title=f"A{i}", year=2025, citation_count=0)
    for i in range(5):
        builder.add_paper(f"b{i}", title=f"B{i}", year=2025, citation_count=0)
    for i in range(4):
        builder.add_citation(f"a{i}", f"a{i+1}")
        builder.add_citation(f"b{i}", f"b{i+1}")
    comms = builder.communities()
    assert len(comms) >= 2


def test_bridges(builder):
    builder.add_paper("p1", title="A", year=2025, citation_count=0)
    builder.add_paper("bridge", title="Bridge", year=2025, citation_count=0)
    builder.add_paper("p2", title="C", year=2025, citation_count=0)
    builder.add_citation("p1", "bridge")
    builder.add_citation("bridge", "p2")
    bridges = builder.bridges()
    assert bridges["bridge"] >= bridges.get("p1", 0)


def test_node_count(builder):
    assert builder.node_count() == 0
    builder.add_paper("p1", title="A", year=2025, citation_count=0)
    assert builder.node_count() == 1
