"""Pre-built topology patterns for common agent orchestration scenarios."""
from __future__ import annotations

from sage.topology.engine import TopologyEngine, TopologyType


def vertical(engine: TopologyEngine, coordinator_name: str, worker_names: list[str]) -> str:
    """Create a vertical topology: one coordinator delegates to sequential workers.

    Returns the topology ID.
    """
    topo_id = engine.create_topology(TopologyType.VERTICAL)
    coord_id = engine.add_node(topo_id, name=coordinator_name, role="coordinator")

    for name in worker_names:
        worker_id = engine.add_node(topo_id, name=name, role="worker")
        engine.connect(topo_id, coord_id, worker_id, "delegates_to")

    return topo_id


def horizontal(engine: TopologyEngine, names: list[str]) -> str:
    """Create a horizontal topology: parallel agents, all peers.

    Returns the topology ID.
    """
    topo_id = engine.create_topology(TopologyType.HORIZONTAL)
    node_ids = []

    for name in names:
        nid = engine.add_node(topo_id, name=name, role="explorer")
        node_ids.append(nid)

    # Connect all as peers
    for i, nid in enumerate(node_ids):
        for j in range(i + 1, len(node_ids)):
            engine.connect(topo_id, nid, node_ids[j], "peers_with")

    return topo_id


def mesh(engine: TopologyEngine, coordinator_name: str, agent_names: list[str]) -> str:
    """Create a mesh topology: coordinator + fully connected agents.

    Returns the topology ID.
    """
    topo_id = engine.create_topology(TopologyType.MESH)
    coord_id = engine.add_node(topo_id, name=coordinator_name, role="coordinator")

    agent_ids = []
    for name in agent_names:
        aid = engine.add_node(topo_id, name=name, role="agent")
        engine.connect(topo_id, coord_id, aid, "delegates_to")
        agent_ids.append(aid)

    # Fully connect agents as peers
    for i, aid in enumerate(agent_ids):
        for j in range(i + 1, len(agent_ids)):
            engine.connect(topo_id, aid, agent_ids[j], "peers_with")

    return topo_id
