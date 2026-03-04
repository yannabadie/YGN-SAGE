import pytest
import asyncio
from unittest.mock import Mock

from sage.topology.engine import TopologyEngine, TopologyType
from sage.topology.planner import TopologyPlanner, StochasticDTS

@pytest.mark.asyncio
async def test_topology_planner_generate():
    engine = TopologyEngine()
    mock_llm = Mock()
    mock_llm.generate = Mock()
    
    async def mock_generate(*args, **kwargs):
        class MockResponse:
            content = """
            {
                "type": "vertical",
                "nodes": [
                    {"id": "a1", "name": "Planner", "role": "coordinator"},
                    {"id": "a2", "name": "Coder", "role": "worker"}
                ],
                "edges": [
                    {"source": "a1", "target": "a2", "type": "delegates_to"}
                ]
            }
            """
        return MockResponse()
        
    mock_llm.generate.side_effect = mock_generate
    
    planner = TopologyPlanner(engine, mock_llm)
    topo_id = await planner.generate_topology("Build a web server")
    
    topo = engine.get_topology(topo_id)
    assert topo is not None
    assert topo.type == TopologyType.VERTICAL
    assert topo.node_count() == 2
    
    edges = topo.all_edges()
    assert len(edges) == 1
    assert edges[0].edge_type == "delegates_to"

def test_sdts_backpropagate_and_select():
    sdts = StochasticDTS()
    task = "test task"
    
    # Select should be None initially
    assert sdts.select_node(task) is None
    
    # Backpropagate a new topology
    topo_def_1 = '{"type": "vertical"}'
    sdts.backpropagate(task, topo_def_1, reward=1.0)
    
    # Select should now return topo_def_1
    assert sdts.select_node(task) == topo_def_1
    
    # Backpropagate a better topology
    topo_def_2 = '{"type": "horizontal"}'
    sdts.backpropagate(task, topo_def_2, reward=5.0)
    
    # Select should now favor topo_def_2 due to higher reward
    assert sdts.select_node(task) == topo_def_2
