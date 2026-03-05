import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../sage-python/src')))
from sage.agent import Agent, AgentConfig
from sage.llm.google import GoogleProvider
from sage.llm.base import LLMConfig
from sage.tools.registry import ToolRegistry
from sage.tools.base import Tool

# We create local proxy tools that simulate Claude interacting with our MCP Gateway via the protocol.
@Tool.define(
    name="mcp_z3_verify_sql",
    description="Proxy to the YGN-SAGE MCP Server: Formally verifies SQL before execution.",
    parameters={
        "type": "object",
        "properties": {
            "table": {"type": "string"},
            "where_clause": {"type": "string"},
            "update_values": {"type": "object"}
        },
        "required": ["table", "where_clause", "update_values"]
    }
)
async def mcp_z3_verify_sql(table: str, where_clause: str, update_values: dict) -> str:
    # Simulating the network call to the MCP gateway
    import sage_discover.mcp_gateway as gw
    res = gw.z3_verify_sql_update(table, where_clause, update_values)
    return str(res)

@Tool.define(
    name="mcp_optimize_mes",
    description="Proxy to the YGN-SAGE MCP Server: Compiles constraints to Z3, verifies, and runs eBPF optimization.",
    parameters={
        "type": "object",
        "properties": {
            "objective": {"type": "string"},
            "constraints": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["objective", "constraints"]
    }
)
async def mcp_optimize_mes(objective: str, constraints: list) -> str:
    # Simulating the network call to the MCP gateway
    import sage_discover.mcp_gateway as gw
    res = await gw.optimize_mes_schedule(objective, constraints)
    return str(res)

async def run_b2b_demo():
    print("\n=======================================================")
    print(" 🤖 CLIENT DEMO: Claude-Enterprise via YGN-SAGE MCP")
    print("=======================================================")
    
    provider = GoogleProvider()
    registry = ToolRegistry()
    registry.register(mcp_z3_verify_sql)
    registry.register(mcp_optimize_mes)
    
    config = AgentConfig(
        name="Claude_Enterprise_Proxy",
        llm=LLMConfig(provider="google", model="gemini-3.1-flash-lite-preview"),
        system_prompt="You are an Enterprise AI Assistant. You use the YGN-SAGE MCP tools to guarantee the safety and speed of your actions. Always explain your reasoning using <think> tags.",
        enforce_system3=True
    )
    
    agent = Agent(config=config, llm_provider=provider, tool_registry=registry)
    
    # 1. Test the ERP SQL Verification
    print("\n--- SCENARIO 1: ERP Database Update ---")
    task1 = "A user asked to update the price of item 'A12' in the 'products' table to 99.99. Use the MCP tool to verify this SQL update before committing."
    print(f"User: {task1}")
    res1 = await agent.run(task1)
    print(f"Agent:\n{res1}\n")
    
    # 2. Test the MES Optimization
    print("\n--- SCENARIO 2: Factory MES Optimization ---")
    agent.step_count = 0 # reset
    task2 = "Machine CNC-04 just broke down. We must re-route production. The objective is MAXIMIZE_THROUGHPUT. The hard constraints are 'CNC-04_STATUS=OFFLINE' and 'ORDER_77_DEADLINE=TODAY'. Use the MCP tool to generate an optimal execution plan."
    print(f"User: {task2}")
    res2 = await agent.run(task2)
    print(f"Agent:\n{res2}\n")

if __name__ == "__main__":
    asyncio.run(run_b2b_demo())