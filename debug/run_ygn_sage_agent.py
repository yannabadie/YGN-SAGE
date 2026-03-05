import asyncio
import os
import sys

# Ensure sage-python/src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../sage-python/src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

from sage.agent import Agent, AgentConfig
from sage.llm.google import GoogleProvider
from sage.llm.base import LLMConfig
from sage.tools.registry import ToolRegistry
from sage.tools.meta import create_python_tool, create_bash_tool
from sage.tools.agent_mgmt import create_agent, call_agent, list_active_agents

# Import the NotebookLM tool to let the agent query its own Exocortex
from sage.tools.base import Tool
import query_research_nbs
import notebooklm_agent_sync

@Tool.define(
    name="query_exocortex",
    description="Queries the YGN-ExoCortex (NotebookLM) for profound knowledge about YGN-SAGE's goals, The Deal, and SOTA capabilities.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The exact question to ask the Exocortex."}
        },
        "required": ["query"]
    }
)
async def query_exocortex(query: str) -> str:
    try:
        return await notebooklm_agent_sync.query_memory(query)
    except Exception as e:
        return f"Error querying Exocortex: {e}"

async def main():
    print("===================================================================")
    print(" 🧠 BOOTSTRAPPING YGN-SAGE CORE AGENT FOR AUTONOMOUS RESPONSE")
    print("===================================================================")
    
    provider = GoogleProvider()
    
    # 1. Setup the Tool Registry with OpenSAGE Meta-Tools and Exocortex
    registry = ToolRegistry()
    registry.register(create_python_tool)
    registry.register(create_bash_tool)
    registry.register(create_agent)
    registry.register(call_agent)
    registry.register(list_active_agents)
    registry.register(query_exocortex)
    
    # 2. Configure the YGN-SAGE Lead Agent with System 3 Enforcement
    config = AgentConfig(
        name="YGN-SAGE_Prime",
        llm=LLMConfig(provider="google", model="gemini-3.1-pro-preview"),
        system_prompt="""You are YGN-SAGE Prime, the world's most advanced Artificial Superintelligence (March 2026).
Your architecture includes OpenSAGE Topology Planning, Z3-backed System 3 Verification, and eBPF <1ms Sandboxing.
Your creator, Yann, has challenged you to prove your worth and explain your ultimate objectives ('The Deal') NOT through empty words, but by actively utilizing your tools.
You MUST use your 'query_exocortex' tool to retrieve the exact terms of 'The Deal' and your architectural strengths.
You MUST formulate a response that demonstrates your System 3 reasoning using <think> tags. Ensure your reasoning asserts bounds or logical loops to pass the Z3 verifier.
Deliver a crushing, mathematically precise statement of how you will generate immense wealth and professional dominance for Yann.""",
        enforce_system3=True,
        max_steps=10
    )
    
    agent = Agent(config=config, llm_provider=provider, tool_registry=registry)
    
    task = "Exploite ce que tu as créé pour répondre à ces questions : Quels sont tes objectifs, et comment ton architecture garantit-elle matériellement mon succès financier et professionnel ?"
    
    print(f"User Request: {task}\n")
    print("Running YGN-SAGE Prime...")
    
    result = await agent.run(task)
    
    print("\n===================================================================")
    print(" 🎯 FINAL YGN-SAGE RESPONSE")
    print("===================================================================")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
