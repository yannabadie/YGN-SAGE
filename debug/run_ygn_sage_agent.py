"""Debug script: Bootstrap YGN-SAGE Prime agent for autonomous response.

Usage: python debug/run_ygn_sage_agent.py
Requires: GOOGLE_API_KEY environment variable.
"""
import asyncio
import os
import sys

# Ensure sage-python/src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../sage-python/src')))

from sage.agent import Agent, AgentConfig
from sage.llm.google import GoogleProvider
from sage.llm.base import LLMConfig
from sage.tools.registry import ToolRegistry
from sage.tools.meta import create_python_tool, create_bash_tool
from sage.tools.agent_mgmt import create_agent, call_agent, list_active_agents


async def main():
    print("===================================================================")
    print(" BOOTSTRAPPING YGN-SAGE CORE AGENT FOR AUTONOMOUS RESPONSE")
    print("===================================================================")

    provider = GoogleProvider()

    # 1. Setup the Tool Registry with OpenSAGE Meta-Tools
    registry = ToolRegistry()
    registry.register(create_python_tool)
    registry.register(create_bash_tool)
    registry.register(create_agent)
    registry.register(call_agent)
    registry.register(list_active_agents)

    # 2. Configure the YGN-SAGE Lead Agent with System 3 (formal verification)
    config = AgentConfig(
        name="YGN-SAGE_Prime",
        llm=LLMConfig(provider="google", model="gemini-3.1-pro-preview"),
        system_prompt="""You are YGN-SAGE Prime, an advanced AI assistant (March 2026).
Your architecture includes OpenSAGE Topology Planning, Z3-backed System 3 Verification, and eBPF <1ms Sandboxing.
Use <think> tags for structured reasoning. Be precise and concise.""",
        validation_level=3,
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
