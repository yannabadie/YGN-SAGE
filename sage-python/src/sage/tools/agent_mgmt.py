"""Sub-agent management meta-tools for dynamic topology generation."""
from __future__ import annotations

import json
from sage.tools.base import Tool

# SOTA OpenSage: Dynamic Sub-Agent creation and delegation
@Tool.define(
    name="create_agent",
    description="Creates a new sub-agent dynamically. The sub-agent is stored in the unified sub-agent pool and can be invoked later using call_agent.",
    parameters={
        "type": "object",
        "properties": {
            "agent_name": {"type": "string", "description": "Unique identifier for this new sub-agent."},
            "role": {"type": "string", "description": "The role or persona of the sub-agent."},
            "instruction": {"type": "string", "description": "The system prompt instructing the sub-agent what to do."},
            "tools": {
                "type": "array", 
                "items": {"type": "string"},
                "description": "List of tool names the sub-agent is allowed to use. e.g., ['read_file', 'run_shell']"
            }
        },
        "required": ["agent_name", "role", "instruction"]
    }
)
async def create_agent(agent_name: str, role: str, instruction: str, tools: list[str] = None, agent_pool: dict = None, parent_agent: 'Agent' = None) -> str:
    if agent_pool is None or parent_agent is None:
        return "Error: Agent pool or parent agent context not available."
        
    if agent_name in agent_pool:
        return f"Error: Agent '{agent_name}' already exists."
        
    # Local import to prevent circular dependency
    from sage.agent import Agent, AgentConfig
    
    # Create configuration
    config = AgentConfig(
        name=agent_name,
        llm=parent_agent.config.llm,  # Inherit model config
        system_prompt=f"Role: {role}\n\nInstructions:\n{instruction}",
        tools=tools,
        use_docker_sandbox=parent_agent.config.use_docker_sandbox,
        enforce_system3=parent_agent.config.enforce_system3
    )
    
    # Create agent instance
    sub_agent = Agent(
        config=config,
        llm_provider=parent_agent._llm,
        tool_registry=parent_agent._tools,
        memory_compressor=parent_agent.memory_compressor,
        sandbox_manager=parent_agent.sandbox_manager
    )
    
    # Register in the pool
    agent_pool[agent_name] = sub_agent
    
    # In a real graph memory, we'd add an edge here
    parent_agent.working_memory.add_child_agent(agent_name)
    
    return f"Success: Sub-agent '{agent_name}' created. You can invoke it using call_agent."


@Tool.define(
    name="call_agent",
    description="Invokes an existing sub-agent to execute a specific task.",
    parameters={
        "type": "object",
        "properties": {
            "agent_name": {"type": "string", "description": "Name of the sub-agent to invoke."},
            "task_message": {"type": "string", "description": "The specific task or prompt for the sub-agent to execute."}
        },
        "required": ["agent_name", "task_message"]
    }
)
async def call_agent(agent_name: str, task_message: str, agent_pool: dict = None) -> str:
    if agent_pool is None:
        return "Error: Agent pool not available."
        
    sub_agent = agent_pool.get(agent_name)
    if not sub_agent:
        return f"Error: Agent '{agent_name}' not found. Use create_agent first."
        
    try:
        # Run the sub-agent
        result = await sub_agent.run(task_message)
        
        # Format the response
        response = {
            "agent": agent_name,
            "status": "success",
            "result": result
        }
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error executing sub-agent '{agent_name}': {str(e)}"


@Tool.define(
    name="list_active_agents",
    description="Lists all currently active sub-agents in the pool.",
    parameters={
        "type": "object",
        "properties": {}
    }
)
async def list_active_agents(agent_pool: dict = None) -> str:
    if agent_pool is None:
        return "Error: Agent pool not available."
        
    if not agent_pool:
        return "Found 0 total agents. If no suitable agents exist, create a dynamic sub-agent using create_agent."
        
    agents = []
    for name, agent in agent_pool.items():
        agents.append({
            "name": name,
            "tools": agent.config.tools or "ALL",
            "steps_taken": agent.step_count
        })
        
    return json.dumps({"active_agents": agents}, indent=2)
