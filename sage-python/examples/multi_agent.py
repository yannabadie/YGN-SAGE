"""Multi-agent composition: sequential pipeline with specialist agents."""

import asyncio
from sage import create, Agent, AgentConfig
from sage.agents import SequentialAgent


async def main():
    system = await create()

    # Create specialist agents using Agent constructor directly
    researcher = Agent(AgentConfig(
        name="researcher",
        llm=system.agent_loop.config.llm,
        system_prompt="You are a research analyst. Summarize key findings.",
    ), llm_provider=system.agent_loop._llm)

    writer = Agent(AgentConfig(
        name="writer",
        llm=system.agent_loop.config.llm,
        system_prompt="You write clear technical documentation.",
    ), llm_provider=system.agent_loop._llm)

    # Chain them: research → write
    pipeline = SequentialAgent("doc-pipeline", [researcher, writer])
    result = await pipeline.run("Explain how transformers work in 500 words")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
