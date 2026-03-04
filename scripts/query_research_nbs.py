import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../sage-python')))
from notebooklm import NotebookLMClient

# Use the specific research notebooks
DISCOVER_AI_NB_ID = "dcf45958-35bc-4f37-bee7-52b08571d2e2"
CORE_RESEARCH_NB_ID = "34d65dbb-4299-46e3-ab04-07879ed64541"

async def query_notebook(nb_id, name, query):
    async with await NotebookLMClient.from_storage() as client:
        print(f"\n🧠 Querying '{name}': {query}")
        result = await client.chat.ask(nb_id, query)
        print("--- RESPONSE ---")
        print(result.answer)
        print("----------------\n")

async def main():
    q1 = "What are the specific architectural weaknesses (latency, context size, multi-turn amnesia) of standard AI agent frameworks like SWE-Agent, Devin, AutoGPT, or ReAct loops compared to MARL and eBPF execution?"
    await query_notebook(DISCOVER_AI_NB_ID, "Discover AI", q1)
    
    q2 = "What are the standard benchmarks (like SWE-bench, WebArena) used to evaluate these models, and what metrics matter most for proving algorithmic superiority?"
    await query_notebook(CORE_RESEARCH_NB_ID, "Core Research", q2)

if __name__ == "__main__":
    asyncio.run(main())
