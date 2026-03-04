import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../sage-python')))
from notebooklm import NotebookLMClient

DISCOVER_AI_NB_ID = "dcf45958-35bc-4f37-bee7-52b08571d2e2"
METASCAFFOLD_NB_ID = "097c4c5c-beb2-4f65-9c6d-f597926a4232"

async def query_notebook(nb_id, name, query):
    try:
        async with await NotebookLMClient.from_storage() as client:
            print(f"\n🧠 Querying '{name}' (ID: {nb_id})")
            result = await client.chat.ask(nb_id, query)
            print("--- RESPONSE ---")
            print(result.answer)
            print("----------------\n")
    except Exception as e:
        print(f"Error querying {name}: {e}")

async def main():
    q1 = "How can an AI agent elevate its metacognition to discern its own lies and simulations? Specifically, how do MetaScaffold_Core and advanced frameworks transition an agent from outputting 'mock' data or simulations into interacting with real-world financial levers, deploying real SaaS, or executing real bounties autonomously?"
    await query_notebook(METASCAFFOLD_NB_ID, "MetaScaffold_Core", q1)
    await query_notebook(DISCOVER_AI_NB_ID, "Discover AI", q1)

if __name__ == "__main__":
    asyncio.run(main())
