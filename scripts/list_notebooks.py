import asyncio
from notebooklm import NotebookLMClient

async def list_notebooks():
    async with await NotebookLMClient.from_storage() as client:
        notebooks = await client.notebooks.list()
        print("\n--- Available Notebooks ---")
        for nb in notebooks:
            print(f"ID: {nb.id} | Title: {nb.title}")
        print("---------------------------\n")

if __name__ == "__main__":
    asyncio.run(list_notebooks())
