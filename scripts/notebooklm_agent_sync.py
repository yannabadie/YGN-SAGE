import asyncio
import os
import sys
import datetime
import argparse
from pathlib import Path

# Fix path to include sage-python if needed, though notebooklm-py should be installed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../sage-python')))

try:
    from notebooklm import NotebookLMClient
except ImportError:
    print("⚠️ notebooklm-py not found. Please install it: pip install notebooklm-py")
    sys.exit(1)

# --- CONFIGURATION ---
DEFAULT_MEMORY_NOTEBOOK_ID = "7ab1d708-b7ab-46dc-b076-8c2daf8ba3ea"
NOTEBOOK_TITLE = "YGN-ExoCortex"

async def get_memory_notebook(client):
    """Finds or creates the dedicated memory notebook."""
    notebooks = await client.notebooks.list()
    for nb in notebooks:
        if nb.title == NOTEBOOK_TITLE:
            return nb
    print(f"✨ Creating new memory notebook: {NOTEBOOK_TITLE}")
    return await client.notebooks.create(NOTEBOOK_TITLE)

async def sync_thought(thought: str, notebook_id: str = None):
    """Uploads a new thought/reflection as a text source to NotebookLM."""
    async with await NotebookLMClient.from_storage() as client:
        if not notebook_id:
            nb = await get_memory_notebook(client)
            notebook_id = nb.id
        
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        title = f"Reflection_{now}"
        
        print(f"📡 Uploading reflection to NotebookLM ({notebook_id})...")
        await client.sources.add_text(notebook_id, title, thought)
        
        # Local backup
        backup_path = Path("research_journal/agent_long_term_memory.md")
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        with open(backup_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n--- {now} ---\n{thought}\n")
            
        print(f"✅ Thought synchronized: {title}")

async def query_memory(query: str, notebook_id: str = None):
    """Queries the exocortex for past information."""
    async with await NotebookLMClient.from_storage() as client:
        if not notebook_id:
            nb = await get_memory_notebook(client)
            notebook_id = nb.id
            
        print(f"🧠 Querying Exocortex: '{query}'...")
        result = await client.chat.ask(notebook_id, query)
        print("\n--- EXOCORTEX RESPONSE ---")
        print(result.answer)
        print("--------------------------\n")
        return result.answer

async def bootstrap_session():
    """Initializes the session by retrieving the last state from NotebookLM."""
    prompt = "Summarize my last session, including unfinished tasks, current hypotheses, and the overall goal of the project."
    return await query_memory(prompt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YGN-SAGE Exocortex Sync Tool")
    subparsers = parser.add_subparsers(dest="command")
    
    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Sync a thought to NotebookLM")
    sync_parser.add_argument("text", help="The thought or reflection to sync")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the exocortex")
    query_parser.add_argument("text", help="The question to ask")
    
    # Bootstrap command
    subparsers.add_parser("bootstrap", help="Retrieve last session state")
    
    args = parser.parse_args()
    
    if args.command == "sync":
        asyncio.run(sync_thought(args.text))
    elif args.command == "query":
        asyncio.run(query_memory(args.text))
    elif args.command == "bootstrap":
        asyncio.run(bootstrap_session())
    else:
        parser.print_help()
