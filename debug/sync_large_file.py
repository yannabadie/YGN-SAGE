import asyncio
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
from notebooklm_agent_sync import sync_thought

async def main():
    file_path = "docs/plans/comprehensive_knowledge_transfer.md"
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    print("Uploading comprehensive knowledge transfer to NotebookLM...")
    await sync_thought(content)
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())
