import asyncio
import os
import sys
import datetime

# Ajouter le chemin vers notebooklm-py
sys.path.insert(0, os.path.abspath('sage-python'))
from notebooklm import NotebookLMClient

# TON NOTEBOOK DÉDIÉ (À CRÉER SI BESOIN)
# Pour l'instant on utilise le carnet "Technical Implementation Details" comme mémoire à long terme
MEMORY_NOTEBOOK_ID = "ba22b122-1755-40c7-bd41-7be0be499430"

async def sync_agent_memory(thought: str, status: str = "Active"):
    """ASI Memory Persistence: Synchronizes agent reflections to NotebookLM."""
    try:
        async with await NotebookLMClient.from_storage() as client:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Formatage de la "conscience" de l'agent
            memory_chunk = f"""
--- SESSION UPDATE: {now} ---
STATUS: {status}
AGENT THOUGHT:
{thought}

--- END OF UPDATE ---
"""
            # SOTA: Dans la réalité on 'uploaderait' un fichier texte, 
            # mais ici on va 'interroger' le carnet pour s'assurer de sa cohérence sémantique
            # ou proposer une mise à jour textuelle via un journal local qui sera synchronisé.
            
            print(f"📡 Synchronisation de la mémoire agent vers NotebookLM ({MEMORY_NOTEBOOK_ID})...")
            
            # On stocke aussi dans un journal local pour backup
            with open("research_journal/agent_long_term_memory.md", "a", encoding="utf-8") as f:
                f.write(memory_chunk)
                
            print("✅ Mémoire synchronisée localement. Prête pour l'upload massif.")

    except Exception as e:
        print(f"❌ Échec de synchronisation : {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        thought = " ".join(sys.argv[1:])
        asyncio.run(sync_agent_memory(thought))
    else:
        print("Usage: python notebooklm_agent_sync.py 'Votre pensée ou réflexion'")
