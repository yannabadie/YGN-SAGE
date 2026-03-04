import asyncio
import logging
from notebooklm import NotebookLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NotebookLM-List")

async def list_available_notebooks():
    try:
        async with await NotebookLMClient.from_storage() as client:
            logger.info("📡 Récupération de la liste des notebooks...")
            notebooks = await client.notebooks.list()
            
            if not notebooks:
                print("Aucun notebook trouvé.")
                return

            print("\n--- VOS NOTEBOOKS NOTEBOOKLM ---")
            for nb in notebooks:
                print(f"ID: {nb.id} | Titre: {nb.title}")
            print("-" * 32)
            
    except Exception as e:
        logger.error(f"❌ Erreur lors de la connexion : {e}")
        print("\n💡 Astuce : Assurez-vous d'avoir lancé 'notebooklm login' dans votre terminal.")

if __name__ == "__main__":
    asyncio.run(list_available_notebooks())
