import asyncio
import logging
import os
from notebooklm import NotebookLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NotebookLM-Sync")

NOTEBOOK_ID = "ba22b122-1755-40c7-bd41-7be0be499430"

FILES_TO_SYNC = [
    "sage-core/src/lib.rs",
    "sage-core/src/simd_sort.rs",
    "sage-core/src/sandbox/ebpf.rs",
    "sage-core/src/memory.rs",
    "sage-python/src/sage/evolution/engine.py",
    "sage-python/src/sage/strategy/solvers.py",
    "sage-python/src/sage/llm/google.py",
    "memory-bank/progress.md"
]

async def sync_codebase():
    try:
        async with await NotebookLMClient.from_storage() as client:
            logger.info(f"🚀 Début de la synchronisation vers le notebook : {NOTEBOOK_ID}")
            
            for file_path in FILES_TO_SYNC:
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        source_title = f"CODE: {file_path}"
                        logger.info(f"📤 Upload de {file_path}...")
                        # On utilise add_text pour envoyer le contenu brut du code
                        await client.sources.add_text(NOTEBOOK_ID, title=source_title, content=content)
                else:
                    logger.warning(f"⚠️ Fichier non trouvé : {file_path}")

            logger.info("✅ Synchronisation terminée. Pose d'une question stratégique...")
            
            prompt = """
            Analyse la codebase de YGN-SAGE que je viens de t'envoyer. 
            En particulier, regarde l'implémentation du Quicksort H96 en Rust (simd_sort.rs) 
            et son intégration avec le moteur d'évolution en Python.
            
            Comment pouvons-nous optimiser davantage le partitionnement AVX-512 pour 
            battre NumPy sur des tableaux de 1M d'éléments ? 
            Identifie-tu des goulots d'étranglement dans le pont PyO3 actuel ?
            """
            
            result = await client.chat.ask(NOTEBOOK_ID, prompt)
            print("\n" + "="*20 + " ANALYSE NOTEBOOKLM " + "="*20)
            print(result.answer)
            print("="*60)

    except Exception as e:
        logger.error(f"❌ Erreur : {e}")

if __name__ == "__main__":
    asyncio.run(sync_codebase())
