import asyncio
import logging
from notebooklm import NotebookLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NotebookLM-MARL")

NOTEBOOK_ID = "34d65dbb-4299-46e3-ab04-07879ed64541"

async def query_marl_notebook():
    try:
        async with await NotebookLMClient.from_storage() as client:
            logger.info(f"📡 Interrogation du notebook de recherche : {NOTEBOOK_ID}")
            
            prompt = """
            Analyse les recherches sur VAD-CFR et MARL présentes dans ce notebook. 
            Comment ces concepts de théorie des jeux peuvent-ils être directement appliqués 
            à notre système d'évolution de code (MAP-Elites) ?
            
            Plus précisément :
            1. Comment stabiliser les scores de fitness volatils issus de sandbox non-déterministes en utilisant VAD-CFR ?
            2. Peux-tu proposer une architecture de 'Méta-Allocateur' de ressources qui utilise le regret accumulé 
               pour décider quels agents de mutation (Gemini vs Claude) utiliser ?
            3. Identifie-tu des opportunités pour utiliser SHOR-PSRO dans la topologie des agents YGN-SAGE ?
            """
            
            result = await client.chat.ask(NOTEBOOK_ID, prompt)
            print("\n" + "="*20 + " RECHERCHE CORE & MARL " + "="*20)
            print(result.answer)
            print("="*60)

    except Exception as e:
        logger.error(f"❌ Erreur : {e}")

if __name__ == "__main__":
    asyncio.run(query_marl_notebook())
