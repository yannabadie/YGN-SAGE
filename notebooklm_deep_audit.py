import asyncio
import logging
from notebooklm import NotebookLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NotebookLM-DeepAudit")

NOTEBOOKS = {
    "RESEARCH": "34d65dbb-4299-46e3-ab04-07879ed64541", # Core Research & MARL
    "TECHNICAL": "ba22b122-1755-40c7-bd41-7be0be499430" # Technical Implementation
}

async def deep_audit():
    async with await NotebookLMClient.from_storage() as client:
        # 1. Audit du Notebook de RECHERCHE
        logger.info(f"🔍 Audit approfondi du Notebook RECHERCHE ({NOTEBOOKS['RESEARCH']})...")
        research_prompt = """
        Effectue un inventaire exhaustif de TOUTES les spécifications algorithmiques mentionnées dans tes sources.
        Je veux :
        - Les hyperparamètres exacts pour VAD-CFR et SHOR-PSRO (escompte, lambda, seuils).
        - Les mandates ASI : quelles sont les conditions strictes pour qu'une évolution soit validée ?
        - Y a-t-il des mentions de 'H102' ou 'H112' ? Si oui, quelles sont leurs spécifications ?
        """
        research_audit = await client.chat.ask(NOTEBOOKS["RESEARCH"], research_prompt)
        
        # 2. Audit du Notebook TECHNIQUE
        logger.info(f"🔍 Audit approfondi du Notebook TECHNIQUE ({NOTEBOOKS['TECHNICAL']})...")
        tech_prompt = """
        Effectue un inventaire exhaustif des contraintes matérielles et logicielles.
        Je veux :
        - Les cibles SIMD exactes (AVX-512, AMX, etc.) et les instructions privilégiées.
        - La structure attendue pour la mémoire Zero-copy Arrow (types de colonnes, schémas).
        - Les objectifs de performance chiffrés (latence sandbox, throughput agent).
        """
        tech_audit = await client.chat.ask(NOTEBOOKS["TECHNICAL"], tech_prompt)

        print("\n" + "💎 SYNTHÈSE DE L'AUDIT DE RECHERCHE " + "="*30)
        print(research_audit.answer)
        print("\n" + "🛠 SYNTHÈSE DE L'AUDIT TECHNIQUE " + "="*30)
        print(tech_audit.answer)
        print("="*70)

if __name__ == "__main__":
    asyncio.run(deep_audit())
