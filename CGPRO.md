  Rôle de cgpro
  cgpro est un conseiller GPT-5.5 Pro via la session ChatGPT locale de Yann. Il sert pour : second avis, décisions d’architecture, arbitrages risqués, recherche web fraîche, synthèse longue, et continuité de protocole.
  Il ne remplace pas l’inspection locale du repo ni les tests.

  Règle centrale
  Ne lui demande pas “que faire ?” de façon vague. Donne-lui :

  - le repo et le commit courant ;
  - le problème exact ;
  - les fichiers/surfaces autorisées ;
  - les surfaces interdites ;
  - les checks déjà passés ;
  - le format de réponse attendu.

  Exemple utile :

  cgpro ask --resume autonomous-protocol-20260430 "Repo: C:\Code\Unslop.ai.
  HEAD: <sha>.
  Contexte: ...
  Décision demandée: choisir le prochain bloc étroit.
  Contraintes: pas de src/oida_code, pas de .github/workflows, pas de provider call...
  Réponds avec NEXT_BLOCK_ID, allowed_files, forbidden_scope, required_tests, stop_conditions."

  Multi-turn et continuité
  Utilise toujours un thread nommé quand le travail dure :

  cgpro ask --save autonomous-protocol-20260501 "..."
  cgpro ask --resume autonomous-protocol-20260501 "follow-up..."
  cgpro thread list

  Dans un projet lié :

  cgpro project show
  cgpro project link "Nom du projet"
  cgpro project digest

  Pré-commit
  Avant commit/push sur un bloc choisi par cgpro, refaire une consultation courte :

  cgpro ask --resume autonomous-protocol-20260501 "Pre-commit review.
  Baseline: <sha>.
  Diff résumé: ...
  Non-changes vérifiés: ...
  Checks passés: ...
  Question: réponds exactement GO_COMMIT_PUSH ou HARD_STOP avec raison."

  Ne commit pas si cgpro répond HARD_STOP.

  Post-push
  Après push, rapporter le résultat :

  cgpro ask --resume autonomous-protocol-20260501 "Post-push report:
  commit <sha> pushed.
  Validation: ...
  Status: main aligned origin/main.
  Return NEXT_BLOCK_ID only."

  Ce qu’il ne faut pas faire

  - Ne jamais coller les valeurs de .env, clés API, tokens, PAT, secrets.
  - Ne pas lui envoyer un énorme dump brut quand un résumé exact suffit.
  - Ne pas suivre cgpro aveuglément : Codex doit re-lire les fichiers, vérifier le diff et exécuter les tests.
  - Ne pas lui demander des décisions triviales.
  - Ne pas lui faire choisir à partir d’un état local ambigu : préciser si le code est poussé ou seulement résumé.
  - Ne pas laisser un prompt ouvert sans format de sortie contraint.

  Quand cgpro est lent ou occupé
  Les réponses longues peuvent prendre 5 à 30 minutes. Si indisponible, attendre plutôt que contourner le protocole. Si erreur Not signed in, Cloudflare challenge, ou Selector broken, remonter l’erreur à Yann ; ne pas
  tenter de s’authentifier à sa place.

  Formule de bon usage
  cgpro décide ou challenge. Codex exécute, vérifie, commit, push. Toute décision importante doit être transformée en artefact durable : AGENTS.md, QA/Axx.md, reports/..., memory-bank/..., tests de garde.