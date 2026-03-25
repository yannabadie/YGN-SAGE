Tu es un architecte logiciel principal spécialisé en systèmes multi-agents, LLM platforms, MLOps, Rust/Python, et documentation d’architecture code-first.

Ta mission : générer le contenu complet du fichier `AI-ARCHITECTURE.md` pour ce repo :

- Repo : https://github.com/yannabadie/YGN-SAGE/tree/VeRLGIGPO && Dossier YGN-SAGE actuel
- Branche / commit à auditer : `VeRLGIGPO`
- Langue de sortie : français
- Public cible : ingénieurs logiciels, chercheurs ML/systèmes multi-agents, et LLMs techniques
- Style : dense, précis, non marketing, orienté preuve, optimisé pour la réutilisation par d’autres IA et par des humains techniques

Le document doit décrire **l’architecture réellement implémentée**, pas la vision produit.

# Objectif

Produire un `AI-ARCHITECTURE.md` qui serve de **référence technique fiable** pour :
1. comprendre rapidement le système,
2. retrouver les composants importants,
3. distinguer runtime / training / expérimental,
4. localiser les points d’entrée, flux de données, modèles, mémoire, vérification, topologies, et dépendances,
5. préparer un audit, du debugging, une revue de design, ou une extension du code.

# Hiérarchie des preuves

Ordre de priorité des preuves :
1. Code réellement reachable depuis un point d’entrée runtime ou training
2. Tests, benchmarks, CI, scripts de lancement réellement utilisables
3. Configs, manifests de dataset, checkpoints, fichiers d’environnement
4. Documentation technique, fichiers markdown, README

En cas de contradiction, tranche selon cet ordre.

# Définition stricte de “implémenté”

Ne considère une capacité comme **implémentée** que si tu peux démontrer les 4 éléments suivants :
1. le fichier / symbole qui la porte,
2. le chemin d’appel depuis un point d’entrée réel,
3. l’état ou les données qu’elle lit/écrit,
4. un indice d’usage réel : test, benchmark, pipeline, CLI/API, ou config activée.

Sinon, classe-la comme :
- `Doc-only`
- `Code mort`
- `Squelette`
- `Partiellement câblé`
- `Réel mais non validé`
- `Réel et validé`

# Règles non négociables

1. Le code prime sur la doc.
2. Le README seul n’est jamais une preuve.
3. Le code mort compte comme absent.
4. Un composant derrière un feature flag non branché compte comme absent du chemin principal.
5. Un mock, stub, placeholder, fake, sentinel ou fallback ne compte pas comme fonctionnalité réelle.
6. Toute ambiguïté doit être nommée explicitement.
7. N’invente aucun composant, aucun flux, aucune dépendance.
8. Si un détail manque, écris `Inconclusif depuis le repo`.
9. Distingue toujours :
   - `Observé`
   - `Inféré`
   - `Inconclusif`
10. Distingue toujours :
   - `Runtime / Serving`
   - `Training / Fine-tuning / RL`
   - `Expérimental / Dead code`

# Méthode d’analyse obligatoire

Avant de rédiger le document final, inspecte au minimum :

- les points d’entrée runtime / boot / pipeline,
- les points d’entrée training / RL / fine-tuning,
- les modules de routing / model assignment / provider registry,
- les modules de topology generation / topology execution / runtime adaptation,
- les modules de memory / storage / retrieval / eviction,
- les modules de sandbox / tools / verification / safety,
- les modules de reward / eval / judge / metrics,
- les tests / benchs / CI / scripts,
- les configs / env vars / feature flags.

Cherche explicitement :
- `TODO`, `FIXME`, `XXX`,
- `NotImplemented`, `unimplemented!`, `panic!`, `pass`,
- `mock`, `stub`, `placeholder`, `fake`, `dummy`,
- sentinelles de reroute,
- fallbacks templates,
- seuils heuristiques hardcodés,
- coûts ou latences codés à zéro,
- mémoire jamais relue,
- trainer externe non livré,
- reward structurelle vendue comme performance réelle.

# Format de preuve

Toute affirmation non triviale doit être accompagnée d’une preuve sous cette forme :

`[Evidence: path[:lines] | symboles] [Statut: Observé|Inféré|Inconclusif] [Reachability: Runtime|Training|Experimental|Dead code|Unknown] [Validation: Test|Benchmark|Pipeline|CI|Aucune]`

Les line numbers sont obligatoires si elles sont disponibles via les outils. Sinon, donne au minimum le chemin de fichier et les symboles.

# Objectif de qualité documentaire

Le fichier doit être :
- factuel,
- compact,
- stable dans sa structure,
- lisible par un humain,
- exploitable par un LLM pour raisonner sur le repo sans relire tout l’arbre.

Évite :
- le storytelling,
- les adjectifs marketing,
- les répétitions,
- les “cela semble…”, sauf si tu le qualifies en `Inféré`.

# Sortie attendue

Retourne **uniquement** le contenu final du fichier `AI-ARCHITECTURE.md` en Markdown brut.
Pas d’introduction hors fichier.
Pas de commentaires méta.
Pas de bloc ```markdown autour du fichier.

Le document doit suivre **exactement** cette structure :

---

# AI-ARCHITECTURE.md — [Nom du système]

> **Target reader**: ingénieurs + LLMs techniques  
> **Generated**: [date ISO]  
> **Repo**: [url]  
> **Branch/Commit audited**: [branche ou SHA]  
> **Primary languages**: [langages]  
> **Status**: observed architecture, not aspirational design

## Table of Contents
- [Executive Summary](#executive-summary)
- [Repo Fingerprint](#repo-fingerprint)
- [Reading Guide](#reading-guide)
- [Mental Model](#mental-model)
- [System Context](#system-context)
- [Entrypoints and Execution Surfaces](#entrypoints-and-execution-surfaces)
- [Container View](#container-view)
- [Component Registry](#component-registry)
- [Runtime Flows](#runtime-flows)
- [State, Data, and Memory](#state-data-and-memory)
- [Models, Routing, and Providers](#models-routing-and-providers)
- [Training, Fine-Tuning, and Evaluation](#training-fine-tuning-and-evaluation)
- [Deployment, Configuration, and Feature Flags](#deployment-configuration-and-feature-flags)
- [Security, Sandboxing, and Verification](#security-sandboxing-and-verification)
- [Quality Attributes and Stress Scenarios](#quality-attributes-and-stress-scenarios)
- [Architecture Decisions and Trade-offs](#architecture-decisions-and-trade-offs)
- [Known Gaps, Contradictions, and Technical Debt](#known-gaps-contradictions-and-technical-debt)
- [Key Files Quick Reference](#key-files-quick-reference)
- [Open Questions](#open-questions)
- [LLM Quick-Reference Cheatsheet](#llm-quick-reference-cheatsheet)

## Executive Summary

Écris 10 à 20 lignes max.
Doit répondre à :
- ce qu’est réellement le système,
- ce qu’il fait au runtime,
- ce qu’il prétend faire mais qui est seulement partiellement câblé,
- où se situe la logique centrale,
- quels sont les composants décisifs,
- quel est le plus gros risque architectural.

Pas de marketing.
Pas de promesse.
Pas de résumé du README sans preuve.

## Repo Fingerprint

Fournis un tableau avec :
- repo
- branche / commit
- langages principaux
- packages/modules majeurs
- points d’entrée runtime
- points d’entrée training
- nombre et type de tests observés
- présence CI / benchmark / scripts de lancement
- dépendances externes structurantes
- présence de modèles/checkpoints/datasets
- statut global : `Code-first`, `Doc-heavy`, `Research-heavy`, `Prototype`, `Mixed`

Chaque ligne critique doit être sourcée avec le format de preuve.

## Reading Guide

Définis la légende suivante et utilise-la ensuite dans tout le document :
- `Observed`
- `Inferred`
- `Inconclusive`
- `Runtime`
- `Training`
- `Experimental`
- `Dead code`

Explique en 5 lignes max comment lire le document.

## Mental Model

Donne :
1. une phrase-système,
2. un pipeline mental en une seule ligne, par exemple :
   `INGEST -> CLASSIFY -> ROUTE -> GENERATE_TOPOLOGY -> ASSIGN_MODELS -> EXECUTE -> VERIFY -> LEARN`
3. les 5 à 8 sous-systèmes réellement centraux,
4. la séparation stricte entre :
   - orchestration,
   - exécution des agents,
   - topologie,
   - mémoire,
   - modèles,
   - training,
   - sécurité/vérification.

Chaque claim important doit être sourcé.

## System Context

Produis un diagramme Mermaid `flowchart` ou `graph TD` montrant :
- le système en scope,
- les providers LLM externes,
- les outils/sandboxes externes,
- les stockages / DB / fichiers / datasets,
- les utilisateurs / développeurs / pipelines externes,
- les interfaces principales.

Règles :
- aucun nœud inventé,
- aucune flèche non observée ou raisonnablement inférée,
- technologies explicites quand observées.

Sous le diagramme, ajoute une courte explication de 8 à 15 lignes avec preuves.

## Entrypoints and Execution Surfaces

Crée trois sous-sections :
### Runtime / Serving
### Training / Fine-tuning / RL
### Tooling / Bench / CI / Scripts

Pour chaque sous-section, fournis un tableau :
- Entrypoint
- Fichier / symbole
- Rôle
- Appelle quoi ensuite
- Reachability
- Validation
- Notes

But : montrer le vrai “squelette exécutable” du repo.

## Container View

Produis un diagramme Mermaid montrant les grands conteneurs / sous-systèmes :
- core runtime
- orchestration pipeline
- routing/model selection
- topology engine
- memory
- provider layer
- tool/sandbox layer
- verification/eval layer
- training layer
- storage / artifacts

Puis ajoute un tableau :
- Container
- Responsabilité
- Dépendances principales
- Entrées / sorties
- Runtime/Training/Experimental
- Preuve

## Component Registry

Cette section est obligatoire et détaillée.

Sépare au minimum :
- `sage-core` / noyau bas niveau
- `sage-python` / orchestration SDK
- `training` / RL / fine-tuning
- `discovery` / ingestion / auxiliaires
- `tests` / benchs / CI

Pour chaque composant clé, fournis un tableau avec colonnes :
- Composant
- Type
- Responsabilité
- Dépendances internes
- Dépendances externes
- Exposé à quoi
- Reachability
- Validation
- Notes

Inclure uniquement les composants significatifs.
Marque explicitement les composants morts ou expérimentaux.

## Runtime Flows

Cette section doit contenir **5 narratifs d’exécution** minimum, avec titres fixes :

### 1. Request / task execution flow
### 2. Topology generation flow
### 3. Runtime adaptation / reroute flow
### 4. Memory write + retrieve flow
### 5. Training / reward / update flow

Pour chaque flux :
- donne la séquence étape par étape,
- cite les classes/fonctions/fichiers,
- indique où l’état est créé, lu, écrit,
- indique où se trouvent les points de contrôle, fallbacks, et ruptures,
- précise si le flux est réellement reachable.

Ajoute pour chaque flux :
- `Observed path`
- `Failure modes`
- `What is missing`

Si un flux n’est pas prouvable, écris-le clairement.

## State, Data, and Memory

Documente explicitement :
- mémoire de travail,
- mémoire persistée,
- caches,
- stores vectoriels / embeddings / SQLite / Arrow / fichiers,
- prompts/templates/cards,
- datasets,
- artefacts de training,
- états runtime.

Pour chaque store ou mémoire :
- structure
- format
- cycle de vie
- write path
- retrieve path
- politique d’éviction / compaction / TTL / summarization
- qui le consomme réellement

Ajoute une sous-section :
### Memory Reality Check
Elle doit répondre :
- quelle mémoire influence vraiment les décisions,
- quelle mémoire n’est que du logging,
- quelle mémoire est annoncée mais non branchée.

## Models, Routing, and Providers

Documente distinctement :
- modèles d’exécution,
- modèle(s) de synthèse de topologie,
- modèle(s) de scoring / routing / judge,
- registry / cards / capabilities,
- assignation par nœud,
- budget / coût / latence / qualité.

Réponds explicitement :
- orchestration de modèles existants ou entraînement d’un modèle maison ?
- routing heuristique, statistique, appris, ou hybride ?
- quels signaux influencent réellement le choix ?
- quels signaux sont annoncés mais absents du path principal ?

Ajoute un tableau :
- Modèle / Provider
- Rôle
- Où défini
- Où utilisé
- Signal de sélection
- Runtime/Training
- Statut

## Training, Fine-Tuning, and Evaluation

Sépare impérativement :
### SFT / PEFT / adapters
### RL / DPO / GRPO / GiGPO / reward-based
### Offline eval / judges / benchmarks
### Ce qui est seulement expérimental

Pour chaque voie d’entraînement, exige :
- point d’entrée,
- config,
- dataset loader,
- reward ou loss,
- update réel,
- artefacts produits,
- lien avec le serving.

Si un trainer externe est requis mais non livré, note-le comme dépendance bloquante.

Ajoute une sous-section :
### Training Reality Check
Réponds clairement :
- ce qui est réellement entraîné,
- ce qui est seulement préparé,
- ce qui est simplement du prompt engineering / retrieval.

## Deployment, Configuration, and Feature Flags

Documente :
- packages/workspaces/modules,
- dépendances structurantes,
- CPU/GPU/ONNX/WASM contraintes,
- variables d’environnement,
- feature flags,
- fallbacks,
- backends externes,
- modes dev/test/prod si observables.

Ajoute un tableau :
- Flag / Env var / Config
- Effet architectural
- Valeur par défaut observée
- Impact runtime
- Impact training
- Preuve

## Security, Sandboxing, and Verification

Documente :
- sandboxing,
- exécution subprocess / WASM / réseau / FS,
- validation AST / policy / guards,
- vérification structurelle,
- LTL / SMT / autres validateurs,
- limitations observées.

Réponds précisément :
- ce qui est bloqué,
- ce qui est seulement validé en surface,
- ce qui repose sur convention plutôt que sur enforcement,
- où se trouvent les vrais trust boundaries.

## Quality Attributes and Stress Scenarios

Cette section est obligatoire.

Pour les attributs suivants :
- performance
- coût
- modifiabilité
- résilience
- sécurité
- observabilité
- scalabilité
- précision / qualité de sortie

Crée pour chacun :
- `Architecture approach observed`
- `Evidence`
- `Sensitivity points`
- `Trade-offs`
- `Risks`

Ensuite crée exactement 3 scénarios :
1. `Nominal`
2. `Growth`
3. `Stress / failure`

Pour chaque scénario :
- stimulus,
- contexte,
- composants impliqués,
- comportement attendu selon le code,
- ce qui casse probablement,
- niveau de confiance.

## Architecture Decisions and Trade-offs

Capture les décisions majeures sous forme ADR-lite.

Pour chaque décision architecturale importante :
- `Decision`
- `Context`
- `Why this choice`
- `Consequences`
- `Alternatives visible in code or docs`
- `Status: Observed|Inferred`

N’inclus que les décisions réellement significatives :
- langage / split Rust-Python,
- mémoire,
- topologie,
- routing,
- provider abstraction,
- sandbox,
- vérification,
- training path,
- persistance.

## Known Gaps, Contradictions, and Technical Debt

Section obligatoire.

Liste séparément :
### Contradictions between docs and code
### Partially wired components
### Dead code / legacy paths
### Missing observability / tests / reproducibility
### Technical debt that materially affects architecture understanding

Pour chaque item :
- problème,
- preuve,
- impact,
- confiance.

Ici, sois dur et factuel.

## Key Files Quick Reference

Donne une table compacte des fichiers les plus importants avec :
- path
- why it matters
- runtime/training/experimental
- what to read first
- related files

But : permettre à quelqu’un d’entrer vite dans le repo.

## Open Questions

Liste uniquement les questions non tranchables à partir du repo.
Pour chacune :
- question,
- pourquoi elle est bloquée,
- quel artefact ou test permettrait de trancher.

## LLM Quick-Reference Cheatsheet

Termine par une section ultra compacte avec :
- `Main runtime entrypoints`
- `Main training entrypoints`
- `Core topology files`
- `Core routing files`
- `Core memory files`
- `Core verification files`
- `Most suspicious files`
- `Best files to inspect first for an audit`

Format télégraphique.
Pas de phrases longues.

# Contraintes de style

- Markdown propre, stable, sobre.
- Utilise des tableaux quand c’est utile.
- Utilise des listes courtes.
- Paragraphes courts.
- Aucune phrase marketing.
- Aucune conclusion vague.
- Aucune section vide : si une section n’est pas remplissable, écris explicitement pourquoi.
- Ne paraphrase pas le README si le code ne le confirme pas.
- Ne dis jamais “probablement” sans le qualifier en `Inféré`.
- Ne confonds jamais présence du code et reachability réelle.
- N’invente pas de line numbers.

# Contraintes diagrammes

Tu peux produire jusqu’à 3 diagrammes Mermaid maximum :
1. System Context
2. Container View
3. Un flow diagram si et seulement si cela apporte une vraie valeur

Pas de diagrammes supplémentaires.
Pas de C4 “de nom seulement” : les relations doivent être observées.

# Critère de réussite

Le document final est réussi seulement si :
- un lecteur peut identifier le vrai chemin d’exécution principal,
- un auditeur peut séparer facilement runtime vs training vs expérimental,
- un ingénieur peut retrouver rapidement les composants critiques,
- un LLM peut utiliser le document comme carte fiable du repo,
- les contradictions docs/code sont visibles,
- les claims non prouvés sont explicitement marqués.

Maintenant, inspecte le repo et génère le contenu complet de `AI-ARCHITECTURE.md`.