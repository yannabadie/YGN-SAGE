---
title: "Sandbox Safety"
type: issue
severity: P1
status: durci-pas-resolu
tags:
  - issue
  - security
  - sandbox
created: 2026-04-07
---

# Sandbox Safety

## Probleme

`sage-python/src/sage/tools/meta.py:129` execute `/bin/bash -c` avec une simple blocklist regex.
Expose via `boot_tools.py:64`.

## Etat actuel

- Blocklist regex durcie : mkfs, rm-rf, dd, shutdown, reboot, fork bomb, chmod 777
- Tests valident que la blocklist fonctionne
- **Mais** : une blocklist n'est jamais exhaustive

## Risque residuel

Un attaquant peut contourner la blocklist avec :
- Encodage (base64, hex)
- Aliases, fonctions shell
- Chemins absolus non prevus
- Combinaisons de commandes innocentes

## Solutions possibles

1. **Sandbox formelle** (Wasm WASI) : deja en place en couche 2 mais pas utilisee pour meta-tools
2. **Isolation processus** : seccomp/AppArmor (Linux), Job Objects (Windows)
3. **Deny-by-default** : au lieu de bloquer le mauvais, autoriser seulement le bon

## Papers associes

Aucun paper specifique dans Researches/. La securite sandbox est un probleme d'ingenierie, pas de recherche.
