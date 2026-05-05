#!/usr/bin/env bash
# Cycle-13 setup: clone pi-mono v0.73.0 into external/pi-mono (gitignored).
#
# Per cgpro DESIGN E (2026-05-05, conv `cgpro_pi_mono_pivot_20260505`,
# verdict GO_TIER_1_PLUS_2 trap Q3 + cycle-13 plan trap #1):
#   - pi-mono v0.73 is young (released 2026-05-04). Pin EXACT to mitigate churn.
#   - Use a clone (not submodule) for cycle-12 scaffolding to match
#     existing external/rustpython + external/meta-harness pattern.
#   - Promote to git submodule when cycle-13 arm B work begins.
#   - external/pi-mono/ is .gitignore'd — re-run this script on fresh clones.
#
# Usage:
#   ./scripts/setup_pi_mono.sh [--update]
#
# Behavior:
#   Default: clone into external/pi-mono if absent; no-op if present.
#   --update: fetch + reset to pinned commit (destructive on local edits).
set -euo pipefail

readonly REPO_URL="https://github.com/badlogic/pi-mono.git"
readonly REPO_TAG="v0.73.0"
readonly REPO_COMMIT="dbcb473d6fdb96f60570b9ebe73e7aa6316fa8fb"
readonly TARGET_DIR="external/pi-mono"

# Resolve to repo root (script may be invoked from anywhere).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

UPDATE_MODE=0
if [[ "${1:-}" == "--update" ]]; then
  UPDATE_MODE=1
fi

if [[ -d "${TARGET_DIR}/.git" ]]; then
  if [[ "${UPDATE_MODE}" -eq 1 ]]; then
    echo "[setup_pi_mono] Updating ${TARGET_DIR} to ${REPO_TAG} (${REPO_COMMIT:0:7})..."
    git -C "${TARGET_DIR}" fetch --tags origin
    git -C "${TARGET_DIR}" reset --hard "${REPO_COMMIT}"
  else
    CURRENT_COMMIT="$(git -C "${TARGET_DIR}" rev-parse HEAD)"
    if [[ "${CURRENT_COMMIT}" == "${REPO_COMMIT}" ]]; then
      echo "[setup_pi_mono] ${TARGET_DIR} already at ${REPO_TAG} (${REPO_COMMIT:0:7})."
    else
      echo "[setup_pi_mono] WARNING: ${TARGET_DIR} HEAD is ${CURRENT_COMMIT:0:7} (expected ${REPO_COMMIT:0:7})."
      echo "[setup_pi_mono] Re-run with --update to reset to pinned commit."
      exit 1
    fi
  fi
else
  echo "[setup_pi_mono] Cloning ${REPO_URL} into ${TARGET_DIR}..."
  mkdir -p "$(dirname "${TARGET_DIR}")"
  git clone "${REPO_URL}" "${TARGET_DIR}"
  git -C "${TARGET_DIR}" fetch --tags origin
  git -C "${TARGET_DIR}" checkout "${REPO_COMMIT}"
  echo "[setup_pi_mono] Cloned at ${REPO_TAG} (${REPO_COMMIT:0:7})."
fi

echo "[setup_pi_mono] Verifying npm package presence (cycle-13 arm B / arm C will need these)..."
ls "${TARGET_DIR}/packages" 2>/dev/null || {
  echo "[setup_pi_mono] WARNING: ${TARGET_DIR}/packages not found — clone may be incomplete."
  exit 1
}

echo "[setup_pi_mono] Done. Cycle-13 arm B can invoke pi-mono coding-agent via:"
echo "  cd ${TARGET_DIR}/packages/coding-agent && npm install && npm run build"
echo "[setup_pi_mono] Or use the npm package directly: npm install @mariozechner/pi-coding-agent@0.73.0"
