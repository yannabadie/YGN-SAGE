#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
UPGRADE_FLAG=()

if [[ "${1:-}" == "--upgrade" ]]; then
  UPGRADE_FLAG=(--upgrade)
elif [[ "${1:-}" != "" ]]; then
  echo "usage: $0 [--upgrade]" >&2
  exit 2
fi

# Pin the compiler toolchain. pip-tools 7.5.3 is current enough for the
# pip 26.0 generation baseline and avoids resolver behavior drift during
# constraints generation.
"$PYTHON_BIN" -m pip install --upgrade "pip==26.0" "pip-tools==7.5.3" maturin

# sage-python declares sage-core as a runtime dependency, but sage-core is not
# a normal PyPI-resolved dependency until B4 publishes platform wheels. Build a
# local wheel so pip-compile can resolve the declared dependency honestly.
cd "$ROOT/sage-core"
"$PYTHON_BIN" -m maturin build --release --features smt,onnx --out target/wheels

# sage-discover depends on ygn-sage, and the published 0.1.0 artifact can lag
# this monorepo's current sage-python metadata. Put a local ygn-sage wheel in
# the same wheelhouse so the discover constraints resolve against this checkout.
cd "$ROOT/sage-python"
"$PYTHON_BIN" -m pip wheel . --no-deps --wheel-dir "$ROOT/sage-core/target/wheels"

# Prefer the just-built local ygn-sage wheel over an equal-version PyPI wheel
# when resolving sage-discover's ygn-sage>=0.1.0 dependency.
for wheel in "$ROOT"/sage-core/target/wheels/ygn_sage-*-py3-none-any.whl; do
  [[ -e "$wheel" ]] || continue
  base="$(basename "$wheel")"
  if [[ "$base" =~ ^(ygn_sage-[^-]+)-py3-none-any\.whl$ ]]; then
    cp "$wheel" "$ROOT/sage-core/target/wheels/${BASH_REMATCH[1]}-1-py3-none-any.whl"
  fi
done

export CUSTOM_COMPILE_COMMAND="./scripts/compile_python_constraints.sh"

cd "$ROOT/sage-python"
"$PYTHON_BIN" -m piptools compile pyproject.toml \
  --extra all \
  --extra dev \
  --strip-extras \
  --resolver=backtracking \
  --newline=lf \
  --no-emit-index-url \
  --no-emit-trusted-host \
  --no-emit-find-links \
  --find-links "$ROOT/sage-core/target/wheels" \
  --unsafe-package ygn-sage \
  --output-file constraints.txt \
  "${UPGRADE_FLAG[@]}"

cd "$ROOT/sage-discover"
# Layer the discover compile on top of sage-python's already-pinned
# constraints. pip constraints are global version limits, not independent
# lock universes — the python-discover CI job installs sage-python AND
# sage-discover in the same env, so the two constraint files MUST be
# composable. Without `--constraint`, pip-compile resolves typer freely
# for sage-discover (e.g. 0.25.0) while sage-python pins it elsewhere
# (e.g. 0.21.2), producing an unsatisfiable dual `-c` install on Linux.
"$PYTHON_BIN" -m piptools compile pyproject.toml \
  --constraint ../sage-python/constraints.txt \
  --strip-extras \
  --resolver=backtracking \
  --newline=lf \
  --no-emit-index-url \
  --no-emit-trusted-host \
  --no-emit-find-links \
  --find-links "$ROOT/sage-core/target/wheels" \
  --output-file constraints.txt \
  "${UPGRADE_FLAG[@]}"
