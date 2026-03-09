#!/usr/bin/env bash
# Build the python-runner Wasm component.
# Prerequisites:
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-tools (or wasm-tools available in PATH)
#
# Output: target/wasm32-unknown-unknown/release/python_runner.component.wasm
set -euo pipefail

cd "$(dirname "$0")"

echo "Building core Wasm module..."
cargo build --release --target wasm32-unknown-unknown

CORE="target/wasm32-unknown-unknown/release/python_runner.wasm"
COMPONENT="target/wasm32-unknown-unknown/release/python_runner.component.wasm"

echo "Wrapping core module into Component Model component..."
wasm-tools component new "$CORE" -o "$COMPONENT"

echo "Validating component..."
wasm-tools validate --features component-model "$COMPONENT"

SIZE=$(wc -c < "$COMPONENT" | tr -d ' ')
echo "Done: $COMPONENT ($SIZE bytes)"
