#!/usr/bin/env bash
# YGN-SAGE benchmark runner — one-command validation
#
# Usage:
#   ./run_benchmarks.sh              # Full suite (Python + Rust + benchmarks)
#   ./run_benchmarks.sh --fast       # Tests only, skip benchmarks
#   ./run_benchmarks.sh --python     # Python tests only
#   ./run_benchmarks.sh --rust       # Rust tests only

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

FAST=false
PYTHON_ONLY=false
RUST_ONLY=false

for arg in "$@"; do
    case $arg in
        --fast) FAST=true ;;
        --python) PYTHON_ONLY=true ;;
        --rust) RUST_ONLY=true ;;
    esac
done

echo "============================================================"
echo "  YGN-SAGE Benchmark Suite"
echo "  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
echo ""

FAILED=0

run_step() {
    local name="$1"
    shift
    echo -e "${YELLOW}>>> $name${NC}"
    if "$@" 2>&1; then
        echo -e "${GREEN}<<< $name: PASSED${NC}"
        echo ""
    else
        echo -e "${RED}<<< $name: FAILED${NC}"
        echo ""
        FAILED=$((FAILED + 1))
    fi
}

# -----------------------------------------------
# Python tests
# -----------------------------------------------
if [ "$RUST_ONLY" = false ]; then
    run_step "Ruff lint" ruff check sage-python/src/

    run_step "Python tests (sage-python)" \
        python -m pytest sage-python/tests/ -q --tb=short

    if [ "$FAST" = false ]; then
        run_step "Routing benchmark" \
            python -m sage.bench --type routing

        run_step "Discovery tests (sage-discover)" \
            python -m pytest sage-discover/tests/ -q --tb=short 2>/dev/null || true
    fi
fi

# -----------------------------------------------
# Rust tests
# -----------------------------------------------
if [ "$PYTHON_ONLY" = false ]; then
    run_step "Cargo fmt check" \
        cargo fmt --check --manifest-path sage-core/Cargo.toml

    run_step "Cargo clippy" \
        cargo clippy --no-default-features --manifest-path sage-core/Cargo.toml -- -D warnings

    run_step "Rust tests" \
        cargo test --no-default-features --manifest-path sage-core/Cargo.toml
fi

# -----------------------------------------------
# Summary
# -----------------------------------------------
echo "============================================================"
if [ "$FAILED" -eq 0 ]; then
    echo -e "  ${GREEN}All benchmarks PASSED${NC}"
else
    echo -e "  ${RED}$FAILED benchmark(s) FAILED${NC}"
fi
echo "============================================================"

exit $FAILED
