#!/usr/bin/env bash
# run_reduced.sh -- build (if needed) and run fluidSim with a small,
# configurable iteration count, for quicker-than-upstream local runs.
#
# Usage: ./run_reduced.sh [iterations]
#   iterations  Number of LBM steps (default: 5). Upstream's own `make run`
#               uses 10000; MakeShort's `run` target is fixed at 1. This
#               script is for anything in between, without editing a
#               Makefile by hand.
#
# The 256x256 grid size is a compile-time constant in main.cu
# (lbm_width/lbm_height) -- not adjustable via this script.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ITERATIONS="${1:-5}"

if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]]; then
    echo "error: iterations must be a non-negative integer, got: $ITERATIONS" >&2
    exit 1
fi

echo "== Building (make -f MakeShort, reusing its build rules) =="
make -f MakeShort main

echo "== Running: ./main $ITERATIONS =="
./main "$ITERATIONS"
