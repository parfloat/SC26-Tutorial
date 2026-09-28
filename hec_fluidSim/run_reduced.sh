#!/usr/bin/env bash
# run_reduced.sh -- build (if needed) and run fluidSim with a small,
# configurable iteration count, for quicker-than-upstream local runs.
#
# Usage: ./run_reduced.sh [iterations] [--overflow]
#   iterations  Number of LBM steps (default: 5). Upstream's own `make run`
#               uses 10000; MakeShort's `run` target is fixed at 1;
#               MakeOverflow's `run` target is fixed at 60. This script is
#               for anything in between, without editing a Makefile by hand.
#   --overflow  Build/run main_overflow (MakeOverflow: omega=1e8,
#               deliberately unstable, no -DVERIFY) instead of the default,
#               numerically-stable main (MakeShort: omega=1.2, -DVERIFY).
#               Useful for finding the minimum iteration count that still
#               produces NaN/Inf before wrapping the run in nixnan --
#               run_example.sh hardcodes 60 for that run, matching
#               MakeOverflow's own `run` target (see its comment: ~38
#               iterations is the estimated overflow point at omega=1e8,
#               60 leaves margin).
#
# The 256x256 grid size is a compile-time constant in main.cu/main_overflow.cu
# (lbm_width/lbm_height) -- not adjustable via this script.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OVERFLOW=0
ITERATIONS=5
for arg in "$@"; do
    case "$arg" in
        --overflow) OVERFLOW=1 ;;
        [0-9]*)     ITERATIONS="$arg" ;;
        *)
            echo "error: unrecognized argument: $arg" >&2
            echo "usage: $0 [iterations] [--overflow]" >&2
            exit 1
            ;;
    esac
done

if ! [[ "$ITERATIONS" =~ ^[0-9]+$ ]]; then
    echo "error: iterations must be a non-negative integer, got: $ITERATIONS" >&2
    exit 1
fi

if [[ "$OVERFLOW" -eq 1 ]]; then
    echo "== Building (make -f MakeOverflow main_overflow) =="
    make -f MakeOverflow main_overflow
    echo "== Running: ./main_overflow $ITERATIONS =="
    ./main_overflow "$ITERATIONS"
else
    echo "== Building (make -f MakeShort main, reusing its build rules) =="
    make -f MakeShort main
    echo "== Running: ./main $ITERATIONS =="
    ./main "$ITERATIONS"
fi
