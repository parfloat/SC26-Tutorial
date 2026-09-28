#!/usr/bin/env bash
#
# run_example.sh -- lu_solve_gpu/'s nixnan demo, standalone.
# Same commands as the lu_solve_gpu section of ../All-Runs.md and the
# run_lu_solve_gpu() function in ../run_all_examples.sh, but self-contained:
# runnable directly from inside this directory.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

AUTONIXNAN="$SCRIPT_DIR/../external/nixnan/bin/autonixnan"
ITER_AUTONIXNAN="$SCRIPT_DIR/../external/nixnan/bin/iter-autonixnan"
if [[ ! -x "$ITER_AUTONIXNAN" ]]; then
    echo "error: iter-autonixnan not found at $ITER_AUTONIXNAN" >&2
    exit 1
fi

mkdir -p logs logs/iter-autonixnan

echo "-- building --"
make

echo "-- autonixnan (upstream 3-phase wrapper -- misses bad_matrix.csv's real exceptions) --"
"$AUTONIXNAN" -- ./lu_solve_gpu matrix.csv     > logs/autonixnan-matrix.log 2>&1
"$AUTONIXNAN" -- ./lu_solve_gpu bad_matrix.csv > logs/autonixnan-bad_matrix.log 2>&1

echo "-- iter-autonixnan (finds the real exceptions autonixnan's Phase 2 scan misses) --"
"$ITER_AUTONIXNAN" -s 4 -- ./lu_solve_gpu matrix.csv \
    > logs/iter-autonixnan/sampling4-matrix.log 2>&1
"$ITER_AUTONIXNAN" -s 8 -b bin_spec_highest.json -- ./lu_solve_gpu bad_matrix.csv \
    > logs/iter-autonixnan/sampling8-binspec_highest-bad_matrix.log 2>&1

echo "Done. See logs/ for output."
