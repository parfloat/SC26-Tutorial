#!/usr/bin/env bash
#
# run_example.sh -- mpir_gpu/'s nixnan demo, standalone.
# Same commands as the mpir_gpu section of ../All-Runs.md and the
# run_mpir_gpu() function in ../run_all_examples.sh, but self-contained:
# runnable directly from inside this directory.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ITER_AUTONIXNAN="$SCRIPT_DIR/../external/nixnan/bin/iter-autonixnan"
if [[ ! -x "$ITER_AUTONIXNAN" ]]; then
    echo "error: iter-autonixnan not found at $ITER_AUTONIXNAN" >&2
    exit 1
fi

mkdir -p iter-autonixnan

echo "-- building --"
make

echo "-- iter-autonixnan (finds real exceptions in geqr2_smem_domino_fast) --"
"$ITER_AUTONIXNAN" -s 4 -t 90 -- ./ir_gpu --n 256 --kappa 100 --config F32 \
    > iter-autonixnan/sampling4-n256-kappa100-F32.log 2>&1

echo "Done. See iter-autonixnan/ for output."
