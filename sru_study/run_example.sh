#!/usr/bin/env bash
#
# run_example.sh -- sru_study/'s nixnan demo, standalone.
# Same commands as the sru_study section of ../All-Runs.md and the
# run_sru_study() function in ../run_all_examples.sh, but self-contained:
# runnable directly from inside this directory.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NIXNAN_LIB="$SCRIPT_DIR/../external/nixnan/nixnan.so"
ITER_AUTONIXNAN="$SCRIPT_DIR/../external/nixnan/bin/iter-autonixnan"
if [[ ! -x "$ITER_AUTONIXNAN" ]]; then
    echo "error: iter-autonixnan not found at $ITER_AUTONIXNAN" >&2
    exit 1
fi

mkdir -p Logs_uninit_bug

echo "-- iter-autonixnan (no sampling) --"
"$ITER_AUTONIXNAN" -s 0 -t 600 -- python3 reproducer.py \
    > Logs_uninit_bug/iter-autonixnan-nosampling.log 2>&1

echo "-- straight nixnan, all kernels together, no sampling (~79s, no timeout wrapper --"
echo "   see ../All-Runs.md for why: nixnan can catch SIGTERM and exit 0 mid-run) --"
LD_PRELOAD="$NIXNAN_LIB" python3 -u reproducer.py \
    > Logs_uninit_bug/straight-nixnan.log 2>&1

echo "Done. See Logs_uninit_bug/ for output."
