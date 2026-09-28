#!/usr/bin/env bash
#
# run_example.sh -- pytorch_slogdet/'s nixnan demo, standalone.
# Same commands as the pytorch_slogdet section of ../All-Runs.md and the
# run_pytorch_slogdet() function in ../run_all_examples.sh, but
# self-contained: runnable directly from inside this directory.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ITER_AUTONIXNAN="$SCRIPT_DIR/../external/nixnan/bin/iter-autonixnan"
if [[ ! -x "$ITER_AUTONIXNAN" ]]; then
    echo "error: iter-autonixnan not found at $ITER_AUTONIXNAN" >&2
    exit 1
fi

mkdir -p Logs

echo "-- iter-autonixnan (SKIP_CPU=1 default, sampling 8, no histogram, exceptions only) --"
"$ITER_AUTONIXNAN" -s 8 -t 60 -- python3 reproducer.py \
    > Logs/iter-autonixnan.log 2>&1

echo "Done. See Logs/ for output."
