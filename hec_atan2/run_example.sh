#!/usr/bin/env bash
#
# run_example.sh -- hec_atan2/'s nixnan demo, standalone.
# Same commands as the hec_atan2 section of ../All-Runs.md and the
# run_hec_atan2() function in ../run_all_examples.sh, but self-contained:
# runnable directly from inside this directory.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NIXNAN_LIB="$SCRIPT_DIR/../external/nixnan/nixnan.so"
if [[ ! -e "$NIXNAN_LIB" ]]; then
    echo "error: nixnan.so not found at $NIXNAN_LIB (build external/nixnan first)" >&2
    exit 1
fi

mkdir -p Log

echo "-- building (MakeShort: 8 coordinates, repeat once) --"
make -f MakeShort

echo "-- default nixnan run (no sampling) --"
LD_PRELOAD="$NIXNAN_LIB" ./main 8 1 \
    > Log/atan2-8x1-nixnan-default-exceptions.log 2>&1

echo "Done. See Log/ for output."
