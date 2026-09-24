#!/usr/bin/env bash
# Stop_First_Error_with_History_FP16.sh -- FP16 histogram trace, then stop at
# the first exception.
#
# Goal: root-cause how FP16's values built up to its exception, by printing a
# dense per-occurrence exponent-histogram trace right up to the moment of the
# first exception, then stopping the program immediately.
#
# Flags used:
#   MAX_ERRORS=1        -- nixnan env var: stop the whole program (SIGTERM)
#                           right after the first exception line is printed.
#   BIN_SPEC_FILE=...    + HISTOGRAM=1 -- watch FP16's smallest two and
#                           largest two normal exponents (FP16 has a 5-bit
#                           exponent, bias 15: smallest = [-14,-13], largest
#                           = [14,15], one below the inf/nan field).
#                           "count":1 is the histogram bucket size -- print
#                           on every single occurrence landing in a watched
#                           range. This is the finest granularity nixnan
#                           offers; note its bins are scoped per kernel
#                           function + exponent range, not per individual
#                           SASS instruction/PC, so "count":1 is the closest
#                           available approximation to a per-instruction
#                           trace.
#
# rd_nixnan's main() runs FP16 first, and FP16 is the only format that fails
# this early (~step 300), so MAX_ERRORS=1 naturally stops the whole program
# at FP16's first exception -- no separate isolation needed.
#
# Caveat: the histogram and exception reports come from two independent
# receiver threads writing to the same output stream with no lock between
# them, so expect a few histogram lines to trail after "Terminating early",
# and occasionally one torn/interleaved line right at the exception itself.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BIN_SPEC="$SCRIPT_DIR/Stop_First_Error_with_History_FP16.json"
OUT="$LOGS_DIR/Stop_First_Error_with_History_FP16.log"

echo "== Running rd_nixnan under nixnan (FP16 histogram trace, MAX_ERRORS=1) -> $OUT =="
HISTOGRAM=1 BIN_SPEC_FILE="$BIN_SPEC" MAX_ERRORS=1 LD_PRELOAD="$NIXNAN_SO" "$EXE" > "$OUT" 2>&1 || true

echo
echo "== Trace summary =="
echo "Total lines: $(wc -l < "$OUT")"
echo "Occurrences in [-14,-13] (smallest two exponents): $(grep -c 'range=\[-14,-13\]' "$OUT" || true)"
echo "Occurrences in [14,15] (largest two exponents):    $(grep -c 'range=\[14,15\]' "$OUT" || true)"
echo "Exception lines printed: $(grep -c 'error \[' "$OUT" || true)"
echo
echo "== The exception, and what stopped the program =="
grep -A1 "error \[" "$OUT" | head -2 || true
