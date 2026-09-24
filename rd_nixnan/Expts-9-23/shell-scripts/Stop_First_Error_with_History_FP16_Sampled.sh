#!/usr/bin/env bash
# Stop_First_Error_with_History_FP16_Sampled.sh -- same idea as
# Stop_First_Error_with_History_FP16.sh (histogram trace to the first
# exception), but with SAMPLING=128 added on top of the "(record_inst)"
# bin spec, to cut the per-occurrence trace down to a short, readable
# dribble of instructions instead of 11,000+ lines.
#
# Flags used (see the report for the full list and explanation):
#   SAMPLING=128         -- instrument only 1 in 128 kernel launches.
#   HISTOGRAM=1 + BIN_SPEC_FILE=Stop_First_Error_with_History_FP16_Sampled.json
#                         -- same [-14,-13]/[14,15] FP16 extreme-exponent
#                            ranges as before, using the print_histo_instrn
#                            branch's "f16 (record_inst)" key so each
#                            threshold hit also names the instruction.
#   MAX_ERRORS=1          -- stop the whole program at the first exception.
#
# Honest tradeoff, not hidden: with only 1 in 128 launches instrumented,
# the launch where FP16 actually first overflows (~step 300) usually isn't
# itself a sampled launch. The next sampled launch downstream sees the
# already-poisoned (NaN) values instead of catching the original Infinity
# -- the same completeness gap Concept 1 (Section 4/5) already documented
# for SAMPLING, showing up again here in a different form.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SAMPLING_N="${SAMPLING_N:-128}"
BIN_SPEC="$SCRIPT_DIR/Stop_First_Error_with_History_FP16_Sampled.json"
OUT="$LOGS_DIR/Stop_First_Error_with_History_FP16_Sampled.log"

echo "== Running rd_nixnan under nixnan (FP16 sampled record_inst trace, SAMPLING=$SAMPLING_N, MAX_ERRORS=1) -> $OUT =="
SAMPLING="$SAMPLING_N" HISTOGRAM=1 BIN_SPEC_FILE="$BIN_SPEC" MAX_ERRORS=1 LD_PRELOAD="$NIXNAN_SO" "$EXE" > "$OUT" 2>&1 || true

echo
echo "== Trace summary =="
echo "Total lines: $(wc -l < "$OUT")"
echo "Histogram threshold lines: $(grep -c 'bin has reached threshold' "$OUT" || true)"
echo "Exception lines printed: $(grep -c 'error \[' "$OUT" || true)"
echo
echo "== The dribble, in full =="
grep "bin has reached threshold\|error \[\|Terminating early" "$OUT" || true
