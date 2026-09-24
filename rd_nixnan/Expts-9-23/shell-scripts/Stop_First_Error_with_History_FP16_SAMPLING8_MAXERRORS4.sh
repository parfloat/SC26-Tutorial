#!/usr/bin/env bash
# Stop_First_Error_with_History_FP16_SAMPLING8_MAXERRORS4.sh -- answers the
# question the _Sampled.sh script's MAX_ERRORS=1/SAMPLING=128 combination
# raised: can raising MAX_ERRORS alone recover the original Infinity
# alongside the downstream NaN? No -- tested directly (see report.tex):
#   - Unsampled + MAX_ERRORS=2: two different Infinity sites, no NaN
#     until roughly site #10 (mixed) / #24 (pure NaN).
#   - SAMPLING=128 + MAX_ERRORS=2 (or any MAX_ERRORS): two different NaN
#     sites, Infinity never appears -- no instrumented launch ever
#     witnesses it, since the true overflow (~step 227) isn't a multiple
#     of 128. This is a stride problem, not a counting-budget problem.
#
# The actual fix is SAMPLING's stride, not MAX_ERRORS: SAMPLING=8 lands an
# instrumented launch on (or next to) the real overflow, so MAX_ERRORS=4
# is enough to surface both exception kinds -- though, honestly, even
# then they arrive combined in single messages tagged "NaN,infinity", not
# as a clean "Infinity line, then later a separate NaN line": different
# threads within the same launch are at different points of the buildup
# simultaneously, and nixnan ORs their exception bits together per
# instruction+operand before reporting.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SAMPLING_N="${SAMPLING_N:-8}"
MAX_ERRORS_N="${MAX_ERRORS_N:-4}"
BIN_SPEC="$SCRIPT_DIR/Stop_First_Error_with_History_FP16_SAMPLING8_MAXERRORS4.json"
OUT="$LOGS_DIR/Stop_First_Error_with_History_FP16_SAMPLING8_MAXERRORS4.log"

echo "== Running rd_nixnan under nixnan (FP16 record_inst trace, SAMPLING=$SAMPLING_N, MAX_ERRORS=$MAX_ERRORS_N) -> $OUT =="
SAMPLING="$SAMPLING_N" HISTOGRAM=1 BIN_SPEC_FILE="$BIN_SPEC" MAX_ERRORS="$MAX_ERRORS_N" LD_PRELOAD="$NIXNAN_SO" "$EXE" > "$OUT" 2>&1 || true

echo
echo "== Trace summary =="
echo "Total lines: $(wc -l < "$OUT")"
echo "Histogram threshold lines: $(grep -c 'bin has reached threshold' "$OUT" || true)"
echo "Exception lines printed: $(grep -c 'error \[' "$OUT" || true)"
echo
echo "== The exceptions caught =="
grep "error \[" "$OUT" || true
