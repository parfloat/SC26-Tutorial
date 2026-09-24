#!/usr/bin/env bash
# 06_fp64_longrun.sh -- run FP64 for real, until it too goes non-finite.
#
# Section 6/7 of the report found that FP64's earlier "step 1900" non-finite
# report was a bug in rd_nixnan.cu's own diagnostic check (an implicit
# double->float narrowing before isfinite()), not a real GPU event -- fixed
# in rd_nixnan.cu. This script demonstrates the real thing: FP64 does
# eventually overflow for real, just far later than FP16/BF16/FP32, and
# nixnan classifies it correctly once it happens.
#
# Build mode: -DFP64_LONGRUN=1 runs all four precisions together up to
# SUPPRESS_AFTER (2x the FP32 trigger step, 1900 -> 3800), then continues
# stepping FP64 alone (FP16/BF16/FP32 are already long non-finite by then)
# until FP64 itself goes non-finite or M_FP64_MAX is hit.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

LONGRUN_EXE="$RD_NIXNAN_DIR/rd_nixnan_fp64_longrun"

echo "== Building rd_nixnan_fp64_longrun (FP64_LONGRUN=1) =="
nvcc -arch=sm_86 -g -lineinfo -DFP64_LONGRUN=1 "$RD_NIXNAN_DIR/rd_nixnan.cu" -o "$LONGRUN_EXE"

OUT="$LOGS_DIR/06_fp64_longrun.log"
echo "== Running rd_nixnan_fp64_longrun under nixnan -> $OUT =="
LD_PRELOAD="$NIXNAN_SO" "$LONGRUN_EXE" > "$OUT" 2>&1

echo
echo "== FP16/BF16/FP32/FP64 exception summary (FP64_LONGRUN) =="
grep -A6 "FP16 Operations\|BF16 Operations\|FP32 Operations\|FP64 Operations" "$OUT" || true
echo
echo "== Where FP64's real overflow happens =="
grep "FP64 first non-finite\|suppressing\|Suppressing" "$OUT" || true
