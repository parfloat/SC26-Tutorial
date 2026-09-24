#!/usr/bin/env bash
# Exp-Output-Rate-FP32-Top2Exp-M2000-start-1-double-10-times.sh -- a run
# where FP32 overflows but FP64 does not, watching only FP32's highest two
# normal exponents (its own count-and-doubling schedule going 1 -> 1024,
# 10 doubling steps, then reset).
#
# Why M=2000 gives "FP32 overflows, FP64 doesn't": Section 8 of the report
# (Making BF16 and FP32 Fail Too) established that at M=2000, FP16 fails at
# step 300, BF16 and FP32 both fail at step 1900, and FP64 correctly stays
# finite through the whole run (it only overflows for real much later, at
# step 14600 -- see 06_fp64_longrun.sh). M=2000 is the smallest of the
# M values already used in this experiment that reproduces exactly the
# "FP32 fails, FP64 doesn't" split this run wants -- no new M value needed.
#
# FP32 is 8 exponent bits, bias 127: unbiased normal range is [-126,127],
# so its highest two exponents (just below the inf/nan field) are [126,127].
#
# count:1, doublings:10 means the reported threshold sequence is
# 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, then reset back to 1 --
# 10 doubling steps from 1 to 1024 (compare
# Exp-Output-Rate-start-1-double-8-times.sh, which went to 256 in 8 steps
# for FP16).
#
# No SAMPLING or MAX_ERRORS here, matching that script: the point is the
# report-rate schedule itself, applied to FP32 this time, not stopping at
# a particular exception.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

EXTENDED_M="${EXTENDED_M:-2000}"
EXTENDED_EXE="$RD_NIXNAN_DIR/rd_nixnan_extended"
BIN_SPEC="$SCRIPT_DIR/Exp-Output-Rate-FP32-Top2Exp-M2000-start-1-double-10-times.json"
OUT="$LOGS_DIR/Exp-Output-Rate-FP32-Top2Exp-M2000-start-1-double-10-times.log"

echo "== Building rd_nixnan_extended (M=$EXTENDED_M) =="
nvcc -arch=sm_86 -g -lineinfo -DM="$EXTENDED_M" "$RD_NIXNAN_DIR/rd_nixnan.cu" -o "$EXTENDED_EXE"

echo "== Running rd_nixnan_extended under nixnan (FP32 top-2-exponent doubling, count=1 doublings=10) -> $OUT =="
HISTOGRAM=1 BIN_SPEC_FILE="$BIN_SPEC" LD_PRELOAD="$NIXNAN_SO" "$EXTENDED_EXE" > "$OUT" 2>&1

echo
echo "== Did FP32 overflow and FP64 stay finite, as expected? =="
grep "=====\|first non-finite" "$OUT" || true
echo
echo "== First 30 reported threshold values (in order) =="
grep "bin has reached threshold" "$OUT" | grep -oP 'count=\K[0-9]+' | head -30 | paste -sd' '
echo
echo "== All 11 threshold levels seen across the whole run (should be 1..1024) =="
grep "bin has reached threshold" "$OUT" | grep -oP 'count=\K[0-9]+' | sort -n | uniq -c
