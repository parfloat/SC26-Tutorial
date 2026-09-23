#!/usr/bin/env bash
# 05_extended_run_all_fail.sh -- make BF16 and FP32 fail too, not just FP16.
#
# Domain enlargement does NOT do this: the reaction term lambda*u[i] is a
# per-point multiplicative growth factor, independent of N/dx (already
# verified earlier this session with rd_nixnan_large.cu, a 10x-domain
# variant that still only failed in FP16 at M=1600). What actually works is
# running more time steps: at M=2000 (vs. the default 1600), FP16 still
# fails earliest (step 300), but BF16 and FP32 now also overflow, together,
# around step 1900.
#
# This builds a separate M=2000 binary (rd_nixnan_extended) rather than
# changing the default rd_nixnan build, so every other stage in this
# experiment keeps running against the original M=1600 baseline.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

EXTENDED_M="${EXTENDED_M:-2000}"
EXTENDED_EXE="$RD_NIXNAN_DIR/rd_nixnan_extended"

echo "== Building rd_nixnan_extended (M=$EXTENDED_M) =="
nvcc -arch=sm_86 -lineinfo -DM="$EXTENDED_M" "$RD_NIXNAN_DIR/rd_nixnan.cu" -o "$EXTENDED_EXE"

OUT="$LOGS_DIR/05_extended_M${EXTENDED_M}.log"
echo "== Running rd_nixnan_extended under nixnan -> $OUT =="
LD_PRELOAD="$NIXNAN_SO" "$EXTENDED_EXE" > "$OUT" 2>&1

echo
echo "== FP16/BF16/FP32/FP64 exception summary (M=$EXTENDED_M) =="
grep -A6 "FP16 Operations\|BF16 Operations\|FP32 Operations\|FP64 Operations" "$OUT" || true
echo
echo "NOTE: FP64's own printf check (grep below) also reports going non-finite,"
echo "but nixnan's exception classifier reports zero NaN/Infinity for FP64 above"
echo "-- a real, observed gap, not fixed by more margin (also checked at M=2500)."
grep "=====\|first non-finite" "$OUT" || true
