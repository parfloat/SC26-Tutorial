#!/usr/bin/env bash
# 04_histograms_per_precision.sh -- "show the histograms per precision
# bucket": HISTOGRAM=1 turns on nixnan's exponent-histogram tracking for
# every instrumented FP instruction, regardless of format. At program exit
# it prints the [min,max] exponent range actually seen for each of
# bf16/f16/f32/f64 -- run here against all four kernels in rd_nixnan.cu in
# one pass.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

OUT="$LOGS_DIR/04_histograms_per_precision.log"
echo "== HISTOGRAM=1 (per-precision exponent ranges) -> $OUT =="
HISTOGRAM=1 LD_PRELOAD="$NIXNAN_SO" "$EXE" > "$OUT" 2>&1
grep -A5 "FP exponent ranges" "$OUT" || true
