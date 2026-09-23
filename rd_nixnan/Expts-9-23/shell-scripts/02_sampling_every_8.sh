#!/usr/bin/env bash
# 02_sampling_every_8.sh -- "show sampling", level 2: SAMPLING=8, i.e. only
# every 8th launch of each repeated kernel is instrumented. Confirmed
# functional in the current nixnan build (verified empirically before
# writing this experiment -- see the "sampling regression" note in
# latex/report.tex: an earlier build had SAMPLING as a documented no-op).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

OUT="$LOGS_DIR/02_sampling_every_8.log"
echo "== SAMPLING=8 (every 8th kernel call instrumented) -> $OUT =="
SAMPLING=8 LD_PRELOAD="$NIXNAN_SO" "$EXE" > "$OUT" 2>&1
grep -A6 "FP16 Operations\|FP32 Operations\|BF16 Operations\|FP64 Operations" "$OUT" || true
