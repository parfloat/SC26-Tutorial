#!/usr/bin/env bash
# 01_sampling_every_call.sh -- "show sampling", level 1: instrument every
# single kernel call (SAMPLING unset/0, nixnan's default). This is the
# reference/baseline exception count everything else in this experiment is
# compared against.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

OUT="$LOGS_DIR/01_sampling_every_call.log"
echo "== SAMPLING unset (every kernel call instrumented) -> $OUT =="
LD_PRELOAD="$NIXNAN_SO" "$EXE" > "$OUT" 2>&1
grep -A6 "FP16 Operations\|FP32 Operations\|BF16 Operations\|FP64 Operations" "$OUT" || true
