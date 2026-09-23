#!/usr/bin/env bash
# 00_build.sh -- build nixnan.so (if needed) and the rd_nixnan target binary.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RD_NIXNAN_DIR="$(cd "$EXPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$RD_NIXNAN_DIR/.." && pwd)"

NIXNAN_SO="$REPO_ROOT/external/nixnan/nixnan.so"
if [[ ! -f "$NIXNAN_SO" ]]; then
  echo "== Building nixnan.so (external/nixnan) =="
  ( cd "$REPO_ROOT/external/nixnan" && make )
else
  echo "== nixnan.so already built at $NIXNAN_SO =="
fi

echo "== Building rd_nixnan (the 4-precision FP16/BF16/FP32/FP64 FTCS demo) =="
( cd "$RD_NIXNAN_DIR" && nvcc -arch=sm_86 rd_nixnan.cu -o rd_nixnan )
echo "== Build complete: $RD_NIXNAN_DIR/rd_nixnan =="
