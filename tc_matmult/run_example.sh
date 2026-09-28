#!/usr/bin/env bash
#
# run_example.sh -- tc_matmult/'s nixnan demo, standalone.
# Same commands as the tc_matmult section of ../All-Runs.md and the
# run_tc_matmult() function in ../run_all_examples.sh, but self-contained:
# runnable directly from inside this directory.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NIXNAN_LIB="$SCRIPT_DIR/../external/nixnan/nixnan.so"
if [[ ! -e "$NIXNAN_LIB" ]]; then
    echo "error: nixnan.so not found at $NIXNAN_LIB (build external/nixnan first)" >&2
    exit 1
fi

mkdir -p Logs

echo "-- building --"
nvcc -arch=sm_86 -g -lineinfo leimao-mm-128x128.cu -o leimao-mm-128x128
nvcc -arch=sm_86 -g -lineinfo leimao-mm-128x128-overflow.cu -o leimao-mm-128x128-overflow
nvcc -arch=sm_86 -g -lineinfo leimao-simt-variant-fp16-128x128-overflow.cu -o leimao-simt-variant-fp16-128x128-overflow

echo "-- baseline (no nixnan) --"
./leimao-mm-128x128 > Logs/leimao-mm-128x128-baseline-no-nixnan.log 2>&1

echo "-- unsampled, default exception checking (128x128, never reaches NaN/Inf at this magnitude) --"
LD_PRELOAD="$NIXNAN_LIB" ./leimao-mm-128x128 \
    > Logs/leimao-mm-128x128-unsampled-default-exceptions.log 2>&1

echo "-- forced FP16 overflow: tensor-core __half accumulator --"
LD_PRELOAD="$NIXNAN_LIB" ./leimao-mm-128x128-overflow \
    > Logs/leimao-mm-128x128-overflow-nixnan.log 2>&1

echo "-- forced FP16 overflow: SIMT pure __half accumulate (no WMMA) --"
LD_PRELOAD="$NIXNAN_LIB" ./leimao-simt-variant-fp16-128x128-overflow \
    > Logs/leimao-simt-variant-fp16-128x128-overflow-nixnan.log 2>&1

echo "Done. See Logs/ for output."
