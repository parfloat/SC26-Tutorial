#!/usr/bin/env bash
#
# run_example.sh -- rd_nixnan/'s nixnan demo, standalone.
# Same commands as the rd_nixnan section of ../All-Runs.md and the
# run_rd_nixnan() function in ../run_all_examples.sh, but self-contained:
# runnable directly from inside this directory.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ITER_AUTONIXNAN="$SCRIPT_DIR/../external/nixnan/bin/iter-autonixnan"
if [[ ! -x "$ITER_AUTONIXNAN" ]]; then
    echo "error: iter-autonixnan not found at $ITER_AUTONIXNAN" >&2
    exit 1
fi

mkdir -p iter_autonixnan

echo "-- building --"
nvcc -std=c++17 -g -O2 -lineinfo -arch=sm_86 -o reaction_diffusion_gpu_fp64 reaction_diffusion_gpu.cu
nvcc -std=c++17 -g -O2 -lineinfo -arch=sm_86 -DRD_USE_FLOAT -o reaction_diffusion_gpu_fp32 reaction_diffusion_gpu.cu
nvcc -std=c++17 -g -O2 -lineinfo -arch=sm_86 -DFP64_LONGRUN=1 -o rd_nixnan_fp64_longrun rd_nixnan.cu

echo "-- iter-autonixnan sweeps (no sampling, histogram on, bin_spec_highest_c2d4.json) --"
"$ITER_AUTONIXNAN" -s 0 -t 180 -b bin_spec_highest_c2d4.json -- ./reaction_diffusion_gpu_fp32 \
    > iter_autonixnan/sampling0-histogram-c2d4-fp32.log 2>&1
"$ITER_AUTONIXNAN" -s 0 -t 180 -b bin_spec_highest_c2d4.json -- ./reaction_diffusion_gpu_fp64 \
    > iter_autonixnan/sampling0-histogram-c2d4-fp64.log 2>&1
"$ITER_AUTONIXNAN" -s 0 -t 180 -b bin_spec_highest_c2d4.json -- ./rd_nixnan_fp64_longrun \
    > iter_autonixnan/sampling0-histogram-c2d4-fp64longrun.log 2>&1

echo "Done. See iter_autonixnan/ for output."
