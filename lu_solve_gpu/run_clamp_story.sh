#!/usr/bin/env bash
#
# run_clamp_story.sh -- "the epsilon clamp": a silent failure, a defensive fix, a loud failure.
# Tutorial-time reproduction of the story on the slides (LU_Slides.pdf, "clamp story" slides).
#
#   ./run_clamp_story.sh step1     no clamp, FP64:  residual explodes, nixnan reports NOTHING
#   ./run_clamp_story.sh step2     clamp added, FP64: pivot becomes 0.0, NaN everywhere, nixnan finds it
#   ./run_clamp_story.sh fp32      the same two steps in FP32 (20x20)
#   ./run_clamp_story.sh n128      the same, FP64 and FP32, on 128x128 matrices (cond ~3e17)
#   ./run_clamp_story.sh all       everything above, then the summary table   (default)
#   ./run_clamp_story.sh summary   re-print the summary table from the saved logs
#
# Each run is made twice: natively (residual, pivot, solution) and under nixnan (LD_PRELOAD, no options).
# Logs: logs/clamp_story/<fp64|fp32>_clamp<on|off>_<matrix>_{native,nixnan}.log
# Env: NIXNAN_LIB=<path to nixnan.so>  (default ../external/nixnan/nixnan.so)   ARCH=sm_86

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NIXNAN_LIB="${NIXNAN_LIB:-$PWD/../external/nixnan/nixnan.so}"
ARCH="${ARCH:-sm_86}"
SRC=lu_solve_gpu_clamp_study.cu
BIN=bin_story
LOGS=logs/clamp_story
mkdir -p "$BIN" "$LOGS"
unset LD_LIBRARY_PATH   # a stale cuda dir here can shadow the toolkit nvcc built against

[[ -f "$NIXNAN_LIB" ]] || { echo "error: nixnan.so not found at $NIXNAN_LIB (set NIXNAN_LIB)" >&2; exit 1; }

build() {   # build <fp64|fp32> <on|off>
    local real=double clamp=1
    [[ $1 == fp32 ]] && real=float
    [[ $2 == off ]] && clamp=0
    local exe="$BIN/lu_$1_clamp$2"
    if [[ ! -x $exe || $SRC -nt $exe ]]; then
        echo "-- building $exe (REAL=$real CLAMP=$clamp)"
        nvcc -std=c++17 -g -O2 -lineinfo -arch="$ARCH" -DREAL=$real -DCLAMP=$clamp -o "$exe" "$SRC" || exit 1
    fi
}

run() {     # run <fp64|fp32> <on|off> <matrix-basename>
    build "$1" "$2"
    local exe="$BIN/lu_$1_clamp$2" tag="$1_clamp$2_$3"
    ./"$exe" "$3.csv" > "$LOGS/${tag}_native.log" 2>&1
    LD_PRELOAD="$NIXNAN_LIB" ./"$exe" "$3.csv" > "$LOGS/${tag}_nixnan.log" 2>&1
    printf '%-34s ' "$tag"
    grep -h -E "GPU pivots" "$LOGS/${tag}_native.log" | sed -E 's/GPU pivots: min \|U\[i\]\[i\]\| = ([^ ]+).*/pivot \1/' | tr '\n' ' '
    grep -h "Residual" "$LOGS/${tag}_native.log" | sed -E 's/Residual norm \|\|Ax - b\|\|: /residual /'
}

matrices_for() {  # make sure the 128x128 matrices exist (committed; regenerated if deleted)
    [[ -f bad_matrix_128.csv && -f matrix_128.csv ]] || python3 gen_matrix_n.py 128 1
}

step1() { echo "== step 1: no clamp (FP64)";   for m in matrix bad_matrix; do run fp64 off $m; done; }
step2() { echo "== step 2: clamp added (FP64)"; for m in matrix bad_matrix; do run fp64 on  $m; done
          echo "-- where does the first exception come from?  (PRINT_ILL_INSTR=1 ENABLE_FUN_DETAIL=1)"
          LD_PRELOAD="$NIXNAN_LIB" PRINT_ILL_INSTR=1 ENABLE_FUN_DETAIL=1 ./$BIN/lu_fp64_clampon bad_matrix.csv \
              > "$LOGS/fp64_clampon_bad_matrix_detail.log" 2>&1
          grep -m1 "error \[" "$LOGS/fp64_clampon_bad_matrix_detail.log" | sed 's/^#nixnan: //'
          echo "   exceptions per kernel:source-line:"
          grep "error \[" "$LOGS/fp64_clampon_bad_matrix_detail.log" | sed -E 's/.*in function (\w+) at .*\/([^\/]+:[0-9]+) of.*/\1 \2/' | sort | uniq -c | sed 's/^/     /'; }
fp32()  { echo "== FP32, 20x20";  for c in off on; do for m in matrix bad_matrix; do run fp32 $c $m; done; done; }
n128()  { matrices_for; echo "== 128x128";
          for p in fp64 fp32; do for c in off on; do for m in matrix_128 bad_matrix_128; do run $p $c $m; done; done; done; }

case "${1:-all}" in
    step1) step1 ;;
    step2) step2 ;;
    fp32)  fp32 ;;
    n128)  n128 ;;
    summary) ;;
    all)   step1; step2; fp32; n128 ;;
    *) sed -n 3,14p "$0"; exit 1 ;;
esac
echo
python3 story_summary.py "$LOGS"
