#!/usr/bin/env bash
#
# run_chol_story.sh -- Cholesky mixed-precision study (see EXPERIMENT_PLAN.md). Implemented so far: E0, E1 (on M0 and M1), E2, E3 and E4.
#
#   ./run_chol_story.sh e0       budget: wall time of every planned run, native and under nixnan (limits: 60 s / 120 s)
#   ./run_chol_story.sh e1       reference: M0 and M1 (n=256) factor in FP64 and FP32; errors and nixnan counters
#   ./run_chol_story.sh e2       where does each storage format / precision break? delta sweep, CPU cross-check, nixnan on chosen points
#   ./run_chol_story.sh e3       naive reduced precision: FP16/BF16 storage (FP32 arithmetic) and true FP16/BF16 arithmetic;
#                                nixnan report + per-instruction trace (PRINT_ILL_INSTR) at the failing and the silent points
#   ./run_chol_story.sh e4       blocked Cholesky, FP32 panel, TENSOR-CORE trailing update (WMMA FP16 / BF16 / TF32, FP32 accumulate)
#   ./run_chol_story.sh all      e0, e1, e2, e3, e4, then the summary          (default)
#   ./run_chol_story.sh summary  re-print the tables from the saved logs
#
# Logs: logs/e0/, logs/e1/.   Env: NIXNAN_LIB=<nixnan.so> (default ../external/nixnan/nixnan.so)  ARCH=sm_86
# E0 also times the existing ../mpir_gpu/ir_gpu once (native and nixnan, NIXNAN_TIMEOUT=120) if it is built.

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NIXNAN_LIB="${NIXNAN_LIB:-$PWD/../external/nixnan/nixnan.so}"
ARCH="${ARCH:-sm_86}"
NATIVE_LIMIT=60; NIXNAN_LIMIT=120
mkdir -p bin_chol logs/e0 logs/e1 logs/e2 logs/e3 logs/e4
MATS=(M0 M1)   # from gen_chol_matrices.py (matrices/matrices.tsv has their definitions and kappa)
unset LD_LIBRARY_PATH   # a stale CUDA dir here can shadow the toolkit nvcc built against
[[ -f "$NIXNAN_LIB" ]] || { echo "error: nixnan.so not found at $NIXNAN_LIB (set NIXNAN_LIB)" >&2; exit 1; }

build() {   # build <f64|f32>
    local real=double; [[ $1 == f32 ]] && real=float
    local exe=bin_chol/chol_$1
    if [[ ! -x $exe || chol_study.cu -nt $exe ]]; then
        echo "-- building $exe (REAL=$real)"
        nvcc -std=c++17 -O2 -lineinfo -arch="$ARCH" -DREAL=$real -o "$exe" chol_study.cu || exit 1
    fi
}

[[ -f matrices/M1_n256.csv && -f matrices/M0_n64.csv ]] || python3 gen_chol_matrices.py >/dev/null

build_half() {   # build_half <h16|hbf16>
    local exe=bin_chol/chol_$1 def=""; [[ $1 == hbf16 ]] && def=-DUSE_BF16
    if [[ ! -x $exe || chol_half.cu -nt $exe ]]; then
        echo "-- building $exe"; nvcc -std=c++17 -O2 -lineinfo -arch="$ARCH" $def -o "$exe" chol_half.cu || exit 1
    fi
}

now() { date +%s.%N; }
secs() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.1f", b-a}'; }

# run_pair <dir> <tag> <exe> <args...>: native run then nixnan run; wall times to <dir>/<tag>.time
run_pair() {
    local dir=$1 tag=$2; shift 2
    local t0 t1 t2
    t0=$(now); "$@" > "$dir/${tag}_native.log" 2>&1; t1=$(now)
    LD_PRELOAD="$NIXNAN_LIB" TIME_KERNELS=1 "$@" > "$dir/${tag}_nixnan.log" 2>&1; t2=$(now)
    echo "$(secs "$t0" "$t1") $(secs "$t1" "$t2")" > "$dir/$tag.time"
}

e0() {
    echo "== E0: budget (native < ${NATIVE_LIMIT} s, nixnan < ${NIXNAN_LIMIT} s), FP32"
    build f32
    : > logs/e0/budget.tsv
    printf '%-26s %9s %9s %8s %s\n' run native_s nixnan_s kernels verdict
    for m in "${MATS[@]}"; do for n in 64 256; do for p in unblocked blocked; do
        tag=${m}_n${n}_f32_${p}
        run_pair logs/e0 "$tag" ./bin_chol/chol_f32 --csv matrices/${m}_n$n.csv --path $p
        read -r nat nix < logs/e0/$tag.time
        k=$(grep -o "running kernel \[[^]]*\]" logs/e0/${tag}_nixnan.log | sort -u | wc -l)
        v=PASS; awk -v a="$nat" -v b="$nix" -v A=$NATIVE_LIMIT -v B=$NIXNAN_LIMIT 'BEGIN{exit !(a<A && b<B)}' || v=OVER-BUDGET
        printf '%-26s %9s %9s %8s %s\n' "$tag" "$nat" "$nix" "$k" "$v" | tee -a logs/e0/budget.tsv
    done; done; done
    if [[ -x ../mpir_gpu/ir_gpu ]]; then
        echo "-- reference: existing ../mpir_gpu/ir_gpu (cuSOLVER/cuBLAS), n=256 kappa=1e3 F32"
        (cd ../mpir_gpu
         t0=$(now); timeout $NATIVE_LIMIT ./ir_gpu --n 256 --kappa 1e3 --config F32 > ../cholesky_mixed_prec/logs/e0/mpir_gpu_native.log 2>&1; rc=$?; t1=$(now)
         LD_PRELOAD="$NIXNAN_LIB" NIXNAN_TIMEOUT=$NIXNAN_LIMIT ./ir_gpu --n 256 --kappa 1e3 --config F32 > ../cholesky_mixed_prec/logs/e0/mpir_gpu_nixnan.log 2>&1; t2=$(now)
         echo "mpir_gpu/ir_gpu: native $(secs "$t0" "$t1") s (rc=$rc), nixnan $(secs "$t1" "$t2") s" | tee ../cholesky_mixed_prec/logs/e0/mpir_gpu.time)
    fi
}

e1() {
    echo "== E1: reference, M0 and M1, n=256"
    for c in f64 f32; do build $c; done
    for m in "${MATS[@]}"; do for c in f64 f32; do for p in unblocked blocked; do
        tag=${m}_${c}_$p
        ./bin_chol/chol_$c --csv matrices/${m}_n256.csv --path $p > logs/e1/${tag}_native.log 2>&1
        LD_PRELOAD="$NIXNAN_LIB" ./bin_chol/chol_$c --csv matrices/${m}_n256.csv --path $p > logs/e1/${tag}_nixnan.log 2>&1
    done; done; done
}

e2() {
    echo "== E2: delta = 2^-e sweep, A_delta n=256 (kappa = (n+delta)/delta), native (fast: one process per precision)"
    for c in f64 f32; do build $c; done
    ./bin_chol/chol_f32 --sweep 2 26 --stores f32,tf32,f16,bf16 --paths unblocked,blocked > logs/e2/sweep_f32.log 2>&1
    ./bin_chol/chol_f64 --sweep 2 56 --stores f64 --paths unblocked,blocked > logs/e2/sweep_f64.log 2>&1
    echo "-- CPU cross-check (LAPACK float32 on the rounded matrix)"
    python3 e2_emulate.py 2 26 > /dev/null
    echo "-- nixnan on chosen points: delta = 2^-12 (matrix M1) and each storage format's first failing delta, both paths"
    rm -f logs/e2/confirm_*.log
    python3 chol_summary.py --e2-points | while read -r store e path; do
        tag=${store}_e${e}_${path}
        ./bin_chol/chol_f32 --log2delta -$e --store $store --path $path > logs/e2/confirm_${tag}_native.log 2>&1
        LD_PRELOAD="$NIXNAN_LIB" ./bin_chol/chol_f32 --log2delta -$e --store $store --path $path > logs/e2/confirm_${tag}_nixnan.log 2>&1
        echo "   $tag"
    done
}

# e3_run <tag> <cmd...>: native, nixnan default report, nixnan with PRINT_ILL_INSTR=1 ENABLE_FUN_DETAIL=1
e3_run() {
    local tag=$1; shift
    "$@" > logs/e3/${tag}_native.log 2>&1
    LD_PRELOAD="$NIXNAN_LIB" "$@" > logs/e3/${tag}_nixnan.log 2>&1
    LD_PRELOAD="$NIXNAN_LIB" PRINT_ILL_INSTR=1 ENABLE_FUN_DETAIL=1 "$@" > logs/e3/${tag}_detail.log 2>&1
    echo "   $tag"
}

e3() {
    echo "== E3: naive reduced precision (A_delta, n=256)"
    build f32; build_half h16; build_half hbf16
    rm -f logs/e3/*
    echo "-- E3a: FP16 / BF16 storage, FP32 arithmetic, delta = 2^-12 (matrix M1), unblocked and blocked"
    for st in f16 bf16; do for p in unblocked blocked; do
        e3_run a_${st}_e12_$p ./bin_chol/chol_f32 --log2delta -12 --store $st --path $p
    done; done
    echo "-- E3b: true FP16 / BF16 arithmetic (every operation rounds to the narrow format), unblocked"
    ./bin_chol/chol_h16 --sweep 2 14 > logs/e3/sweep_h16.log 2>&1
    ./bin_chol/chol_hbf16 --sweep 2 14 > logs/e3/sweep_hbf16.log 2>&1
    python3 chol_summary.py --e3-points | while read -r exe e kind; do
        e3_run b_${exe#chol_}_e${e}_$kind ./bin_chol/$exe --log2delta -$e
    done
}

e4() {
    echo "== E4: blocked Cholesky with a tensor-core trailing update (A_delta, n=256, NB=32)"
    if [[ ! -x bin_chol/chol_tc || chol_tc.cu -nt bin_chol/chol_tc ]]; then
        echo "-- building bin_chol/chol_tc"; nvcc -std=c++17 -g -O2 -lineinfo -arch="$ARCH" -o bin_chol/chol_tc chol_tc.cu || exit 1
    fi
    rm -f logs/e4/*
    cuobjdump -sass bin_chol/chol_tc | grep -oE "HMMA[.A-Z0-9_]*" | sort | uniq -c > logs/e4/sass_hmma.txt
    echo "-- SASS tensor-core instructions:"; sed 's/^/   /' logs/e4/sass_hmma.txt
    ./bin_chol/chol_tc --sweep 2 26 --fmts f16,bf16,tf32 > logs/e4/sweep_tc.log 2>&1
    echo "-- nixnan on chosen points (native, default report, PRINT_ILL_INSTR trace)"
    python3 chol_summary.py --e4-points | while read -r fmt e kind; do
        tag=${fmt}_e${e}_$kind
        ./bin_chol/chol_tc --fmt $fmt --log2delta -$e > logs/e4/${tag}_native.log 2>&1
        LD_PRELOAD="$NIXNAN_LIB" ./bin_chol/chol_tc --fmt $fmt --log2delta -$e > logs/e4/${tag}_nixnan.log 2>&1
        LD_PRELOAD="$NIXNAN_LIB" PRINT_ILL_INSTR=1 ENABLE_FUN_DETAIL=1 ./bin_chol/chol_tc --fmt $fmt --log2delta -$e > logs/e4/${tag}_detail.log 2>&1
        echo "   $tag"
    done
}

case "${1:-all}" in
    e0) e0 ;;
    e1) e1 ;;
    e2) e2 ;;
    e3) e3 ;;
    e4) e4 ;;
    summary) ;;
    all) e0; e1; e2; e3; e4 ;;
    *) sed -n 3,11p "$0"; exit 1 ;;
esac
echo
python3 chol_summary.py
