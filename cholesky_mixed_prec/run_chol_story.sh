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
#   ./run_chol_story.sh e5       random Q diag(sigma) Q^T matrices (kappa 1e2..1e7, 5 seeds) x five configurations; nixnan on chosen points
#   ./run_chol_story.sh e6       nixnan histogram "danger zone" view (healthy vs failing), cost of HISTOGRAM / SAMPLING / FUNCTION_WHITELIST / INSTR_MEM
#   ./run_chol_story.sh e7       explicit relative pivot guard (stop before the sqrt) vs nixnan on the same inputs
#   ./run_chol_story.sh sqrt     what does nixnan report for sqrt(0)? IEEE sqrtf (MUFU.RSQ sequence) vs -prec-sqrt=false (MUFU.SQRT)
#   ./run_chol_story.sh all      e0, e1, e2, e3, e4, sqrt, then the summary          (default)
#   ./run_chol_story.sh summary  re-print the tables from the saved logs
#
# Logs: logs/e0/, logs/e1/.   Env: NIXNAN_LIB=<nixnan.so> (default ../external/nixnan/nixnan.so)  ARCH=sm_86
# E0 also times the existing ../mpir_gpu/ir_gpu once (native and nixnan, NIXNAN_TIMEOUT=120) if it is built.

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NIXNAN_LIB="${NIXNAN_LIB:-$PWD/../external/nixnan/nixnan.so}"
ARCH="${ARCH:-sm_86}"
NATIVE_LIMIT=60; NIXNAN_LIMIT=120
mkdir -p bin_chol logs/e0 logs/e1 logs/e2 logs/e3 logs/e4 logs/e5 logs/e6 logs/e7 logs/sqrt
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

# pivot_sass <binary> <out>: the SASS of pivot_kernel (the sqrt), without the address arithmetic
pivot_sass() {
    cuobjdump -sass "$1" | awk '/Function : .*pivot_kernel/{p=1} /Function : .*scale_kernel/{p=0} p' \
      | grep -E "^\s+/\*[0-9a-f]{4}\*/" | sed -E 's/\s*\/\* 0x[0-9a-f]+ \*\/\s*$//; s/^\s+//' > "$2"
}

sqrtstudy() {
    echo "== sqrt: what does nixnan report for sqrt(0)? IEEE sqrtf vs -prec-sqrt=false (MUFU.SQRT)"
    build f32
    [[ -x bin_chol/chol_f32_fsqrt && bin_chol/chol_f32_fsqrt -nt chol_study.cu ]] || \
        nvcc -std=c++17 -g -O2 -lineinfo -arch="$ARCH" -DREAL=float -prec-sqrt=false -o bin_chol/chol_f32_fsqrt chol_study.cu || exit 1
    if [[ ! -x bin_chol/chol_tc ]]; then nvcc -std=c++17 -g -O2 -lineinfo -arch="$ARCH" -o bin_chol/chol_tc chol_tc.cu || exit 1; fi
    [[ -x bin_chol/chol_tc_fsqrt && bin_chol/chol_tc_fsqrt -nt chol_tc.cu ]] || \
        nvcc -std=c++17 -g -O2 -lineinfo -arch="$ARCH" -prec-sqrt=false -o bin_chol/chol_tc_fsqrt chol_tc.cu || exit 1
    rm -f logs/sqrt/*
    pivot_sass bin_chol/chol_f32 logs/sqrt/sass_pivot_ieee.txt
    pivot_sass bin_chol/chol_f32_fsqrt logs/sqrt/sass_pivot_fast.txt
    for b in ieee fast; do
        [[ $b == ieee ]] && f32=chol_f32 tc=chol_tc || { f32=chol_f32_fsqrt; tc=chol_tc_fsqrt; }
        for k in e3a e4; do
            if [[ $k == e3a ]]; then cmd=(./bin_chol/$f32 --log2delta -12 --store f16 --path unblocked); else cmd=(./bin_chol/$tc --fmt f16 --log2delta -12); fi
            "${cmd[@]}" > logs/sqrt/${k}_${b}_native.log 2>&1
            LD_PRELOAD="$NIXNAN_LIB" "${cmd[@]}" > logs/sqrt/${k}_${b}_nixnan.log 2>&1
            LD_PRELOAD="$NIXNAN_LIB" PRINT_ILL_INSTR=1 ENABLE_FUN_DETAIL=1 "${cmd[@]}" > logs/sqrt/${k}_${b}_detail.log 2>&1
            echo "   ${k}_$b"
        done
    done
}

# E6: nixnan's histogram ("danger zone") view and its cost. Scenarios (healthy delta=2^-6 vs failing delta=2^-12):
#   s1 chol_f32 --store f16 (FP32 arithmetic on FP16-rounded A)   s2 chol_tc --fmt f16 (tensor-core update)   s3 chol_h16 (true FP16)
# Bins: one per decade of binades around the pivot size; the zero bin (exponent field 0) carries (record_inst).
E6_COUNT=64
e6_specs() {
    python3 - "$E6_COUNT" <<'PY'
import json, sys
c = int(sys.argv[1])
f32 = [[-127, -127], [-126, -30], [-29, -14], [-13, -13], [-12, -12], [-11, -8], [-7, -1], [0, 0], [1, 10]]
f16 = [[-15, -15], [-14, -11], [-10, -8], [-7, -1], [0, 0], [1, 10]]
def spec(fmt, bins):
    d = {"count": c, "doublings": 0, "max_reports": 0}
    d[fmt] = [b for b in bins if b[0] != bins[0][0]]
    d[fmt + " (record_inst)"] = [bins[0]]
    return d
json.dump(spec("f32", f32), open("logs/e6/spec_f32.json", "w"), indent=1)
json.dump(spec("f16", f16), open("logs/e6/spec_f16.json", "w"), indent=1)
PY
}
e6() {
    echo "== E6: histogram danger-zone view (healthy delta=2^-6 vs failing delta=2^-12), and the cost of nixnan's options"
    build f32; build_half h16
    [[ -x bin_chol/chol_tc ]] || nvcc -std=c++17 -g -O2 -lineinfo -arch="$ARCH" -o bin_chol/chol_tc chol_tc.cu || exit 1
    mkdir -p logs/e6; rm -f logs/e6/*
    e6_specs
    local -A CMD=( [s1]="./bin_chol/chol_f32 --store f16 --path unblocked --log2delta" [s2]="./bin_chol/chol_tc --fmt f16 --log2delta" [s3]="./bin_chol/chol_h16 --log2delta" )
    local -A SPEC=( [s1]=f32 [s2]=f32 [s3]=f16 )
    for sc in s1 s2 s3; do for e in 6 12; do
        local tag=${sc}_e$e t0 t1
        t0=$(now)
        timeout 110 env HISTOGRAM=1 BIN_SPEC_FILE=logs/e6/spec_${SPEC[$sc]}.json LD_PRELOAD="$NIXNAN_LIB" ${CMD[$sc]} -$e > logs/e6/${tag}_stdout.log 2> logs/e6/${tag}_hist.log
        t1=$(now); echo "$(secs "$t0" "$t1")" > logs/e6/$tag.time
        echo "   $tag: $(grep -c 'reached threshold' logs/e6/${tag}_hist.log) threshold reports in $(cat logs/e6/$tag.time) s"
    done; done
    echo "-- cost of the options (s1, delta=2^-12, wall seconds)"
    echo "rank1_kernel(float*, int, int)" > logs/e6/whitelist.txt   # the full demangled name is matched
    local c="./bin_chol/chol_f32 --store f16 --path unblocked --log2delta -12" r
    run_opt() {   # run_opt <label> <env...>
        local label=$1; shift; local t0 t1; t0=$(now)
        timeout 110 env LD_PRELOAD="$NIXNAN_LIB" "$@" $c > /dev/null 2> logs/e6/opt_${label}.log; t1=$(now)
        echo "$label $(secs "$t0" "$t1")" >> logs/e6/options.tsv; echo "   $label $(secs "$t0" "$t1")"
    }
    : > logs/e6/options.tsv
    t0=$(now); $c > /dev/null 2>&1; t1=$(now); echo "native $(secs "$t0" "$t1")" >> logs/e6/options.tsv; echo "   native $(secs "$t0" "$t1")"
    run_opt report
    run_opt detail PRINT_ILL_INSTR=1 ENABLE_FUN_DETAIL=1
    run_opt histogram HISTOGRAM=1 BIN_SPEC_FILE=logs/e6/spec_f32.json
    run_opt hist_sampling4 HISTOGRAM=1 BIN_SPEC_FILE=logs/e6/spec_f32.json SAMPLING=4
    run_opt whitelist_rank1 FUNCTION_WHITELIST=logs/e6/whitelist.txt
    run_opt hist_whitelist HISTOGRAM=1 BIN_SPEC_FILE=logs/e6/spec_f32.json FUNCTION_WHITELIST=logs/e6/whitelist.txt
    run_opt instr_mem INSTR_MEM=1
}

build_guard() {   # E5/E7 binaries: the same kernels plus the relative-pivot statistic and the --guard option (-DGUARD)
    local o=(-std=c++17 -g -O2 -lineinfo -arch="$ARCH" -DGUARD)
    [[ -x bin_chol/g_chol_f32 && bin_chol/g_chol_f32 -nt chol_study.cu ]] || nvcc "${o[@]}" -DREAL=float -o bin_chol/g_chol_f32 chol_study.cu || exit 1
    [[ -x bin_chol/g_chol_tc && bin_chol/g_chol_tc -nt chol_tc.cu ]] || nvcc "${o[@]}" -o bin_chol/g_chol_tc chol_tc.cu || exit 1
    [[ -x bin_chol/g_chol_h16 && bin_chol/g_chol_h16 -nt chol_half.cu ]] || nvcc "${o[@]}" -o bin_chol/g_chol_h16 chol_half.cu || exit 1
}

# E5: random Q diag(sigma) Q^T matrices (kappa 1e2..1e7, seeds 0-4, n=256) through five configurations
E5_KAPPAS=(1e2 1e3 1e4 1e5 1e6 1e7); E5_SEEDS=(0 1 2 3 4)
e5_run() {   # e5_run <config> <csv>   (prints the RESULT line)
    case $1 in
        F32)       ./bin_chol/g_chol_f32 --csv "$2" --path blocked --store f32 ;;
        mixed-F16) ./bin_chol/g_chol_tc --csv "$2" --fmt f16 ;;
        mixed-BF16) ./bin_chol/g_chol_tc --csv "$2" --fmt bf16 ;;
        mixed-TF32) ./bin_chol/g_chol_tc --csv "$2" --fmt tf32 ;;
        naive-F16) ./bin_chol/g_chol_h16 --csv "$2" ;;
    esac
}
e5() {
    echo "== E5: random Q diag(sigma) Q^T, n=256, kappa 1e2..1e7, seeds 0-4, five configurations"
    build_guard
    mkdir -p matrices_q logs/e5; rm -f logs/e5/*
    : > logs/e5/matrices.txt; : > logs/e5/sweep.log
    for k in "${E5_KAPPAS[@]}"; do for s in "${E5_SEEDS[@]}"; do
        f=matrices_q/Q_n256_k${k}_s${s}.csv
        python3 gen_chol_matrices.py --n 256 --qlq "$k" --seed "$s" --out "$f" >> logs/e5/matrices.txt
        for c in F32 mixed-F16 mixed-BF16 mixed-TF32 naive-F16; do
            echo "CASE config=$c kappa=$k seed=$s $(e5_run $c "$f" | grep RESULT)" >> logs/e5/sweep.log
        done
    done; done
    echo "-- $(grep -c CASE logs/e5/sweep.log) native runs done"
    echo "-- nixnan on chosen points (default report + PRINT_ILL_INSTR trace)"
    python3 chol_summary.py --e5-points | while read -r c k s; do
        f=matrices_q/Q_n256_k${k}_s${s}.csv; tag=${c}_k${k}_s${s}
        [[ -f $f ]] || { echo "   missing $f"; continue; }
        LD_PRELOAD="$NIXNAN_LIB" TIME_KERNELS=1 bash -c "$(declare -f e5_run); e5_run $c $f" > logs/e5/${tag}_nixnan.log 2>&1
        LD_PRELOAD="$NIXNAN_LIB" PRINT_ILL_INSTR=1 ENABLE_FUN_DETAIL=1 bash -c "$(declare -f e5_run); e5_run $c $f" > logs/e5/${tag}_detail.log 2>&1
        echo "   $tag"
    done
}

# E7: detect instead of crash. Relative pivot guard (pivot / original diagonal <= tau, evaluated BEFORE the sqrt), tau = C_GUARD * u of the
# narrowest format fed to the arithmetic. Compared with nixnan's report on the same input.
C_GUARD=100
e7_u() { case $1 in F32) echo 5.96e-8 ;; mixed-F16|mixed-TF32|naive-F16) echo 4.883e-4 ;; mixed-BF16) echo 3.906e-3 ;; esac; }   # 2^-24, 2^-11, 2^-8
e7() {
    echo "== E7: relative pivot guard (tau = ${C_GUARD} u) vs nixnan, same inputs"
    build_guard
    [[ -f matrices_q/Q_n256_k1e2_s0.csv ]] || { echo "run ./run_chol_story.sh e5 first (it generates matrices_q/)"; exit 1; }
    mkdir -p logs/e7; rm -f logs/e7/*
    while read -r c k s; do
        local f=matrices_q/Q_n256_k${k}_s${s}.csv tag=${c}_k${k}_s${s} u tau
        u=$(e7_u $c); tau=$(awk -v u=$u -v c=$C_GUARD 'BEGIN{printf "%.4g", u*c}')
        e5_g() { case $1 in   # e5_g <config> <csv> <tau|0>
            F32) ./bin_chol/g_chol_f32 --csv "$2" --path blocked --store f32 --guard "$3" ;;
            mixed-F16) ./bin_chol/g_chol_tc --csv "$2" --fmt f16 --guard "$3" ;;
            mixed-BF16) ./bin_chol/g_chol_tc --csv "$2" --fmt bf16 --guard "$3" ;;
            mixed-TF32) ./bin_chol/g_chol_tc --csv "$2" --fmt tf32 --guard "$3" ;;
            naive-F16) ./bin_chol/g_chol_h16 --csv "$2" --guard "$3" ;; esac; }
        e5_g $c "$f" 0 > logs/e7/${tag}_plain_native.log 2>&1
        e5_g $c "$f" $tau > logs/e7/${tag}_guard_native.log 2>&1
        LD_PRELOAD="$NIXNAN_LIB" bash -c "$(declare -f e5_g); e5_g $c $f 0" > logs/e7/${tag}_plain_nixnan.log 2>&1
        LD_PRELOAD="$NIXNAN_LIB" bash -c "$(declare -f e5_g); e5_g $c $f $tau" > logs/e7/${tag}_guard_nixnan.log 2>&1
        echo "$tau" > logs/e7/${tag}.tau
        echo "   $tag tau=$tau"
    done <<'PTS'
F32 1e7 0
mixed-F16 1e2 0
mixed-F16 1e4 0
mixed-F16 1e6 0
mixed-BF16 1e2 0
mixed-TF32 1e3 0
naive-F16 1e3 0
naive-F16 1e4 0
PTS
    echo "-- guard on A_delta (E4 point: tensor core, FP16 panel, delta=2^-12), tau = ${C_GUARD} u"
    ./bin_chol/g_chol_tc --fmt f16 --log2delta -12 > logs/e7/Adelta_f16_e12_plain_native.log 2>&1
    ./bin_chol/g_chol_tc --fmt f16 --log2delta -12 --guard 4.883e-2 > logs/e7/Adelta_f16_e12_guard_native.log 2>&1
    LD_PRELOAD="$NIXNAN_LIB" ./bin_chol/g_chol_tc --fmt f16 --log2delta -12 > logs/e7/Adelta_f16_e12_plain_nixnan.log 2>&1
    LD_PRELOAD="$NIXNAN_LIB" ./bin_chol/g_chol_tc --fmt f16 --log2delta -12 --guard 4.883e-2 > logs/e7/Adelta_f16_e12_guard_nixnan.log 2>&1
    echo "-- held-out check of tau = c u: kappa between the E5 grid points (150..700), seeds 10-12, mixed-FP16 and naive-FP16, no guard"
    : > logs/e7/heldout.log
    for k in 150 200 300 500 700; do for sd in 10 11 12; do
        f=matrices_q/Q_n256_k${k}_s${sd}.csv
        python3 gen_chol_matrices.py --n 256 --qlq "$k" --seed "$sd" --out "$f" > /dev/null
        for c in mixed-F16 naive-F16; do echo "HELD config=$c kappa=$k seed=$sd $(e5_g $c "$f" 0 | grep RESULT)" >> logs/e7/heldout.log; done
    done; done
    echo "-- cost of the check (FP32 unblocked, M1 n=256: 256 host syncs; wall seconds, best of 3)"
    for g in 0 1e-6; do
        best=99; for i in 1 2 3; do t0=$(now); ./bin_chol/g_chol_f32 --csv matrices/M1_n256.csv --path unblocked --guard $g >/dev/null 2>&1; t1=$(now); best=$(awk -v a=$best -v b="$(secs "$t0" "$t1")" 'BEGIN{print (b<a)?b:a}'); done
        echo "guard_tau=$g $best" >> logs/e7/cost.tsv
    done
}

case "${1:-all}" in
    e0) e0 ;;
    e1) e1 ;;
    e2) e2 ;;
    e3) e3 ;;
    e4) e4 ;;
    e5) e5 ;;
    e6) e6 ;;
    e7) e7 ;;
    sqrt) sqrtstudy ;;
    summary) ;;
    all) e0; e1; e2; e3; e4; e5; e6; e7; sqrtstudy ;;
    *) sed -n 3,15p "$0"; exit 1 ;;
esac
echo
python3 chol_summary.py
