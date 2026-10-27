#!/usr/bin/env bash
#
# run_all_examples.sh -- reproduce nixnan's headline finding for every case
# study in this tutorial, in one place. This is the shell-script form of
# All-Runs.md: same directories, same commands, same env vars, same log
# destinations (running this script for real will overwrite those logs).
#
# Usage:
#   ./run_all_examples.sh                 # run every case study, in order
#   ./run_all_examples.sh list            # list case study names
#   ./run_all_examples.sh <name> [<name> ...]   # run only the named studies
#
# Each case study is a self-contained function below; run only what you
# need. Several of these are slow the first time a fresh process touches a
# given CUDA kernel (NVBit's one-time fatbinary-enumeration cost, most
# visible with PyTorch/cuSOLVER kernels -- see autonixnan-doc.tex Sec 2.8
# and the Karpathy-NGram/pytorch_slogdet sections of All-Runs.md); the
# Karpathy-NGram study in particular takes ~30 minutes on its own.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NIXNAN_LIB="$SCRIPT_DIR/external/nixnan/nixnan.so"
AUTONIXNAN="$SCRIPT_DIR/external/nixnan/bin/autonixnan"
ITER_AUTONIXNAN="$SCRIPT_DIR/external/nixnan/bin/iter-autonixnan"
KSEL_AUTONIXNAN="$SCRIPT_DIR/external/nixnan/bin/ksel-autonixnan"

section() {
    echo
    echo "======================================================================"
    echo "$1"
    echo "======================================================================"
}

check_prereqs() {
    if [[ ! -e "$NIXNAN_LIB" ]]; then
        echo "error: nixnan.so not found at $NIXNAN_LIB (build external/nixnan first)" >&2
        exit 1
    fi
}

# ----------------------------------------------------------------------
# tc_matmult/ -- Lei Mao Tensor Core blog kernels + forced-overflow variants
# ----------------------------------------------------------------------
run_tc_matmult() {
    section "tc_matmult/"
    cd "$SCRIPT_DIR/tc_matmult"
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

    cd "$SCRIPT_DIR"
}

# ----------------------------------------------------------------------
# rd_nixnan/ -- reaction-diffusion PDE solve, FP16/BF16/FP32/FP64
# ----------------------------------------------------------------------
run_rd_nixnan() {
    section "rd_nixnan/"
    cd "$SCRIPT_DIR/rd_nixnan"
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

    cd "$SCRIPT_DIR"
}

# ----------------------------------------------------------------------
# lu_solve_gpu/ -- 20x20 LU factorization with partial pivoting
# ----------------------------------------------------------------------
run_lu_solve_gpu() {
    section "lu_solve_gpu/"
    cd "$SCRIPT_DIR/lu_solve_gpu"
    mkdir -p logs logs/iter-autonixnan

    echo "-- building --"
    make

    echo "-- autonixnan (upstream 3-phase wrapper -- misses bad_matrix.csv's real exceptions) --"
    "$AUTONIXNAN" -- ./lu_solve_gpu matrix.csv     > logs/autonixnan-matrix.log 2>&1
    "$AUTONIXNAN" -- ./lu_solve_gpu bad_matrix.csv > logs/autonixnan-bad_matrix.log 2>&1

    echo "-- iter-autonixnan (finds the real exceptions autonixnan's Phase 2 scan misses) --"
    "$ITER_AUTONIXNAN" -s 4 -- ./lu_solve_gpu matrix.csv \
        > logs/iter-autonixnan/sampling4-matrix.log 2>&1
    "$ITER_AUTONIXNAN" -s 8 -b bin_spec_highest.json -- ./lu_solve_gpu bad_matrix.csv \
        > logs/iter-autonixnan/sampling8-binspec_highest-bad_matrix.log 2>&1

    cd "$SCRIPT_DIR"
}

# ----------------------------------------------------------------------
# mpir_gpu/ -- mixed-precision (Carson-Higham) iterative refinement Cholesky
# ----------------------------------------------------------------------
run_mpir_gpu() {
    section "mpir_gpu/"
    cd "$SCRIPT_DIR/mpir_gpu"
    mkdir -p iter-autonixnan

    echo "-- building --"
    make

    echo "-- iter-autonixnan (finds real exceptions in geqr2_smem_domino_fast) --"
    "$ITER_AUTONIXNAN" -s 4 -t 90 -- ./ir_gpu --n 256 --kappa 100 --config F32 \
        > iter-autonixnan/sampling4-n256-kappa100-F32.log 2>&1

    cd "$SCRIPT_DIR"
}

# ----------------------------------------------------------------------
# hec_atan2/ -- HeCBench atan2-cuda, reduced-run local copy
# ----------------------------------------------------------------------
run_hec_atan2() {
    section "hec_atan2/"
    cd "$SCRIPT_DIR/hec_atan2"
    mkdir -p Log

    echo "-- building (MakeShort: 8 coordinates, repeat once) --"
    make -f MakeShort

    echo "-- default nixnan run (no sampling) --"
    LD_PRELOAD="$NIXNAN_LIB" ./main 8 1 \
        > Log/atan2-8x1-nixnan-default-exceptions.log 2>&1

    cd "$SCRIPT_DIR"
}

# ----------------------------------------------------------------------
# hec_fluidSim/ -- HeCBench fluidSim-cuda, deliberately-unstable local copy
# ----------------------------------------------------------------------
run_hec_fluidSim() {
    section "hec_fluidSim/"
    cd "$SCRIPT_DIR/hec_fluidSim"
    mkdir -p Log

    echo "-- building (MakeOverflow: omega=1e8, deliberately unstable) --"
    make -f MakeOverflow

    echo "-- default nixnan run, 60 iterations (no sampling) --"
    LD_PRELOAD="$NIXNAN_LIB" ./main_overflow 60 \
        > Log/fluidSim-overflow-60iter-nixnan-default-exceptions.log 2>&1

    cd "$SCRIPT_DIR"
}

# ----------------------------------------------------------------------
# pytorch_slogdet/ -- reproducer for pytorch/pytorch#173638
# ----------------------------------------------------------------------
run_pytorch_slogdet() {
    section "pytorch_slogdet/"
    cd "$SCRIPT_DIR/pytorch_slogdet"
    mkdir -p Logs

    echo "-- iter-autonixnan (SKIP_CPU=1 default, sampling 8, no histogram, exceptions only) --"
    "$ITER_AUTONIXNAN" -s 8 -t 60 -- python3 reproducer.py \
        > Logs/iter-autonixnan.log 2>&1

    cd "$SCRIPT_DIR"
}

# ----------------------------------------------------------------------
# Karpathy-NGram/ -- gpt_finite_state_cuda.py, single state transition
# ----------------------------------------------------------------------
run_karpathy_ngram() {
    section "Karpathy-NGram/ (slow: ~30 minutes, one-time CUDA fatbinary enumeration)"
    cd "$SCRIPT_DIR/Karpathy-NGram"
    mkdir -p Logs

    echo "-- ksel-autonixnan, all 14 kernels whitelisted together, no sampling --"
    "$KSEL_AUTONIXNAN" -ksel --ksel-file ./ksel -s 0 -t 2100 -- python3 gpt_finite_state_cuda.py \
        > Logs/ksel-autonixnan-cuda-allkernels-nosampling.log 2>&1

    cd "$SCRIPT_DIR"
}

# ----------------------------------------------------------------------
# sru_study/ -- reproducer for asappresearch/sru#193
# ----------------------------------------------------------------------
run_sru_study() {
    section "sru_study/"
    cd "$SCRIPT_DIR/sru_study"
    mkdir -p Logs_uninit_bug

    echo "-- iter-autonixnan (no sampling) --"
    "$ITER_AUTONIXNAN" -s 0 -t 600 -- python3 reproducer.py \
        > Logs_uninit_bug/iter-autonixnan-nosampling.log 2>&1

    echo "-- straight nixnan, all kernels together, no sampling (~79s, no timeout wrapper --"
    echo "   see All-Runs.md for why: nixnan can catch SIGTERM and exit 0 mid-run) --"
    LD_PRELOAD="$NIXNAN_LIB" python3 -u reproducer.py \
        > Logs_uninit_bug/straight-nixnan.log 2>&1

    cd "$SCRIPT_DIR"
}

# ----------------------------------------------------------------------
# hello_nixnan/ -- 58-line demo: exceptions in every precision, STG leaks, one tensor-core op
# ----------------------------------------------------------------------
run_hello_nixnan() {
    section "hello_nixnan/"
    cd "$SCRIPT_DIR/hello_nixnan"

    echo "-- building (needs sm_80+) --"
    nvcc -lineinfo -arch=native -o hello_nixnan hello_nixnan.cu

    echo "-- plain nixnan run (report printed on exit) --"
    LD_PRELOAD="$NIXNAN_LIB" ./hello_nixnan 2>&1 | tee hello_nixnan.report.run.log

    echo "-- per-instruction detail (PRINT_ILL_INSTR) -> exceptions.md, hello_nixnan.detail.log --"
    python3 collect_exceptions.py "$NIXNAN_LIB"

    cd "$SCRIPT_DIR"
}

# ----------------------------------------------------------------------

ALL_STUDIES=(hello_nixnan tc_matmult rd_nixnan lu_solve_gpu mpir_gpu hec_atan2 hec_fluidSim pytorch_slogdet karpathy_ngram sru_study)

usage() {
    echo "Usage: $0 [list | <study> [<study> ...]]"
    echo "Known studies: ${ALL_STUDIES[*]}"
    echo "No arguments runs every study, in order."
}

run_study() {
    case "$1" in
        hello_nixnan)     run_hello_nixnan ;;
        tc_matmult)       run_tc_matmult ;;
        rd_nixnan)        run_rd_nixnan ;;
        lu_solve_gpu)     run_lu_solve_gpu ;;
        mpir_gpu)         run_mpir_gpu ;;
        hec_atan2)        run_hec_atan2 ;;
        hec_fluidSim)     run_hec_fluidSim ;;
        pytorch_slogdet)  run_pytorch_slogdet ;;
        karpathy_ngram)   run_karpathy_ngram ;;
        sru_study)        run_sru_study ;;
        *)
            echo "error: unknown study '$1'" >&2
            usage
            exit 1
            ;;
    esac
}

main() {
    if [[ $# -gt 0 && "$1" == "list" ]]; then
        printf '%s\n' "${ALL_STUDIES[@]}"
        exit 0
    fi

    check_prereqs

    local studies=("${ALL_STUDIES[@]}")
    if [[ $# -gt 0 ]]; then
        studies=("$@")
    fi

    local start_time
    start_time=$(date +%s)

    for s in "${studies[@]}"; do
        run_study "$s"
    done

    local elapsed=$(( $(date +%s) - start_time ))
    section "Done: ${#studies[@]} case stud$([ ${#studies[@]} -eq 1 ] && echo y || echo ies) in ${elapsed}s"
}

main "$@"
