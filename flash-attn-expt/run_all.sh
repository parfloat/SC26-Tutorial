#!/usr/bin/env bash
# run_all.sh -- reproduce every flash-attn-expt nixnan run (README.md, PLAN.md).
#   PY : python with torch 2.6.0+cu124 and bitsandbytes (default: the BioMistral venv)
#   NN : nixnan.so (default: ../external/nixnan/nixnan.so, branch print_histo_instrn)
#
# Why whitelists: instrumenting all of PyTorch makes NVBit spend ~30 s per
# cuBLAS kernel, so the runs instrument only the kernels under study.
# FUNCTION_WHITELIST matches the full demangled name, so pass 1 runs every
# target with a dummy whitelist (nothing instrumented, a few seconds each)
# just to log the exact names nixnan sees.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=${PY:-$HOME/repos/newmed/venv/bin/python}
NN=${NN:-$(readlink -f ../external/nixnan/nixnan.so)}
K=logs/kernel-names; L=logs/flash
mkdir -p $K $L/sweep
echo none > $K/dummy-whitelist.txt
P127=$(python3 -c "import random;r=random.Random(5);print(''.join(r.choice('0123456789') for _ in range(127)))")

names() {   # names <tag> <python args...>: log kernel names, instrument nothing
    local tag=$1; shift
    timeout 600 env LD_PRELOAD=$NN FUNCTION_WHITELIST=$PWD/$K/dummy-whitelist.txt \
        LOGFILE=$K/$tag.log "$PY" "$@" > $K/$tag.stdout 2>&1
}
nixnan() {  # nixnan <name> <SAMPLING> <whitelist> <python args...>
    local name=$1 s=$2 wl=$3; shift 3
    timeout 7200 env LD_PRELOAD=$NN FUNCTION_WHITELIST=$PWD/$wl SAMPLING=$s INSTR_MEM=1 \
        LOGFILE=$L/$name.log "$PY" "$@" > $L/$name.stdout 2>&1
}
kernels() { cat "$@" | sed -n 's/^#nixnan: running kernel \[\(.*\)\] \.\.\.$/\1/p' | sort -u; }

# --- pass 1: kernel names
names generate generate.py
names lse experiment_lse.py --input A
names train train.py --steps 3 --out /tmp/flash-attn-expt-names.pt
names gen-nf4 generate.py --linear nf4
names gen-fp16 generate.py --linear fp16
names tile-T128 generate.py --prompt "$P127" --length 127        # even-tile kernel variant
kernels $K/*.log | grep pytorch_flash > flash-kernels.txt
{ cat flash-kernels.txt; kernels $K/gen-nf4.log | grep -E "kDequantize|kQuantize|kgemm_4bit"; } | sort -u > wl-nf4.txt
{ cat flash-kernels.txt; kernels $K/gen-fp16.log | grep -E "cutlass|gemvx|gemmSN|dot_kernel|reduce_1Block"; } | sort -u > wl-fp16.txt

# --- pass 2: monitored runs
nixnan train-sampling32 32 flash-kernels.txt train.py --out tinygpt-nixnan.pt   # step 7 (~1 h)
nixnan eval-312-full 0 flash-kernels.txt generate.py                           # step 8
for i in A B; do                                                               # step 9
    "$PY" experiment_lse.py --input $i --oracle > logs/lse-$i-oracle.txt 2>&1
    nixnan lse-$i-full 0 flash-kernels.txt experiment_lse.py --input $i
done
nixnan eval-312-full-nf4 0 wl-nf4.txt generate.py --linear nf4                 # step 10
nixnan eval-312-full-fp16 0 wl-fp16.txt generate.py --linear fp16
for T in 2 4 8 16 17 18 20 24 28 31 32 64 96 120 127 128; do                   # step 11
    n=$((T - 1)); nixnan sweep/T$T 0 flash-kernels.txt generate.py --prompt "${P127:0:$n}" --length $n
done
python3 summarize.py $L/*.log > logs/SUMMARY.txt
