#!/usr/bin/env bash
# TinyGPT FP16 divergence under nixnan, stopped at the first NaN report (watch_nan.py).
#   pass 0: log the exact kernel names (dummy whitelist, nothing instrumented)
#   pass 1: exceptions in every kernel except tile-based GEMMs, flash attention and the warp
#           softmax, which compute on padding lanes and report NaN even in healthy step 1
#           (logs/false-positive-*.log)                          -> logs/nixnan-exceptions.log
#   pass 2: histogram bins 2^9..2^17 with instruction attribution on the FP16 residual add
#           and LayerNorm (whitelist: HISTOGRAM over all kernels overflows nixnan's
#           instruction table, logs/histogram-crash.log)       -> logs/nixnan-climb.log.xz
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=${PY:-$HOME/repos/newmed/venv/bin/python}
NN=${NN:-$(readlink -f ../../external/nixnan/nixnan.so)}
LR=${LR:-1}
mkdir -p logs
echo none > logs/dummy-whitelist.txt
names() { sed -n 's/^#nixnan: running kernel \[\(.*\)\] \.\.\.$/\1/p' logs/kernels-names.log | sort -u; }

timeout 900 env LD_PRELOAD=$NN FUNCTION_WHITELIST=$PWD/logs/dummy-whitelist.txt LOGFILE=$PWD/logs/kernels-names.log \
    "$PY" train_fp16.py --lr $LR > /dev/null 2>&1
names | grep -E "pytorch_flash|cutlass|^ampere_|cublasLt|softmax_warp" > blacklist.txt
names | grep -E "CUDAFunctor_add<c10::Half>|vectorized_layer_norm_kernel<c10::Half" > hist-whitelist.txt

timeout 3600 python3 watch_nan.py logs/nixnan-exceptions.log -- \
    env LD_PRELOAD=$NN FUNCTION_BLACKLIST=$PWD/blacklist.txt "$PY" train_fp16.py --lr $LR
timeout 3600 python3 watch_nan.py logs/nixnan-climb.log -- \
    env LD_PRELOAD=$NN FUNCTION_WHITELIST=$PWD/hist-whitelist.txt BIN_SPEC_FILE=$PWD/spec-climb.json HISTOGRAM=1 \
    "$PY" train_fp16.py --lr $LR
python3 summarize_climb.py logs/nixnan-climb.log logs/nixnan-exceptions.log > logs/SUMMARY.txt
xz -T0 -9 -f logs/nixnan-climb.log
"$PY" train_fp16.py --lr $LR --oracle > logs/oracle-lr$LR.txt 2>&1
cat logs/SUMMARY.txt
