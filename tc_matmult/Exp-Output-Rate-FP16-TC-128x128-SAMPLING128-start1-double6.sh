#!/usr/bin/env bash
# Exp-Output-Rate-FP16-TC-128x128-SAMPLING128-start1-double6.sh -- observe
# leimao-mm-128x128.cu's FP16 tensor-core (HMMA) kernel under nixnan,
# combining SAMPLING=128 with the native exponentially decaying
# report-rate mechanism (BIN_SPEC_FILE's count+doublings, see
# tc_matmult/report.tex and Exp-Output-Rate-start-1-double-8-times.sh
# under rd_nixnan/Expts-9-23/shell-scripts/ for the mechanism itself).
#
# 128x128x128, not 1024x1024x1024: the first attempt at this, against the
# original 1024^3 leimao-mm.cu, was killed after 10+ minutes with no
# output. That kernel executes on the order of 262,000 HMMA instructions
# per launch, and at BIN_SPEC_FILE's finest "count":1 granularity (report
# on every single occurrence), that is an overwhelming message volume
# through nixnan's channel. 128^3 is 1/512 the work -- ~512 HMMA
# executions per launch -- enough to make count:1 observation tractable.
#
# Schedule: count:1, doublings:6 -> report every 1st occurrence, then
# every 2nd, 4th, 8th, 16th, 32nd, 64th (6 doubling steps, 7 levels total:
# 1,2,4,8,16,32,64), then reset back to 1.
#
# Watched ranges, empirically chosen, not guessed: a plain HISTOGRAM=1 run
# (no BIN_SPEC_FILE) against this exact 128x128x128 kernel reports
# "Exponent range for f16: [-9, 8]" (narrower than the 1024^3 kernel's
# [zero, 8] -- fewer samples means less chance of hitting the rarest,
# near-zero/subnormal bin). So: [-9,-8] (the smallest two exponents this
# workload actually reaches) and [7,8] (the largest two) -- not this
# project's usual [-14,-13]/[14,15] "smallest/largest two of FP16's full
# range" convention, which this random-uniform(-256,256) workload never
# gets close to at either end.
#
# Honest note on SAMPLING=128 here: unlike rd_nixnan (1600+ kernel
# launches), leimao-mm-128x128.cu only launches its kernel ~88 times
# total (4 transpose combinations x (1 correctness check + 10 warmups +
# 10 timed repeats)). SAMPLING gates instrumentation by kernel_count %
# SAMPLING == 0 -- with only 88 launches, SAMPLING=128 only ever
# instruments launch #0 (the very first correctness check), and every one
# of measure_performance's 80 timed launches runs completely
# uninstrumented. This script still sets SAMPLING=128 exactly as asked;
# see report.tex for why this means the reported HMMA latency in this
# specific run does NOT show nixnan's real per-launch overhead, and for
# the separate, unsampled measurement that does.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NIXNAN_SO="$SCRIPT_DIR/../external/nixnan/nixnan.so"
EXE="$SCRIPT_DIR/leimao-mm-128x128"
BIN_SPEC="$SCRIPT_DIR/Exp-Output-Rate-FP16-TC-128x128-SAMPLING128-start1-double6.json"
LOGS_DIR="$SCRIPT_DIR/Logs"
OUT="$LOGS_DIR/Exp-Output-Rate-FP16-TC-128x128-SAMPLING128-start1-double6.log"
mkdir -p "$LOGS_DIR"

echo "== Building leimao-mm-128x128 (FP16 tensor core) =="
nvcc -arch=sm_86 -g -lineinfo "$SCRIPT_DIR/leimao-mm-128x128.cu" -o "$EXE"

echo "== Running under nixnan (SAMPLING=128, count=1 doublings=6) -> $OUT =="
SAMPLING=128 HISTOGRAM=1 BIN_SPEC_FILE="$BIN_SPEC" LD_PRELOAD="$NIXNAN_SO" "$EXE" > "$OUT" 2>&1

echo
echo "== Reported HMMA latencies in this (sampled) run =="
grep "Latency" "$OUT" || true
echo
echo "== Threshold levels seen, this run =="
grep "bin has reached threshold" "$OUT" | grep -oP 'count=\K[0-9]+' | sort -n | uniq -c
