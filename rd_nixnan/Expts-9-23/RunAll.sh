#!/usr/bin/env bash
# RunAll.sh -- top-level driver for the Expts-9-23 experiment.
#
# Subject: rd_nixnan.cu, the 4-precision (FP16/BF16/FP32/FP64) reaction-
# diffusion FTCS demo, run under "straight nixnan" (LD_PRELOAD=nixnan.so
# with env vars set directly -- no autonixnan wrapper).
#
# Stages:
#   00  Build nixnan.so (if needed) and rd_nixnan.
#   01  Sampling, level 1: every kernel call instrumented (baseline).
#   02  Sampling, level 2: SAMPLING=8 (every 8th call).
#   03  Sampling, level 3: exponentially decaying schedule
#       (1,1,1,1,2,2,2,2,4,4,4,4,8,8,8,8, then reset), per
#       sampling_schedule.json.
#   04  Histograms per precision bucket (HISTOGRAM=1).
#
# All raw output goes to Logs/. Re-running is safe/idempotent -- each stage
# overwrites its own output.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "########################################"
echo "# Expts-9-23: rd_nixnan sampling + histogram experiment"
echo "########################################"

bash shell-scripts/00_build.sh
bash shell-scripts/01_sampling_every_call.sh
bash shell-scripts/02_sampling_every_8.sh
bash shell-scripts/03_sampling_exp_decay.sh
bash shell-scripts/04_histograms_per_precision.sh

echo
echo "== All stages complete. Logs in $SCRIPT_DIR/Logs/ =="
