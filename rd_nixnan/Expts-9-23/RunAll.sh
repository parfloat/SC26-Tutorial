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
#   05  Extended run (M=2000, up from the default 1600): makes BF16 and
#       FP32 fail too, not just FP16. Domain enlargement does NOT do this
#       (the reaction term's growth is independent of N/dx) -- see
#       ExptPlan.md for the full reasoning and what was tried.
#   06  FP64, run for real until it too goes non-finite (step ~14600), after
#       fixing a double->float narrowing bug in rd_nixnan.cu's own
#       non-finite check that had been making step 05 misreport FP64 as
#       failing at step 1900. See ExptPlan.md and the report for the fix
#       and the diagnosis that led to it.
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
bash shell-scripts/05_extended_run_all_fail.sh
bash shell-scripts/06_fp64_longrun.sh

echo
echo "== All stages complete. Logs in $SCRIPT_DIR/Logs/ =="
