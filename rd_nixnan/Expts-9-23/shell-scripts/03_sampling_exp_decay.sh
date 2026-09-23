#!/usr/bin/env bash
# 03_sampling_exp_decay.sh -- "show sampling", level 3: exponentially
# decaying sampling rate driven by sampling_schedule.json (see that file's
# header comment for the full rationale). Runs one short "snap" of rd_nixnan
# per schedule entry, holding SAMPLING constant for that snap, and repeats
# the whole 1->2->4->8 pattern for `cycles` cycles to make the reset-to-1
# behavior directly visible in the logs.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

SCHEDULE_FILE="$SCRIPT_DIR/sampling_schedule.json"
OUT_DIR="$LOGS_DIR/03_sampling_exp_decay"
SUMMARY="$LOGS_DIR/03_sampling_exp_decay_summary.tsv"
mkdir -p "$OUT_DIR"

# Strip leading "//" comment lines, then parse with python3 (nixnan's own
# JSON reader is never involved with this file -- it's pure orchestration).
read -r SNAPS_PER_LEVEL LEVELS_CSV CYCLES < <(
  grep -v '^\s*//' "$SCHEDULE_FILE" | python3 -c '
import json, sys
d = json.load(sys.stdin)
print(d["snaps_per_level"], ",".join(map(str, d["levels"])), d["cycles"])
'
)
IFS=',' read -r -a LEVELS <<< "$LEVELS_CSV"

echo "== Exponential SAMPLING sweep (schedule: $SCHEDULE_FILE) =="
echo "   levels=${LEVELS[*]}  snaps_per_level=$SNAPS_PER_LEVEL  cycles=$CYCLES"

printf "cycle\tlevel\tsnap\tfp16_nan_unique\tfp16_nan_repeats\tfp16_inf_unique\tfp16_inf_repeats\n" > "$SUMMARY"

for (( cycle=1; cycle<=CYCLES; cycle++ )); do
  cycle_dir="$OUT_DIR/cycle_$(printf '%02d' "$cycle")"
  for level in "${LEVELS[@]}"; do
    level_dir="$cycle_dir/sampling_$level"
    mkdir -p "$level_dir"
    for (( snap=1; snap<=SNAPS_PER_LEVEL; snap++ )); do
      snap_pad="$(printf '%02d' "$snap")"
      log="$level_dir/snap_$snap_pad.log"
      SAMPLING="$level" LD_PRELOAD="$NIXNAN_SO" "$EXE" > "$log" 2>&1

      # pull FP16 (the only format that actually overflows in this
      # program's default M=1600 parameters -- see latex/report.tex)
      # NaN/Infinity unique+repeats for the summary.
      block=$(sed -n '/--- FP16 Operations ---/,/^$/p' "$log")
      nan_u=$(grep -oP 'NaN:\s*\K[0-9]+' <<<"$block" | head -1)
      nan_r=$(grep -oP 'NaN:.*\(\K[0-9]+' <<<"$block" | head -1)
      inf_u=$(grep -oP '^#nixnan: Infinity:\s*\K[0-9]+' <<<"$block" | head -1)
      inf_r=$(grep -oP '^#nixnan: Infinity:.*\(\K[0-9]+' <<<"$block" | head -1)
      printf "%d\t%s\t%d\t%s\t%s\t%s\t%s\n" \
        "$cycle" "$level" "$snap" "${nan_u:-0}" "${nan_r:-0}" "${inf_u:-0}" "${inf_r:-0}" >> "$SUMMARY"
    done
    echo "  cycle $cycle, SAMPLING=$level: $SNAPS_PER_LEVEL snaps -> $level_dir/"
  done
  echo "  cycle $cycle complete. Resetting SAMPLING back to ${LEVELS[0]}."
done

echo "== Summary written to $SUMMARY =="
column -t "$SUMMARY"
