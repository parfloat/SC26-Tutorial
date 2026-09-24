#!/usr/bin/env bash
# Exp-Output-Rate-start-1-double-8-times.sh -- exponentially decaying
# REPORT RATE, using nixnan's own native BIN_SPEC_FILE flags (not our
# hand-rolled multi-process SAMPLING orchestration from
# 03_sampling_exp_decay.sh -- see that script's header and report.tex's
# "NixNan Feature Reference" section for why SAMPLING itself has no such
# native schedule).
#
# IMPORTANT PROVENANCE NOTE: this "doublings" feature is a real nixnan
# flag, but it did NOT exist on the auto-nixnan branch this experiment has
# been built on (nor on print_histo_instrn, which branches from it) --
# only on a separate branch, fp-reset. It has been ported into
# print_histo_instrn (preserving record_inst) as part of writing this
# script; see external/nixnan/src/fp-histogram.{cu,cuh} and fp-fun.cu.
# The port also fixes a real concurrency bug present in the fp-reset
# original: multiple GPU threads can observe the same stale threshold
# value at once and race to double/reset it non-atomically. This version
# gates both transitions with atomicCAS so exactly one thread performs
# each doubling or reset.
#
# What "doublings" does (BIN_SPEC_FILE, see the JSON alongside this
# script): a bin starts reporting every "count" occurrences. Each time it
# reports, its own threshold doubles. After it has doubled "doublings"
# times, the NEXT report resets every bin (all formats, in that kernel)
# back to "count" and 0 doublings, and the cycle repeats. With count:1 and
# doublings:8, the reported sequence is 1, 2, 4, 8, 16, 32, 64, 128, 256,
# then reset back to 1 -- exactly 8 doubling steps from 1 to 256.
#
# No SAMPLING or MAX_ERRORS here: the point of this script is the report
# schedule itself, not stopping at a particular exception (see
# Stop_First_Error_with_History_FP16*.sh for that).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BIN_SPEC="$SCRIPT_DIR/Exp-Output-Rate-start-1-double-8-times.json"
OUT="$LOGS_DIR/Exp-Output-Rate-start-1-double-8-times.log"

echo "== Running rd_nixnan under nixnan (exponentially decaying report rate, count=1 doublings=8) -> $OUT =="
HISTOGRAM=1 BIN_SPEC_FILE="$BIN_SPEC" LD_PRELOAD="$NIXNAN_SO" "$EXE" > "$OUT" 2>&1

echo
echo "== First 30 reported threshold values (in order) =="
echo "   Monotonically non-decreasing within a cycle, 1->256, then back to 1."
echo "   Repeats at one level are expected: several grid points can genuinely"
echo "   cross the same threshold concurrently before the next doubling wins."
grep "bin has reached threshold" "$OUT" | grep -oP 'count=\K[0-9]+' | head -30 | paste -sd' '
echo
echo "== All 9 threshold levels seen across the whole run (should be 1..256) =="
grep "bin has reached threshold" "$OUT" | grep -oP 'count=\K[0-9]+' | sort -n | uniq -c
