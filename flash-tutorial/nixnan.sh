#!/usr/bin/env bash
# nixnan.sh NAME SCRIPT.py [ARGS...] -- run a Python script under nixnan,
# instrumenting only PyTorch's flash-attention kernels.
#
#   ./nixnan.sh gen312 generate.py --prompt 312
#   SAMPLING=50 TIMEOUT=600 ./nixnan.sh train3 train.py --steps 3 --out /tmp/t.pt
#
# The kernels instrumented are the exact names listed in flash-kernels.txt
# (nixnan's FUNCTION_WHITELIST matches full demangled names): the three flash
# kernels around the softmax (README.md). nixnan's log names every kernel
# launched, instrumented or not, so after the run the flash kernels that ran
# without instrumentation are listed; LEARN=1 adds them to flash-kernels.txt.
#
# Cost: NVBit scans every function in PyTorch's GPU binaries as they load, a
# fixed ~2.5 min per process on an RTX 3090 whatever is instrumented. SAMPLING
# only reduces the (smaller) per-launch instrumentation cost.
#
# Environment:
#   NIXNAN_SO  nixnan.so          (default: ../external/nixnan/nixnan.so)
#   PY         python             (default: python3)
#   MATCH      grep -E regex      (default: pytorch_flash)
#   TIMEOUT    seconds            (default: 900; 0 = none)
#   LEARN=1    add launched-but-unlisted kernels matching $MATCH to flash-kernels.txt
#   ALL=1      instrument every kernel (much slower)
#   nixnan variables pass through: SAMPLING, INSTR_MEM, HISTOGRAM, BIN_SPEC_FILE, ...
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NIXNAN_SO=${NIXNAN_SO:-$HERE/../external/nixnan/nixnan.so}
PY=${PY:-python3}
MATCH=${MATCH:-pytorch_flash}
TIMEOUT=${TIMEOUT:-900}
LIST=$HERE/flash-kernels.txt
[ $# -ge 2 ] || { sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
[ -f "$NIXNAN_SO" ] || { echo "nixnan.so not found at $NIXNAN_SO (set NIXNAN_SO)"; exit 1; }
NIXNAN_SO=$(readlink -f "$NIXNAN_SO")
name=$1; shift
mkdir -p "$HERE/logs"
log=$HERE/logs/$name.log
touch "$LIST"
wl=(FUNCTION_WHITELIST="$LIST"); [ "${ALL:-0}" = 1 ] && wl=()

start=$(date +%s)
timeout "$TIMEOUT" env LD_PRELOAD="$NIXNAN_SO" "${wl[@]}" LOGFILE="$log" "$PY" "$@"
rc=$?
echo "nixnan: exit $rc after $(( $(date +%s) - start )) s; log: logs/$name.log"
[ $rc = 124 ] && echo "nixnan: TIMEOUT after $TIMEOUT s (raise TIMEOUT or SAMPLING)"

launched=$(sed -n 's/^#nixnan: running kernel \[\(.*\)\] \.\.\.$/\1/p' "$log" | sort -u)
echo "nixnan: $(grep -c . <<< "$launched") kernels launched, $(grep -cxFf "$LIST" <<< "$launched") instrumented," \
     "$(grep -c '#nixnan: error' "$log") exception reports"
new=$(grep -E "$MATCH" <<< "$launched" | grep -vxFf "$LIST")
if [ -n "$new" ] && [ "${ALL:-0}" != 1 ]; then
    echo "nixnan: $(grep -c . <<< "$new") launched kernel(s) matching '$MATCH' not instrumented:"
    sed -E 's/^void ([a-z_:]+)<.*/    \1/' <<< "$new"
    if [ "${LEARN:-0}" = 1 ]; then
        echo "$new" >> "$LIST"; sort -u -o "$LIST" "$LIST"; echo "nixnan: added to flash-kernels.txt"
    fi
fi
exit $rc
