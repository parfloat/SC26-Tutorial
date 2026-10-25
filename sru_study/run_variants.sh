#!/usr/bin/env bash
#
# run_variants.sh -- test whether the NaNs in sru#193 come from the uninitialized input tensor.
#   reproducer.py         x = torch.FloatTensor(20,32,128).cuda()          (uninitialized; logs in Logs_uninit_bug/)
#   reproducer_init0.py   x = torch.zeros(...)                              -> Log_init0/
#   reproducer_normal.py  x = torch.normal(0, 1, size=(20,32,128), cuda)    -> Log_normal/
# Same two nixnan modes as run_example.sh: iter-autonixnan (no sampling) and straight LD_PRELOAD.
# No `timeout` wrapper around the straight run (see ../All-Runs.md: nixnan can swallow SIGTERM).

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NIXNAN_LIB="$PWD/../external/nixnan/nixnan.so"
ITER="$PWD/../external/nixnan/bin/iter-autonixnan"

for v in init0 normal; do
    dir="Log_$v"; mkdir -p "$dir"
    echo "== $v: native =="
    python3 "reproducer_$v.py" > "$dir/native.log" 2>&1
    echo "== $v: iter-autonixnan =="
    ( time "$ITER" -s 0 -t 600 -- python3 "reproducer_$v.py" ) > "$dir/iter-autonixnan-nosampling.log" 2>&1
    echo "== $v: straight nixnan =="
    ( time env LD_PRELOAD="$NIXNAN_LIB" python3 -u "reproducer_$v.py" ) > "$dir/straight-nixnan.log" 2>&1
done
echo done
