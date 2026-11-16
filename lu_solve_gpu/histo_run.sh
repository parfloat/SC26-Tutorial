#!/usr/bin/env bash
#
# histo_run.sh -- run lu_solve_gpu under nixnan's exponent histogram with threshold doubling.
#
#   ./histo_run.sh <matrix.csv> [mode] [outdir]
#
#   mode   fp16acc32 (default) | fp16acc16 | bf16acc32   -> ./lu_solve_gpu_tc (tensor cores)
#          f64                                           -> ./lu_solve_gpu    (double only)
#   outdir default: logs/histo/<matrix>_<mode>
#
# Spec written to <outdir>/spec.json:
#   count = 64, doublings = 0: every (kernel, bin) reports each time it has been hit another 64 times
#   (the threshold is fixed; it does not double).
#   f16, bf16, f32: ONE BINADE PER BIN over the whole biased-exponent range, including the
#   zero/subnormal bin (unbiased -15 / -127) and the inf/NaN bin (16 / 128), so nothing in
#   those formats falls outside a bin. f64 (not the focus): bins of 8 binades.
# Env: NIXNAN_LIB (default ../external/nixnan/nixnan.so), COUNT (default 64), F64_WIDTH (default 8),
#      RECORD_INST=1 (adds the SASS instruction to every report), TMO (seconds, default 110).
# See README_histo_run.md.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

[[ $# -ge 1 && -f $1 ]] || { echo "usage: $0 <matrix.csv> [fp16acc32|fp16acc16|bf16acc32|f64] [outdir]" >&2; exit 2; }
MAT=$1; MODE=${2:-fp16acc32}
NIXNAN_LIB=${NIXNAN_LIB:-$(realpath ../external/nixnan/nixnan.so)}
[[ -f $NIXNAN_LIB ]] || { echo "nixnan.so not found: $NIXNAN_LIB (set NIXNAN_LIB)" >&2; exit 1; }
OUT=${3:-logs/histo/$(basename "$MAT" .csv)_$MODE}
mkdir -p "$OUT"

case $MODE in
    fp16acc32|fp16acc16|bf16acc32) make -s lu_solve_gpu_tc >/dev/null || exit 1; CMD=(./lu_solve_gpu_tc "$MAT" "$MODE") ;;
    f64)                           make -s lu_solve_gpu    >/dev/null || exit 1; CMD=(./lu_solve_gpu "$MAT") ;;
    *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac

python3 - "$OUT/spec.json" "${COUNT:-64}" 0 "${F64_WIDTH:-8}" "${RECORD_INST:-0}" <<'PY'
import json, sys
path, count, dbl, w64, rec = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] == "1"
def per_binade(lo, hi): return [[e, e] for e in range(lo, hi + 1)]
def wide(lo, hi, w):
    out, x = [], lo
    while x <= hi: out.append([x, min(x + w - 1, hi)]); x += w
    return out
sfx = " (record_inst)" if rec else ""
spec = {"count": count, "doublings": dbl, "max_reports": 0,
        "f16" + sfx:  per_binade(-15, 16),      # zero/subnormal, 30 normal binades, inf/NaN
        "bf16" + sfx: per_binade(-127, 128),
        "f32" + sfx:  per_binade(-127, 128),
        "f64" + sfx:  wide(-1023, 1024, w64)}
json.dump(spec, open(path, "w"), indent=1)
PY

echo "== ${CMD[*]}   (count=${COUNT:-64}, no doubling) -> $OUT"
timeout "${TMO:-110}" env HISTOGRAM=1 BIN_SPEC_FILE="$OUT/spec.json" LD_PRELOAD="$NIXNAN_LIB" \
    "${CMD[@]}" > "$OUT/stdout.log" 2> "$OUT/histo.log"
echo "exit status $? ; $(grep -c 'reached threshold' "$OUT/histo.log") threshold reports"

# summary: per format and kernel, how many threshold reports each bin produced
python3 - "$OUT/histo.log" "$OUT/summary.txt" <<'PY'
import re, sys, collections
rows = collections.defaultdict(lambda: collections.defaultdict(int))  # (fmt,kernel)->range->reports
rx = re.compile(r"#nixnan: (\S+(?: \(record_inst\))?) bin has reached threshold: function=(.*?) range=\[(\S+?),(\S+?)\] count=(\d+)")
for l in open(sys.argv[1], errors="replace"):
    m = rx.search(l)
    if m: rows[(m.group(1), m.group(2))][(m.group(3), m.group(4))] += 1
with open(sys.argv[2], "w") as f:
    for (fmt, k), d in sorted(rows.items()):
        f.write(f"{fmt}  {k}\n")
        for (lo, hi), n in d.items():
            f.write(f"   [{lo:>6},{hi:>6}]  reports {n:6d}\n")
print(open(sys.argv[2]).read() or "(no threshold reports)")
PY
