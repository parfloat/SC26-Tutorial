#!/usr/bin/env bash
# run_nixnan.sh -- every nixnan result in README.md, then logs/SUMMARY.md.
#
#   select   all 5 flash kernels (flash-kernels-all.txt), train.py --steps 3:
#            exception reports per kernel, used to choose the 3 captured kernels
#   smoke    smoke_flash.py: one flash call with a +inf key (inf/NaN inside flash)
#   gen312   generate.py: 312 -> 6 digits (prefill + 2 decode steps)
#   train3   train.py --steps 3 (forward + backward)
# smoke, gen312 and train3 instrument the 3 kernels of flash-kernels.txt.
# Each run costs ~2.5 min of NVBit start-up; two run at a time.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export SAMPLING=${SAMPLING:-50} TIMEOUT=${TIMEOUT:-1200}
T=$(mktemp -d)
mkdir -p logs
run() { local name=$1; shift; ./nixnan.sh "$name" "$@" > "logs/$name.stdout" 2>&1; }
LIST=$PWD/flash-kernels-all.txt run select train.py --steps 3 --out "$T/select.pt" &
run smoke smoke_flash.py
wait
run gen312 generate.py --prompt 312 &
run train3 train.py --steps 3 --out "$T/train3.pt"
wait
rm -rf "$T"
python3 summarize.py select smoke gen312 train3 > logs/SUMMARY.md
cat logs/SUMMARY.md
