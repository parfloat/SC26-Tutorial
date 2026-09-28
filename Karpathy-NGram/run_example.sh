#!/usr/bin/env bash
#
# run_example.sh -- Karpathy-NGram/'s nixnan demo, standalone.
# Same command as the Karpathy-NGram section of ../All-Runs.md and the
# run_karpathy_ngram() function in ../run_all_examples.sh, but
# self-contained: runnable directly from inside this directory.
#
# Slow: ~30 minutes, dominated by NVBit's one-time enumeration of
# PyTorch's CUDA fatbinary (see ../autonixnan-doc.tex Sec 2.8 and this
# directory's entry in ../All-Runs.md for why).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

KSEL_AUTONIXNAN="$SCRIPT_DIR/../external/nixnan/bin/ksel-autonixnan"
if [[ ! -x "$KSEL_AUTONIXNAN" ]]; then
    echo "error: ksel-autonixnan not found at $KSEL_AUTONIXNAN" >&2
    exit 1
fi

mkdir -p Logs

echo "-- ksel-autonixnan, all 14 kernels whitelisted together, no sampling (~30 min) --"
"$KSEL_AUTONIXNAN" -ksel --ksel-file ./ksel -s 0 -t 2100 -- python3 gpt_finite_state_cuda.py \
    > Logs/ksel-autonixnan-cuda-allkernels-nosampling.log 2>&1

echo "Done. See Logs/ for output."
