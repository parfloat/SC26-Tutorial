#!/usr/bin/env bash
# finish_training.sh -- detached finisher for step 7 (run with setsid nohup).
# Waits for the nixnan-monitored training; if it died without saving (e.g. the
# launching terminal went away), reruns it; then records the result in
# README.md and commits + pushes the flash-attn-expt directory.
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=$HOME/repos/newmed/venv/bin/python
NN=$(readlink -f ../external/nixnan/nixnan.so)
OUT=logs/flash/train-sampling32
echo "[$(date)] finisher started"
while pgrep -f "python train.py --ou[t] tinygpt-nixnan.pt" >/dev/null; do sleep 30; done
if ! grep -q "^saved" $OUT.stdout; then
    echo "[$(date)] training did not finish; rerunning it"
    timeout 14400 env LD_PRELOAD=$NN FUNCTION_WHITELIST=$PWD/flash-kernels.txt SAMPLING=32 INSTR_MEM=1 \
        LOGFILE=$OUT.log "$PY" train.py --out tinygpt-nixnan.pt > $OUT.stdout 2>&1
fi
echo "[$(date)] training done: $(grep -E 'accuracy' $OUT.stdout)"
COMP=$("$PY" generate.py --ckpt tinygpt-nixnan.pt 2>/dev/null | sed 's/.*: //')
python3 summarize.py logs/flash/*.log > logs/SUMMARY.txt 2>&1
RESULT=$(python3 - "$OUT" "$COMP" <<'P'
import re, sys, subprocess
out, comp = sys.argv[1], sys.argv[2]
s = open(out + ".stdout").read()
loss = re.findall(r"^step\s+\d+\s+loss ([\d.]+)", s, re.M)
acc = re.search(r"accuracy ([\d.]+)", s)
summ = subprocess.run(["python3", "summarize.py", out + ".log"], capture_output=True, text=True).stdout
kern = "; ".join(l.strip() for l in summ.splitlines() if l.startswith("  flash") or l.startswith("  report"))
saved = "saved" in s
print(f"{'finished' if saved else 'DID NOT FINISH'}: final loss {loss[-1] if loss else '?'}, "
      f"next-token acc {acc.group(1) if acc else '?'}, `tinygpt-nixnan.pt` completes {comp}. {kern}")
P
)
python3 - "$RESULT" <<'P'
import sys
s = open("README.md").read()
s = s.replace("TRAIN_RESULT (training still running when this was committed)", sys.argv[1].replace("|", "/"))
open("README.md", "w").write(s)
P
cd .. && git add flash-attn-expt && git commit -q -m "flash-attn-expt: nixnan-monitored training (SAMPLING=32) result

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin Flash-Attn
echo "[$(date)] committed and pushed: rc=$? $(git log --oneline -1)"
