#!/usr/bin/env python3
"""Run a command, copy its merged stdout/stderr to a log with elapsed-time stamps,
and kill it the moment a nixnan NaN error line appears.

    python watch_nan.py LOG -- cmd args...
"""
import os
import re
import signal
import subprocess
import sys
import time

log, cmd = sys.argv[1], sys.argv[sys.argv.index("--") + 1:]
NAN = re.compile(r"#nixnan: error \[[^\]]*NaN")
t0 = time.time()
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
                     text=True, errors="replace", bufsize=1)
with open(log, "w") as f:
    for line in p.stdout:
        f.write(f"[{time.time() - t0:8.1f}s] {line}")
        f.flush()
        if NAN.search(line):
            f.write(f"[{time.time() - t0:8.1f}s] WATCHER: first nixnan NaN report -- killing the run\n")
            f.flush()
            os.killpg(p.pid, signal.SIGKILL)
            break
p.wait()
print(f"exit {p.returncode}; log {log}")
