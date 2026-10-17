#!/usr/bin/env python3
"""Check that this machine can run the tutorial: Python, PyTorch, the GPU,
PyTorch's flash-attention kernel on tensor cores, and nixnan.

    python3 check_env.py              # quick checks (seconds)
    python3 check_env.py --nixnan     # + run smoke_flash.py under nixnan (~1 min)
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NIXNAN_SO = os.environ.get("NIXNAN_SO", os.path.join(HERE, "..", "external", "nixnan", "nixnan.so"))
fails = []


def check(ok, what, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {what}" + (f": {detail}" if detail else ""))
    if not ok:
        fails.append(what)
    return ok


print("Python / PyTorch")
check(sys.version_info >= (3, 9), "Python >= 3.9", sys.version.split()[0])
try:
    import torch
except ImportError:
    check(False, "torch importable", "python3 -m pip install --user -r requirements.txt")
    sys.exit(1)
check(torch.__version__.startswith("2.3.1"), "torch 2.3.1 (requirements.txt)", torch.__version__)
check((torch.version.cuda or "0").split(".")[0] == "12", "torch built for CUDA 12.x", str(torch.version.cuda))

print("GPU")
if not check(torch.cuda.is_available(), "CUDA device visible to torch"):
    sys.exit(1)
cap = torch.cuda.get_device_capability()
check(cap >= (8, 0), "compute capability >= 8.0 (PyTorch's flash attention needs Ampere or newer)",
      f"{torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}")
try:
    drv = open("/proc/driver/nvidia/version").readline()
    m = re.search(r"Kernel Module\s+(\d+)\.(\d+)", drv)
    ver = f"{m.group(1)}.{m.group(2)}" if m else "?"
    print(f"  [info] NVIDIA driver {ver} (NVBit 1.8 documents <= 575.xx; 595.84 tested working)")
except OSError:
    pass

print("Flash attention")
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

q = torch.randn(1, 4, 6, 32, device="cuda", dtype=torch.float16)
try:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            F.scaled_dot_product_attention(q, q, q, is_causal=True)
        torch.cuda.synchronize()
    names = {e.name for e in prof.events()}
    flash = sorted(n for n in names if "flash_fwd" in n)
    check(bool(flash), "SDPA dispatches to PyTorch's flash kernel", flash[0][:70] + "..." if flash else str(names))
except RuntimeError as e:
    check(False, "SDPA flash backend runs", str(e).splitlines()[0])

print("nixnan")
if check(os.path.isfile(NIXNAN_SO), "nixnan.so present", os.path.realpath(NIXNAN_SO)):
    if "--nixnan" in sys.argv:
        r = subprocess.run([os.path.join(HERE, "nixnan.sh"), "smoke", "smoke_flash.py"],
                           cwd=HERE, capture_output=True, text=True)
        log = os.path.join(HERE, "logs", "smoke.log")
        text = open(log).read() if os.path.exists(log) else ""
        hmma = re.search(r"#nixnan: error \[(\w+)\] detected in operand \d+ of instruction (HMMA[^;]*)", text)
        check(r.returncode == 0 and hmma is not None,
              "nixnan reports flash's tensor-core (HMMA) operands",
              f"[{hmma.group(1)}] in {hmma.group(2).strip()}" if hmma else (r.stdout + r.stderr)[-300:])
    else:
        print("  [skip] nixnan run (pass --nixnan)")

print("all checks passed" if not fails else f"{len(fails)} check(s) failed: {fails}")
sys.exit(1 if fails else 0)
