# reproducer.py -- minimal reproduction from
# https://github.com/pytorch/pytorch/issues/173638
# ("`torch.linalg.slogdet` does not propagate `NaN` in CUDA"), copied
# verbatim from the issue body, with only the CPU call made optional
# (SKIP_CPU=1, default) -- the bug is entirely a CUDA-vs-CPU divergence,
# and the CPU call is only there upstream as a reference value; it never
# needs to run for the CUDA path (the one under nixnan observation here)
# to exercise the bug. See README.md for the full issue text.
import os
import torch

print(f"Torch version: {torch.__version__}")

A = torch.tensor([[-1.0, 2.0, 3.0],
                 [float("nan"), 5.0, 6.0],
                 [7.0, 8.0, 10.0]], dtype=torch.float32)
A2 = torch.tensor([[-1.0, 2.0, 3.0],
                 [0, 5.0, 6.0],
                 [7.0, 8.0, 10.0]], dtype=torch.float32)

skip_cpu = os.environ.get("SKIP_CPU", "1") != "0"
if not skip_cpu:
    s_cpu, l_cpu = torch.linalg.slogdet(A)
    print(f"logabsdet for A [CPU] {l_cpu}")
else:
    print("logabsdet for A [CPU] SKIPPED (SKIP_CPU=1; set SKIP_CPU=0 to run it)")

if torch.cuda.is_available():
    s_gpu, l_gpu = torch.linalg.slogdet(A.cuda())
    s2_gpu, l2_gpu = torch.linalg.slogdet(A2.cuda())
    print(f"logabsdet for A [CUDA] {l_gpu}")
    print(f"logabsdet for A2 [CUDA] {l2_gpu}")
else:
    print("CUDA not available -- nothing to reproduce")
