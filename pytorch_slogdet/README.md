# pytorch_slogdet

A `nixnan` case study built around a real, currently-open PyTorch bug: a
CUDA linear-algebra routine that silently drops a `NaN` on the way to its
output -- exactly the pattern this whole tutorial is about, except this
one is a genuine, user-filed report, not a synthesized example.

## The issue

[pytorch/pytorch#173638](https://github.com/pytorch/pytorch/issues/173638):
**`torch.linalg.slogdet` does not propagate `NaN` in CUDA**

> When the input tensor contains `NaN` values, the CPU implementation of
> `torch.linalg.slogdet` propagates that and returns `NaN` as the
> `logabsdet`. However, the CUDA implementation computes it as if there
> was a 0 in the `NaN`'s place and calculates the value. This makes it
> harder to detect corruption in data with the CUDA version.

Minimal reproduction from the issue body:

```python
import torch
A = torch.tensor([[-1.0, 2.0, 3.0],
                 [float("nan"), 5.0, 6.0],
                 [7.0, 8.0, 10.0]], dtype=torch.float32)
A2 = torch.tensor([[-1.0, 2.0, 3.0],
                 [0, 5.0, 6.0],
                 [7.0, 8.0, 10.0]], dtype=torch.float32)

s_cpu, l_cpu = torch.linalg.slogdet(A)
s_gpu, l_gpu = torch.linalg.slogdet(A.cuda())
s2_gpu, l2_gpu = torch.linalg.slogdet(A2.cuda())
```

Reported output (PyTorch 2.10.0+cu128, Tesla T4):

```
logabsdet for A [CPU] nan
logabsdet for A [CUDA] 3.1354942321777344
logabsdet for A2 [CUDA] 3.1354942321777344
```
`A`'s `NaN` and `A2`'s explicit `0` produce the *same* CUDA answer --
the CUDA path is silently treating the `NaN` as if it were exactly zero,
and the corrupted-input case comes back looking like a perfectly
ordinary, plausible number.

Filed by @jiren-the-gray; labeled `module: correctness (silent)`,
`module: cuda`, `module: linear algebra`, `module: NaNs and Infs`.
Open as of this writing.

## This environment's own reproduction

`reproducer.py` is the issue's minimal reproduction, copied verbatim,
with only the CPU call made optional (`SKIP_CPU=1` by default -- the bug
is a CUDA-vs-CPU divergence, and the CPU call is only there upstream as
a reference value; it never needs to run for the CUDA path to exercise
the bug).

Run directly (no `nixnan`) in this environment: **torch 2.3.1+cu121
correctly propagates the `NaN`** --

```
logabsdet for A [CUDA] nan
logabsdet for A2 [CUDA] 3.1354942321777344
```

-- i.e. the silent-corruption bug does **not** reproduce here; it is
apparently specific to a newer PyTorch/cuSOLVER version than the one
installed in this environment (the issue's own environment is
2.10.0+cu128). This is itself a useful, honest finding: a bug report's
behavior doesn't always transfer unchanged to a different
library/driver version, and that has to be checked empirically, not
assumed.

## Running it under nixnan

`torch.linalg.slogdet` on CUDA runs through cuSOLVER's LU factorization
(`getrf`) internally -- a large, heavily-templated library kernel. Two
direct, non-`iter-autonixnan` attempts (plain `LD_PRELOAD=nixnan.so`,
and again with `SAMPLING=512`) both stalled past 120 seconds, stuck
mid-SASS-instrumentation-insertion before the target program ever
printed a line of its own output (see `../autonixnan-doc.tex` Sec 2.8,
and `../All-Runs.md`, for why: this is the same one-time
instrumentation-insertion cost documented at smaller scale for
`lu_solve_gpu`, just much larger here because of cuSOLVER's template
surface).

`iter-autonixnan` (this tutorial's own tool, `external/nixnan/bin/`)
succeeded where those direct attempts did not, by whitelisting one
kernel at a time instead of instrumenting all 19 at once:

```bash
../external/nixnan/bin/iter-autonixnan -s 8 -t 60 -- python3 reproducer.py \
  > iter-autonixnan.log 2>&1
```

All 19 kernels visited, no timeouts. It traces the real NaN propagation
path through `slogdet`'s own CUDA kernels directly:

| Kernel | Finding |
|---|---|
| `getrf_pivot<getrf_params_<float,32,...>>` (cuSOLVER's LU factorization) | 7 distinct NaN sites (5 repeats) |
| `elementwise_kernel<...sign_kernel_cuda...>` (the determinant's sign) | 2, then 1 distinct NaN site |
| `vectorized_elementwise_kernel<...log_kernel_cuda...>` (`log\|·\|` for `logabsdet`) | 2 NaN sites, 1 Infinity site |
| `reduce_kernel<512,1,...>` (reduction across the diagonal) | 10 distinct NaN sites |

The `NaN` genuinely enters via the LU factorization operating on the
literal `NaN` in the input, then propagates through the sign
computation, the log computation, and the final reduction -- this is
the real, kernel-level mechanism, independent of whether a given
PyTorch/cuSOLVER version's *final printed answer* happens to come out
looking corrupted (as in the issue) or correctly as `nan` (as in this
environment).

See `../All-Runs.md` for exact commands/environment variables for every
run in this directory.
