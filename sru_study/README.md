# sru_study

A `nixnan` case study built around a real, closed GitHub issue against
[`asappresearch/sru`](https://github.com/asappresearch/sru) (Simple
Recurrent Units): the exact README example from the `sru` package itself
produces `NaN` in both its output and its recurrent state, immediately,
on first use.

## The issue

[asappresearch/sru#193](https://github.com/asappresearch/sru/issues/193):
**Nan in output from example code**

> I have run the example code in the readme on both 2.6.0 and 3.0.0-dev
> and both have nan values in both the output and state objects using
> pytorch 1.9, I've tried this on my computer using a Titan X and also
> with a fresh install on a cloud T4, this doesn't seem to relate to the
> other nan issue raised here (#185) because this problem appears
> immediately.

A maintainer (`taolei87`) asked for confirmation with the exact
reproduction code, which the reporter confirmed matches:

```python
import torch
from sru import SRU, SRUCell

# input has length 20, batch size 32 and dimension 128
x = torch.FloatTensor(20, 32, 128).cuda()

input_size, hidden_size = 128, 128

rnn = SRU(input_size, hidden_size,
    num_layers = 2,          # number of stacking RNN layers
    dropout = 0.0,           # dropout applied between RNN layers
    bidirectional = False,   # bidirectional RNN
    layer_norm = False,      # apply layer normalization on the output of each layer
    highway_bias = -2,        # initial bias of highway gate (<= 0)
)
rnn.cuda()

output_states, c_states = rnn(x)      # forward pass
```

This is copied verbatim into `reproducer.py`.

## Root cause (already known, per the issue thread)

A later comment on the issue, from someone using
[GPU-FPX](https://github.com/LLNL/GPU-FPX.git) (a different NaN/Inf
detection tool, LLNL) to debug it, identifies the actual cause:

> After using our tool GPU-FPX to debug this issue, we found that the
> problem is raised from `torch.FloatTensor(20, 32, 128).cuda()`, which
> creates a tensor with uninitialized data on GPU memory. If you can
> change the example code as `torch.randn(20,32,128).cuda()` the NaN
> issue can be solved :-).

In other words: `torch.FloatTensor(...).cuda()` allocates GPU memory
without initializing it (unlike `torch.zeros`/`torch.randn`), so `x`
starts out as whatever garbage bytes happened to already be in that GPU
memory region -- which can include bit patterns that decode as `NaN` or
`Inf` in FP32. `SRU`'s recurrence then propagates that garbage through
every subsequent layer and timestep. This is **not a bug in SRU itself**
-- it's the README example itself using an uninitialized-memory
constructor where a random or zero-initialized one was intended. The
same comment also references this tutorial's author by name
(`Prof. Ganesh`), so this issue was apparently already known here before
this case study was built -- it's included as a case where two
independent NaN/Inf-detection tools (GPU-FPX and, here, `nixnan`) can be
compared against the same, already-diagnosed root cause.

## This environment's own reproduction

`sru==2.6.0` (matches the issue's own reported version) is already
installed here. Running `reproducer.py` directly (no `nixnan`) confirms
the bug reproduces exactly as described:

```
output NaN: True
c_states NaN: True
```

## Running it under nixnan

See `../All-Runs.md` for the exact command/environment variables used and
this run's summary; the logs of the original (uninitialized-input) example are in `Logs_uninit_bug/`.

## Control experiment: is the uninitialized input really the cause?

The claim above ("`torch.FloatTensor(20, 32, 128).cuda()` is uninitialized and its garbage causes the
NaNs") was checked by changing only that one line (`run_variants.sh`; same two nixnan modes as before):

| variant | script | logs |
|---|---|---|
| uninitialized (original, the bug) | `reproducer.py` | `Logs_uninit_bug/` |
| all zeros: `torch.zeros(20, 32, 128, device="cuda")` | `reproducer_init0.py` | `Log_init0/` |
| N(0,1): `torch.normal(mean=0.0, std=1.0, size=(20, 32, 128), device="cuda")`, seed 0 | `reproducer_normal.py` | `Log_normal/` |

Each `Log_*/` has `native.log` (no nixnan; prints whether the input, output and state contain NaN),
`iter-autonixnan-nosampling.log` and `straight-nixnan.log` (a plain `LD_PRELOAD` run, all kernels).

Results (RTX 3090, nixnan from `../external/nixnan`):

| | input NaN | output / state NaN (native) | straight nixnan: non-zero exception counters | iter-autonixnan: reports with non-zero counters |
|---|---|---|---|---|
| uninitialized | ~350 of 81,920 | True / True | FP32: NaN 1231 (2,633,048 repeats), Inf 544, -Inf 449, Subnormal 696; FP16: NaN 6, Inf 6, -Inf 6; FP32 memory NaN 3 | the 4 kernel reports contain 9 non-zero counter lines |
| zeros | 0 | False / False | none | none (8 reports) |
| N(0,1) | 0 | False / False | none | none (9 reports) |

So with the same network, weights and nixnan settings, initializing the input removes **every**
exception nixnan reports, including the thousands of FP32 NaN, Inf, -Inf and subnormal sites, and the
NaNs in the output and state. All six nixnan logs reached "Finalizing GPU context".

`check_uninit_input.py` looks at the uninitialized input directly (`Logs_uninit_bug/native-input-check.log`):
`x` contains 344-360 NaN and no Inf in three native runs, and 349 NaN under nixnan, and the output and
state are NaN each time. The count is not constant, because the content is whatever the allocator
happened to hand back. It also depends on what the process did before: a bare script that imported only
`torch` and allocated `x` saw **0** NaN (only zeros and about 400 subnormals), while the same allocation
after `from sru import SRU` sees ~350. So "uninitialized" really means "whatever is left in reused GPU
memory", which is why the bug is intermittent across environments.

Caveats: one run per configuration (no repeats of the nixnan runs); the zero/normal control runs
were all clean, so this shows the uninitialized input is *sufficient to explain* the exceptions, but it
does not say which bytes of the garbage became NaN (that is what `Logs_uninit_bug/` localizes to the
SGEMM and `sru_cuda_forward_kernel_simple`). `Log_normal` additionally prints a few
`warning: Infinite immediate found ... +INF` lines: nixnan's static note that the RNG (`curand`) kernel's
code contains an `+INF` constant operand; no run-time Inf or NaN resulted. Wall times of the nixnan runs
(about 4 minutes straight) are not comparable with the ~79 s recorded earlier; the machine was
busier or slower this time and I did not investigate.
