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
this run's summary; logs are collected in `Logs/`.
