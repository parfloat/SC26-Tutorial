# torch_where

A `nixnan` case study built around a real, long-standing, still-open
PyTorch issue: `torch.where`'s backward pass can produce `NaN` gradients
even when the forward pass is perfectly well-defined and differentiable.

## The issue

[pytorch/pytorch#68425](https://github.com/pytorch/pytorch/issues/68425):
**`torch.where` produces nan in backward pass for differentiable forward
pass**

> In the example below, `torch.where` properly executes a differentiable
> forward pass but fails for calculating the correct gradient value:
> ```python
> a = torch.tensor(100., requires_grad=True)
> b = torch.where(a <= 0, torch.exp(a), 1 + a)
> b.backward()
>
> print(b)
> print(a.grad)
> ```
> outputs
> ```
> tensor(101., grad_fn=<SWhereBackward>)
> tensor(nan)
> ```
>
> Expected behavior: the expected gradient in the example above is `1.`
> for `a.grad`.

Filed against PyTorch 1.9.1. Still open as of this writing.

## Why this happens

`torch.where(cond, x, y)`'s forward pass only evaluates and selects the
branch matching `cond`, but PyTorch's autograd still computes gradients
for **both** branches during backward and combines them by masking:
`grad_x = grad_output * cond`, `grad_y = grad_output * ~cond`. Computing
`grad_x` still requires evaluating `d(exp(a))/da = exp(a)` at `a = 100`,
which overflows to `Inf`. That `Inf` is then multiplied by the masked-out
weight `cond = False -> 0`: `0 * Inf = NaN` -- a real, IEEE-754-correct
`NaN`, not a bug in the multiplication itself. That `NaN` then adds into
the total gradient (`NaN + 1 = NaN`), even though the branch that produced
it was never supposed to contribute anything. `torch.where`'s backward
never "short-circuits" the way its forward does.

## This environment's own reproduction

`reproducer.py` is the issue's example, with one change: the tensor is
placed on `device = torch.device("cuda")` explicitly (the original
snippet, run with no device specified, executes on the CPU -- `nixnan`
instruments CUDA kernels via NVBit/`LD_PRELOAD`, so it needs the actual
computation to happen on the GPU to have anything to catch; see
`Karpathy-NGram/gpt_finite_state_cuda.py` for the same adaptation applied
to an earlier case study in this tutorial).

Run directly (no `nixnan`) in this environment, **the bug still
reproduces exactly**, in torch 2.3.1+cu121 -- four PyTorch major versions
after the issue was filed against 1.9.1:

```
b = tensor(101., device='cuda:0', grad_fn=<WhereBackward0>)
a.grad = tensor(nan, device='cuda:0')
```

## Running it under nixnan

See `../All-Runs.md` for the exact command and this run's summary; logs
are collected in `Logs/`.
