# Why the torch.where NaN-gradient fix is hard (and costly)

Evidence is from the PyTorch issue threads (read with `gh`, 2026-10-20) and from `fix_cost_bench.py`
(this machine: RTX 3090, torch 2.3.1+cu121, N = 2^24 float32, forward+backward of
`where(a<=0, exp(a), 1+a).sum()`, inputs finite so all variants do the same useful work).

## History
| date | event |
|---|---|
| 2017-12-12 | #4132 "x.grad should be 0 but get NaN after x/0" opened. Early answer: the backward of masking holds 0s, and 0 * inf = NaN. Still OPEN; labels `needs design`, `has workaround`. |
| 2019-07 | #23156: "autograd does not distinguish 'no gradient' and 'zero gradient'"; suggested carrying "an additional (sparse?) tensor". Closed not planned 2022-07-25. |
| 2021-01 | #31829 (logsumexp/softmax at -inf): fixed per operator, closed completed 2021-01-11. A maintainer on `where`: "I don't think there is any plan to change that." |
| 2021-08-09 | #4132 triage review: "expected behavior (for now at least)... we may need a first-class concept of masking in PyTorch ... so this won't happen anytime soon." |
| 2021-11 | #68425 (this issue) filed against 1.9.1; closed not planned 2022-07-25 as a duplicate of #4132. |
| 2022-03 | MaskedTensor prototype (PyTorch 1.11) offered as the structural answer. |
| 2025-06-17 | #156212 closed as duplicate; a collaborator: "it won't be fixed, there's no reasonable way to fix that." |

## Why a fix is hard (as stated in the threads)
1. **The NaN is not made by `where`.** `where` correctly sends 0 to the unused branch; the op *before* it
   (`exp`'s backward, `grad * result`) turns 0 into 0 * inf = NaN. A fix has to change that op or fuse it with `where`
   (a custom `Function`), op by op.
2. **No "unused" vs "zero" distinction.** For a numerically-zero gradient the NaN is wanted; for an unused one it is not.
   Telling them apart needs per-element validity metadata through the whole autograd graph, i.e. MaskedTensor-style
   "first-class masking".
3. **A forward-mask fix is data dependent.** One collaborator: it "will require the CPU-GPU sync for cuda inputs".
4. **Changing `mul`'s backward** (copy zeros instead of multiplying) was asked about ("or is the problem the perf hit of
   modifying the mul's backward?") and not answered in the thread.
5. **IEEE-754 argument** (a maintainer): making `inf * 0 == 0` "breaks IEEE754 ... we don't have the metadata".

No maintainer says the issue is left unfixed *for speed*. The performance rationale for `where` gradients appears in
TensorFlow's `tf.where` documentation, not PyTorch's.

## Measured overhead of fix strategies (fix_cost_bench.py, Logs/fix-cost-bench.log)
Stable over four runs (baseline 1.468 ms/iter every time).
| strategy | ms/iter | vs baseline | extra peak memory |
|---|---|---|---|
| baseline `where` | 1.468 | 1.00x | 0 |
| double where (inner `where` on the input) | 1.99 | 1.35x | +16 MiB |
| custom `Function` (masked backward for exp) | 1.90 | 1.29x | +144 MiB |
| mask check with `.item()` (CPU-GPU sync) | 1.57-1.69 | 1.07-1.15x | 0 |

These are my own measurements of plausible fixes on one machine and one op, not PyTorch's numbers. An N = 1024
variant was dropped: its baseline differed from itself by 19% between measurements (launch-bound noise).
The sync variant's cost also depends on how much work is queued behind it, so it is not a stable ratio.
