# PyTorch_Instrumentation: flash attention's padding lanes and NaN guards, dumped without nixnan

nixnan reported thousands of NaN operands in `flash_fwd_kernel`, even though
every tensor TinyGPT hands to flash is finite (`../README.md`, findings 1–2).
The explanation offered there was *padding lanes*: arithmetic on tile rows and
columns outside the sequence. This folder shows those values directly, using
PyTorch source instrumentation and no nixnan.

## How

* **`src/` is PyTorch v2.6.0's own flash-attention**
  (`aten/src/ATen/native/transformers/cuda/flash_attn`). The namespace is
  renamed to `instr_flash`, and only the FP16 head-dim-32 kernels TinyGPT uses
  are compiled. The host code is PyTorch's own `mha_fwd`, so the split
  heuristic and kernel template choice match what SDPA picks.
* **Dump hooks** (`instrumentation.patch`, every hunk marked `INSTRUMENTATION`):
  * In `flash_fwd_kernel` they copy whole tiles: the Q, K and V shared-memory
    tiles as raw FP16 bits, then S = QKᵀ before and after masking, P, O, and the
    per-row `softmax.h` guard `sum == 0 || sum != sum`.
  * In `flash_fwd_splitkv_kernel` they record that `softmax.h` guard per
    (split, head, row).
  * In `flash_fwd_splitkv_combine_kernel` they record the `lse_sum` guard per
    row: both disjuncts, the value it replaced, and the number of non-finite
    values in the combined output row.
* **The instrumented build computes exactly what PyTorch computes.**
  `check_equivalence.py` compares output and LSE against
  `aten::_scaled_dot_product_flash_attention` and finds them **bitwise equal**
  on all 16 checks: 8 shapes (non-split and 2-split), with dumps on and off.
  The results are in `traces/equivalence.txt`. (It must be built without
  `--use_fast_math`; with it the combine LSE differs by 1 ulp.)
* `instr_flash.install()` replaces `tinygpt.flash_attention` with the
  instrumented kernel. Nothing else in the model changes.

```
./make_src.sh                       # optional: regenerate src/ from PyTorch v2.6.0 + instrumentation.patch
python check_equivalence.py         # builds the extension (~5 min, CUTLASS fetched by get_cutlass.sh)
python dump_completion.py           # traces/completion/   312 -> 6, last 4 flash calls
python dump_lse.py --input A        # traces/lse-A/        combine-guard experiment, clean input
python dump_lse.py --input B        # traces/lse-B/        same with +inf in one embedding
```

(Use the BioMistral venv, `~/repos/newmed/venv`, with `CUDA_HOME=/usr/local/cuda-12.4`.)

## 1. The padding lanes, completion of `312` (`traces/completion/`)

The completion makes 3 forward passes × 4 layers = 12 flash calls. The dump
covers the last 4 calls, i.e. the 4 layers of the last forward pass. Each is a
decode step with seqlen_q = 1 and seqlen_k = 6, inside one 128 × 128 tile.
Call 11 (layer 3), summed over the 4 heads:

| region | values | NaN | ±inf | subnormal |
|---|---:|---:|---:|---:|
| Q row 0 (loaded) | 128 | 0 | 0 | 0 |
| **Q rows 1–127 (never loaded)** | 16,256 | **3,239** | 0 | 73 |
| K rows 0–5 (loaded) | 768 | 0 | 0 | 0 |
| K rows 6–127 (never loaded) | 15,616 | 13 | 0 | 16 |
| V rows 6–127 | 15,616 | 0 (all cleared to 0) | 0 | 0 |
| S = QKᵀ, valid (row 0, cols 0–5) | 24 | 0 | 0 | 0 |
| **S, padding rows** | 65,024 | **18,274** | 0 | 0 |
| S after mask, padding rows | 65,024 | 798 | 61,976 (−inf, the mask) | 0 |
| **P, padding rows** | 65,024 | **798** | 0 | 1 |
| **O, padding rows** | 16,256 | **4,256** | 0 | 0 |
| O, row 0 (the real output) | 128 | 0 | 0 | 0 |

What the dump shows:

* **Every value inside the sequence is finite. The NaNs exist only in lanes
  outside it.**
  * Flash never loads Q rows ≥ seqlen_q. The source comment says, "We don't
    need to clear the sQ smem tiles since we'll only write out the valid
    outputs".
  * It also never loads K rows ≥ seqlen_k ("we'll mask out the scores anyway").
  * Those rows hold whatever the shared memory last contained. The tensor-core
    MMA, the softmax and the P·V product all run on them.
* **The `sum != sum` disjunct of `softmax.h:177` does fire, on padding rows.**
  * It fires on 133 of 508 padding rows (sum = NaN), turning their LSE into
    +inf.
  * Those LSEs, and the NaN O rows, are never stored, because of the
    `row < seqlen_q` predicate on the stores.
  * On real rows it fires 0 times.
* **The stale values are the previous flash call's output.**
  * 416 of 508 stale Q rows in calls 9, 10 and 11 are bit-identical to rows of
    the O tile of the call just before.
  * The epilogue stages O in shared memory at the same address as sQ
    (`sO = make_tensor(sQ.data(), ...)`).
  * The NaN padding rows of call n's O (130 rows) therefore become call n+1's
    stale Q rows (130 rows with NaN). The NaNs carry over from one launch to the
    next and never reach global memory.
  * The stale K rows are zeros (404) or K rows of earlier calls (84). The
    other 92 stale Q rows match no dumped flash tile, so they presumably come
    from other kernels that ran on the same SM.
* **The NaNs are not deterministic, but the result is.** A second identical
  run gives a different stale Q tile and a bitwise-identical real output. This
  explains why nixnan's NaN counts varied between repeats (681 / 930 / 937 in
  `../README.md`, finding 2).

Per-call files `call{8..11}_layer{L}.txt` list every head's Q, K, V and O tiles
in full, S/P for the first rows, and all 128 per-row softmax states. Rows
outside the sequence are marked `*`. `dumps.npz` holds every tile, including
the raw FP16 bits.

## 2. What the guards clean up, inputs A and B (`traces/lse-A/`, `traces/lse-B/`)

The setup is the same as `../experiment_lse.py`: a 301-key prompt, so PyTorch
picks 2 KV splits for every call, followed by the combine kernel.
Input B adds +inf to one coordinate of the last prompt embedding. Totals over
all 12 calls (4 heads):

| | A (clean) | B (+inf) |
|---|---:|---:|
| splitkv `softmax.h` guard, real rows: sum != sum → lse −inf | 0 | **1,144** |
| splitkv `softmax.h` guard, padding rows: sum != sum | 1,588 | 3,931 |
| splitkv: split has no key block (causal), lse −inf written directly | 4,096 | 4,096 |
| combine guard, real rows: fired | 0 | **568** |
| — via `lse_sum == 0` | 0 | 568 |
| — via **`lse_sum != lse_sum`** | **0** | **0** |
| combine guard, padding rows (16-row block tail): fired via `lse_sum == 0` | 144 | 144 |
| combined output rows with non-finite values | 0 | **752** |
| completion | `…312489` | `…312000` |

* **`lse_sum != lse_sum` never fires in either input, while `lse_sum == 0`
  fires 568 times in B.** The NaN is cleaned one level earlier.
  * In the split kernel, `softmax.h` maps a NaN row sum to lse = −inf: 1,144
    times on real rows in B.
  * The combine kernel therefore only ever sees finite or −inf LSEs. When both
    splits are −inf, `lse_sum = 0` and the guard writes +inf.
* **The guard cleans the LSE but not the output.**
  * In B, 752 combined output rows contain NaN, while the guard fired on 568
    rows.
  * In call 0 (layer 0, prefill), the guard fires on only 4 rows (row 300 in
    each head). Yet 180 rows have NaN output: rows 256–300 × 4 heads, which
    share the second split with the poisoned key.
  * In the decode steps of layer 0, split 0 is clean, so the guard doesn't fire
    at all. The output is still NaN, because `0 · NaN` in
    `O += lse_scale · Oaccum`.
* **On clean input (A), the `softmax.h` NaN guard fires 1,588 times, all on
  padding rows.** That is the same padding-lane effect as section 1, in the
  split kernel's 64-row tiles.

`call*_combine.txt` lists each cleaned or NaN row: both split LSEs, lse_max,
lse_sum, which disjunct fired, the value replaced, the value stored, and the
number of non-finite O values. It also lists the `softmax.h` records.
`records.npz` has every record of every call.

## Summary

* **The NaNs nixnan counts in `flash_fwd_kernel` are real values in real
  registers, but in lanes outside the sequence.**
  * Shared memory is not cleared, so those lanes hold the previous launch's
    output.
  * The kernel computes on them and then discards them through the
    `row < seqlen_q` store predicates and the −inf mask.
  * None of this depends on 4-bit weights.
* **Of flash's two NaN guards, only the `softmax.h` one sees a NaN.** The
  combine guard's `lse_sum != lse_sum` disjunct is unreachable in practice,
  because NaN sums become −inf before the combine. It is exercised neither by
  clean nor by poisoned input.

## 3. When can `lse_sum != lse_sum` fire? (`traces/trigger/`, `traces/lse-A-scale-0.1/`)

* **Never, for any positive softmax scale and any FP16 input** (±inf and NaN
  included). That covers TinyGPT's and BioMistral's 1/√d.
  * `lse_sum` is NaN only if a split's LSE is +inf or NaN. That needs a row sum
    of +inf, because `softmax.h` maps 0 and NaN to −inf.
  * With a scale σ > 0, every exponent `x·σ₂ − m·σ₂` is ≤ 0. If `m·σ₂`
    overflows, the compiled code gives `∞ − ∞ = NaN`, which the guard catches.
  * The compiled code is unfused: an `FMUL` by the scale, then a separate
    `FADD` of the scaled max (`traces/trigger/sass_scale_apply_exp2.txt`),
    although the source comment expects an FFMA. At scale 1e37 PyTorch's kernel
    returns the same values as ours, so it is unfused too.
  * The proof is in `../latex/report.pdf`, §8.
* **It fires with a negative scale.** The exponent becomes `(m − x)·|σ₂| ≥ 0`.
  * A masked score (−inf) gives `2^{+inf} = +inf`, so the row sum is +inf.
    That is neither 0 nor NaN, so `softmax.h` passes it through and lse = +inf.
  * The combine then computes `inf − inf = NaN`.
  * SDPA does not check the sign of `scale`.
  * `trigger_nan_guard.py`: one decode-shaped call (301 keys, 2 splits). It
    fires on 4 of 4 rows for scales −0.1, −1 and −10, and on 0 rows for
    1/√32, 1e30 and 1e37. Every case matches PyTorch's own kernel, NaN-aware.
* **Inside TinyGPT:** `dump_lse.py --input A --scale -0.1`, with clean input.
  * The NaN disjunct fires **1,208 times**, all in layer 0, where the masks
    supply the −inf scores.
  * Layers 1–3 then fire through `lse_sum == 0` (3,636 times), because their
    inputs are already NaN.
  * With the default scale, the count is 0.

A detailed LaTeX report and 12 beamer slides are in `../latex/`
(`report.pdf`, `slides.pdf`).
