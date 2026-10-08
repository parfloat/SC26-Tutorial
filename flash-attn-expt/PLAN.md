# Flash-Attn experiment: plan

## Question

BioMistral-7B under nixnan (`flash-attn.pptx`, slide 3) reports millions of
NaN/Inf/subnormal operands, mostly inside `pytorch_flash::flash_fwd_kernel`,
and almost none reaching memory. Two explanations compete:

* **H1 (flash attention):** flash attention deliberately lets NaN/Inf arise
  and then filters them, for example at the split-KV combine guard
  `lse_logsum = (lse_sum == 0.f || lse_sum != lse_sum) ? INFINITY : ...`
  (flash-attention `flash_fwd_kernel.h:1204` upstream, `:1157` in PyTorch 2.6).
* **H2 (4-bit):** BioMistral was loaded with bitsandbytes 4-bit quantization,
  and nixnan misreads packed 4-bit data as FP16/FP32 operands.

The plan is to build a model small enough that every tensor can be checked,
run it through the same PyTorch flash kernels BioMistral used (same venv:
torch 2.6.0+cu124), and switch each suspect on and off separately.

## What reading the source shows first (step 0)

From PyTorch v2.6.0 `aten/src/ATen/native/transformers/cuda/flash_attn/`:

1. The guard lives only in `combine_attn_seqk_parallel`, the **split-KV combine
   kernel** (`flash_fwd_splitkv_combine_kernel`). This kernel runs only when
   `set_params_splitkv` picks `num_splits > 1`. That needs more than one
   key block, i.e. `seqlen_k > 256` for head_dim <= 64 (`flash_api.cpp:303`).
   Short sequences never launch it. The kernel in the BioMistral log,
   `flash_fwd_kernel`, is the non-split kernel and does not contain the guard.
2. Inside the split kernel, `softmax.h:176-177` already maps a NaN or zero row
   sum to `lse = -INFINITY` (`Split ? -INFINITY : INFINITY`). A per-split LSE
   is therefore always finite or -inf, and `lse_max == -inf` is clamped to 0
   (`flash_fwd_kernel.h:1150`). So in the combine kernel `lse_sum` is a sum of
   `expf(finite or -inf)`, and **`lse_sum != lse_sum` looks unreachable**. The
   reachable branch is `lse_sum == 0`: every split's LSE is -inf, either
   because the row is fully masked or because NaN scores in **every** split were
   already turned into -inf.
3. That -inf only fixes the LSE. The split's output accumulator still holds
   `P·V` with NaN `P`, and the combine computes `0 * NaN = NaN`. So NaN in Q/K
   still leaks into the attention output.
4. In the non-split kernel, out-of-range rows and columns of a tile are not
   always loaded or cleared. For example, K beyond `seqlen_k` is left in shared
   memory "since we'll mask out the scores anyway". Arithmetic on those lanes can
   see garbage (NaN/Inf/subnormal) that is never written back.
   **H3 (padding lanes)** is a third candidate for "many NaNs, none in memory".

The experiment is designed to test points 1–4 empirically.

## Steps

| # | step | output |
|---|------|--------|
| 1 | Create branch `Flash-Attn` in SC26-Tutorial and directory `flash-attn-expt/` | branch |
| 2 | `gen_training_set.py`: 100 random words over {0..9}, length 1–20, fixed seed | `training-set.txt` |
| 3 | `tinygpt.py`: decoder-only transformer with 4 layers, d_model 128, 4 heads (head_dim 32), RoPE, pre-LN, KV cache. Linears in FP32 with TF32 off, so no tensor cores. Attention in FP16 through `F.scaled_dot_product_attention` with the flash backend **forced**. This is the same dispatch path HF/BioMistral use, not a raw kernel call. | model code |
| 4 | `train.py`: next-token training on the fixed set, saves `tinygpt.pt`, reports accuracy on the training set | checkpoint |
| 5 | `generate.py`: greedy autoregressive completion of `312` to length 6 | completion |
| 6 | Smoke test without nixnan, then commit and push (model + training set) | commit 1 |
| 7 | **Training under nixnan, `SAMPLING=32`** (re-trains with the same seed) | `logs/train-sampling32.log` |
| 8 | **Completion of `312` → 6 under nixnan, full monitoring** (`SAMPLING=0`, `INSTR_MEM=1`) | `logs/eval-312-full.log` |
| 9 | **Controlled A/B on the combine guard** (`experiment_lse.py`). It uses the same model and prompt, with a 297-token digit context in front of `312` so `seqlen_k` = 300 > 256 and the split-KV + combine path runs. The external input is an extra embedding vector added to the last prompt position (the kind a vision encoder or soft prompt supplies). **A:** finite (zeros). **B:** contains +inf (an upstream FP16 overflow). Without nixnan, an oracle run reads flash's returned LSE per layer and step: +inf means the combine guard fired. It also checks that the inputs to the guarded line can't be NaN. Then both inputs run under full nixnan. | `logs/lse-A.log`, `logs/lse-B.log` |
| 10 | **4-bit control:** the same checkpoint with its Linears replaced by bitsandbytes NF4 `Linear4bit` (the BioMistral setting), `312` → 6 under full nixnan, compared with step 8. Also an FP16-linears variant, to separate "FP16 tensor-core GEMM" from "4-bit". | `logs/eval-312-full-nf4.log`, `logs/eval-312-full-fp16.log` |
| 11 | Short-sequence padding control: with clean data the non-split `flash_fwd_kernel` sees seqlen 3–6 against 128-row tiles. Any NaN reported there in step 8 comes from padding lanes (H3), since every real tensor is checked finite. | (from step 8) |
| 12 | Write up the results in `README.md`, then commit and push | commit 2 |

## Fixed choices

* Tokens 0–9, plus BOS = 10 and EOS = 11 (needed to start and end a word).
  BOS/EOS logits are masked during completion, so the output is pure digits.
* Precision: FP32 parameters and linears; FP16 for attention only. Flash
  supports only FP16/BF16 and runs only on tensor cores (HMMA), so attention
  can't avoid tensor cores. Everything else does.
* Python environment: `~/repos/newmed/venv`, the BioMistral venv (torch
  2.6.0+cu124, bitsandbytes 0.48.1), so the flash kernels are byte-identical to
  the ones nixnan saw in BioMistral.
* nixnan: `external/nixnan/nixnan.so` (branch `print_histo_instrn`).

## Execution notes (what changed while executing)

* Direct nixnan runs of PyTorch never finished start-up. NVBit takes about
  30 s per cuBLAS kernel, and `F.embedding` (plus the CUB sort in its backward)
  hangs it outright. All monitored runs therefore use `FUNCTION_WHITELIST`
  (flash kernels; plus bitsandbytes or cuBLAS kernels for step 10), and the
  embedding is computed as a one-hot matmul. The values are bit-identical.
  Details are in README.md, "Getting nixnan to run on PyTorch at all".
* Step 11 grew into a prompt-length sweep (T = 2…128), and a verbose run was
  added to check which stores `INSTR_MEM` actually covers.
* Step 7 writes `tinygpt-nixnan.pt`. `tinygpt.pt` (from step 4) stays the
  committed model, so that every result uses the same weights.
