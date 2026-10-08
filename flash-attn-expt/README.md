# flash-attn-expt: are BioMistral's NaNs from flash attention, or from 4-bit?

This is a controlled version of the BioMistral-7B observation in
`flash-attn.pptx`. nixnan reported millions of NaN/Inf/subnormal operands,
mostly in `pytorch_flash::flash_fwd_kernel`, and none in memory. The slide
attributes the filtering to the split-KV combine guard
`(lse_sum == 0.f || lse_sum != lse_sum) ? INFINITY : ...`. A second suspicion
is that nixnan misreads BioMistral's 4-bit (bitsandbytes NF4) weights.

The test bed is **TinyGPT**: a 4-layer decoder-only transformer over the digits
0–9. Every tensor of TinyGPT can be checked on the host. It runs in the BioMistral venv
(torch 2.6.0+cu124, bitsandbytes 0.48.1), so the flash kernels are the same
binaries. The plan is in `PLAN.md`, and the full run recipe is in `run_all.sh`.

## Findings

1. **Clean data alone reproduces BioMistral's flash NaN pattern. Neither
   4-bit nor bad data is needed.** The completion of `312` has every
   real tensor finite (host check), but nixnan still reports **1,183 FP32 + 184
   FP16 NaN sites** (31,803 + 3,760 repeats) in `flash_fwd_kernel`. They appear
   in the instructions the slide shows: `HMMA.16816.F32` operand 2, FADD
   `[NaN,subnormal]`, FMUL.
2. **These NaNs are phantoms in unused lanes, not data.** Sweeping prompt
   length T (one 128×128 tile): T ≤ 16 gives 300–930 NaN sites; T = 32, 64, 96,
   120, 127, 128 give **zero** NaN. The middle is irregular (T = 28 has NaN,
   T = 17–24 has none). Repeating the same T = 16 run gives different counts
   (681, 930, 937), although the kernel variant and the inputs are identical.
   That is the signature of arithmetic on leftover, uninitialized memory in
   tile rows and columns beyond the sequence. Flash computes on those lanes and
   discards the results, and nixnan checks every lane's operands.
3. **4-bit is not the source.** With the Linears replaced by bitsandbytes NF4
   (`--linear nf4`) and all of bitsandbytes' kernels instrumented, the
   4-bit kernels produce **no NaN or Inf**: two `kQuantizeBlockwise`, two
   `kDequantizeBlockwise` and `kgemm_4bit_inference_naive`. There are only 19
   subnormal sites in the GEMV. The flash NaNs are unchanged in kind (946 + 168
   sites). FP16 cuBLAS GEMMs (`--linear fp16`) likewise show only subnormals.
   nixnan checks FP instructions, while the packed 4-bit data moves through
   integer and shared-memory ops, so nixnan never treats it as a float.
4. **The `lse_sum != lse_sum` disjunct is never exercised. The guard fires
   through `lse_sum == 0`.**
   * In the split kernel, `softmax.h:177` already turns a NaN or zero row sum
     into `lse = -INFINITY`.
   * Controlled A/B (`experiment_lse.py`): same model, same 300-digit prompt.
     The context is long enough that PyTorch picks 2 KV splits, so the combine
     kernel runs. Only the external input differs:

     | | external input | combine guard fired | NaN LSE returned | NaN in combine's LSE path (nixnan) | NaN in combine's O path (nixnan) |
     |---|---|---|---|---|---|
     | A | zeros | 0 entries | 0 | none (only ±inf) | none |
     | B | +inf in one embedding | 568 (head,row) entries | 0 | **none** (only ±inf) | 20 FFMA sites |

   * In B the combine kernel's LSE instructions (`lse − lse_max` FADD, `·log2e`
     FFMA, `expf` = MUFU.EX2) see only ±inf, identical to A. So `lse_sum` is
     never NaN. The guard fires because every split reported −inf.
   * The NaNs that do reach the combine are in the output accumulation
     `O += lse_scale · Oaccum`, i.e. `0 · NaN = NaN`. So the guard does not stop
     NaN propagation: B's completion is `312000` (NaN logits), against A's
     `312489`.
   * The guard also fires in A, on the padding rows of the combine tile
     (the +inf operand in `FADD R9, -R8, R9`), but those rows are never written
     out.
5. **The combine guard can't run at all in the BioMistral-style log.** It only
   exists in `flash_fwd_splitkv_combine_kernel`. PyTorch launches that kernel
   only when `seqlen_k` exceeds one key block: 256 keys for head_dim ≤ 64, and
   128 keys for head_dim 128, which is Mistral's (`flash_api.cpp:303`). The
   kernel named in the slide's log, `flash_fwd_kernel`, is the non-split kernel.
6. **A NaN breaks causality through tiling.** In B only position 300 is
   poisoned, and causal masking should hide it from rows 256–299. Yet those 44
   rows become NaN in layer 0, along with row 300 itself (45/301 in the oracle). They share the second 256-key split with row
   300, and its masked-out NaN V row enters `P·V` as `0 · NaN`. Rows 0–255,
   which are in the other split, stay clean. Which rows get corrupted depends on
   the tile size, not on the model.
7. **nixnan's memory check is blind to flash's stores.** With `INSTR_MEM=1`,
   every global store in the flash kernels (14 `STG.E.128`, 13 `STG.E`, 5
   `STG.E.64`) is skipped as "unknown type", because they write packed FP16
   pairs. Run B writes NaN attention outputs and still reports
   `FP16/FP32 Memory NaN: 0`. So the slide's "most NaNs do not leak into
   memory" can't be concluded from nixnan's memory counters.

## TinyGPT and the spec

| | |
|---|---|
| tokens | 0–9, plus BOS = 10 and EOS = 11 (masked during completion) |
| training set | `training-set.txt`: 100 random words, length 1–20 (`gen_training_set.py`, seed 2026) |
| model | 4 blocks, d_model 128, 4 heads (head_dim 32), RoPE, pre-LN, MLP 4×, 791,808 params |
| precision | FP32 parameters and Linears with TF32 off (SIMT, no tensor cores). Attention in FP16 via `F.scaled_dot_product_attention` with flash forced, which only exists as tensor-core HMMA code. |
| training | 1,500 full-batch AdamW steps. Loss 0.3939 against an exact optimum of 0.3938 for this set (the first digit of each word is unpredictable); next-token accuracy 0.8353 = optimal |
| completion | `312` → **`312452`** (greedy, KV cache). No training word starts with `312`. NF4: `312428`. |

### nixnan runs (`logs/flash/`)

| run | `SAMPLING` | log | result |
|---|---|---|---|
| training (step 7) | 32 | `train-sampling32.log` | finished: final loss 0.3939, next-token acc 0.8353, `tinygpt-nixnan.pt` completes 312 -> 312632. report (nonzero): FP16 NaN: 704 sites (14565356 repeats); FP16 Infinity: 404 sites (7588 repeats); FP16 -Infinity: 352 sites (7864 repeats); FP16 Subnormal: 800 sites (18404848 repeats); FP32 NaN: 1490 sites (78993020 repeats); FP32 Infinity: 751 sites (37938856 repeats); FP32 -Infinity: 1909 sites (495634629 repeats); FP32 Subnormal: 600 sites (4576082 repeats); flash_fwd_kernel                          1493 sites  {'NaN': 2194, 'subnormal': 1308, '-infinity': 1479, 'infinity': 1083}; flash_bwd                                  458 sites  {'infinity': 72, 'subnormal': 92, '-infinity': 448} |
| `312` → 6 (step 8) | 0 (total) | `eval-312-full.log` | flash_fwd_kernel: 1,367 NaN, 57 +inf, 1,864 −inf, 461 subnormal operand reports |
| combine A / B (step 9) | 0 | `lse-A-full.log`, `lse-B-full.log` | table above; oracle: `logs/lse-[AB]-oracle.txt` |
| NF4 Linears (step 10) | 0 | `eval-312-full-nf4.log` | bitsandbytes kernels: 0 NaN/Inf, 19 subnormal sites |
| FP16 Linears (step 10) | 0 | `eval-312-full-fp16.log` | cuBLAS/CUTLASS: 41 subnormal reports at 8 sites, 0 NaN/Inf |
| tile sweep (step 11) | 0 | `sweep/T*.log` | finding 2 |

`python3 summarize.py logs/flash/*.log` prints the per-kernel tables.

## Getting nixnan to run on PyTorch at all

The first direct runs never got past start-up (20+ minutes, 27 GB RSS). Three
causes, each fixed in this directory:

* **cuBLAS instrumentation cost:** about 30 s per new cuBLAS/CUTLASS kernel
  under NVBit. The fix is `FUNCTION_WHITELIST`. The runs instrument only the
  kernels under study: flash, plus bitsandbytes or cuBLAS for step 10.
  `run_all.sh` first runs each target with a dummy whitelist, to log the exact
  kernel names nixnan matches against (full demangled signatures; a full tile
  is a different template instance than a partial one).
* **`F.embedding` hangs NVBit at load time,** even when not instrumented.
  So does the CUB radix sort used by `weight[idx]`'s backward. TinyGPT computes
  the embedding as `one_hot(idx) @ weight` instead. That is bit-identical in
  FP32 without TF32, and the checkpoint is unchanged.
* **`LD_PRELOAD` must not cover a `timeout` wrapper:** write
  `timeout N env LD_PRELOAD=... python`. Otherwise both processes write the
  same nixnan `LOGFILE`.

## Files

| file | step |
|---|---|
| `PLAN.md` | the plan, and the source reading that motivated steps 9–11 |
| `gen_training_set.py`, `training-set.txt` | 2 |
| `tinygpt.py` | 3: model, `complete()`, oracle hook |
| `train.py`, `tinygpt.pt` | 4: training and checkpoint (`tinygpt-nixnan.pt` is the copy trained under nixnan) |
| `generate.py` | 5, 8, 10: completion, `--linear fp32/fp16/nf4` |
| `experiment_lse.py` | 9: A/B combine-guard experiment, `--oracle` |
| `summarize.py` | per-kernel summaries of nixnan logs |
| `run_all.sh` | every run above, end to end |
| `flash-kernels.txt`, `wl-nf4.txt`, `wl-fp16.txt` | nixnan whitelists |
