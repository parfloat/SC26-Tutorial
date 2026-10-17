# flash-tutorial: TinyGPT for a step-by-step flash-attention tutorial

A fresh, self-contained experiment. TinyGPT is a 4-layer decoder-only transformer
over digit strings. Its attention runs through PyTorch's flash-attention
kernels on tensor cores, and nixnan can watch those kernels.

No virtual environment is needed: it uses the system `python3` and a
user-level PyTorch install.

## Requirements

### PyTorch (`requirements.txt`)

| | |
|---|---|
| Python | ≥ 3.9 (tested: system `python3` 3.10.12) |
| PyTorch | **2.3.1, CUDA 12.1 wheel** (`torch==2.3.1+cu121`) |
| install | `python3 -m pip install --user -r requirements.txt` |
| GPU | compute capability ≥ 8.0 (Ampere or newer). PyTorch's flash attention exists only for FP16/BF16 on tensor cores at sm_80+. Tested: RTX 3090 (sm_86) |
| NVIDIA driver | one that supports CUDA 12.1 (≥ 530). Tested: 595.84 |
| CUDA toolkit | not needed to run: the wheel ships its own CUDA runtime, cuBLAS and cuDNN |

### nixnan (`../external/nixnan`, git submodule)

| | |
|---|---|
| nixnan | `parfloat/nixnan`, branch `print_histo_instrn` (commit `ce35185`), built as `external/nixnan/nixnan.so` |
| NVBit | 1.8 (downloaded by nixnan's `make`) |
| build | `cd ../external/nixnan && make` — needs `nvcc` from CUDA ≥ 12.0 (tested 12.4) and GCC ≥ 8.5 |
| run | `LD_PRELOAD=nixnan.so python3 ...`, wrapped by `nixnan.sh` |
| driver | NVBit 1.8 documents driver ≤ 575.xx; **595.84 tested working** (`check_env.py --nixnan`) |
| CUDA | NVBit needs CUDA ≥ 12.0: the cu121 wheel satisfies it |

nixnan uses the CUDA runtime bundled with the PyTorch wheel; nothing else from
PyTorch is needed.

## Quick start

```
python3 check_env.py --nixnan    # Python, torch, GPU, flash on tensor cores, nixnan sees HMMA (~3 min)
python3 test_tinygpt.py          # 8 tests: tokens, checkpoint, completion, KV cache, memorisation
python3 generate.py              # 312 -> 312323
./nixnan.sh gen312 generate.py   # the same under nixnan (~2.5 min) -> logs/gen312.log
```

## TinyGPT

| | |
|---|---|
| tokens | 12, written as hex digits: `0`–`9`, start symbol `A` (id 10), end symbol `F` (id 11). The word `5881` is the sequence `A5881F` |
| data | `training-set.txt`: 100 random words of length 1–20 (`gen_training_set.py`, seed 2026) |
| model | 4 blocks of `x += Attn(LN(x)); x += MLP(LN(x))`, d_model 128, 4 heads × 32, RoPE, MLP 4×, no biases, 791,808 parameters |
| precision | FP32 everywhere, TF32 off, so the Linear layers stay off tensor cores. Attention casts Q, K, V to FP16 and calls `F.scaled_dot_product_attention` with only the flash backend allowed |
| training | `train.py`: 1,500 full-batch AdamW steps, ~1 min. Loss 0.3939, next-token accuracy 0.8353 (the first digit of a word is unpredictable). Bitwise reproducible |
| completion | `generate.py`: greedy, KV cache (one prefill call, then one call per token). `312` → `312323` |

## Running nixnan

`nixnan.sh NAME SCRIPT.py [ARGS...]` runs the script under nixnan with
`FUNCTION_WHITELIST=flash-kernels.txt`, a timeout (`TIMEOUT`, default 900 s),
and any nixnan variable you set (`SAMPLING`, `INSTR_MEM`, `HISTOGRAM`, …).
The log goes to `logs/NAME.log`.

* **Cost.** NVBit scans every function in PyTorch's GPU binaries as they
  load: a fixed ~2.5 min per process (RTX 3090), even with nothing
  instrumented. The instrumentation itself is cheap for these kernels, so
  `SAMPLING` matters only for long runs.
* **Which kernels.** nixnan's whitelist needs exact demangled names.
  `flash-kernels.txt` holds the flash kernels chosen for capture (see below).
  If a run launches a flash kernel not in the list, `nixnan.sh` says so
  and `LEARN=1` adds it.

### The flash kernels this model launches (PyTorch 2.3.1, head dim 32)

Measured with `SAMPLING=50 ./nixnan.sh train3 train.py --steps 3` (all five
instrumented once), then `generate.py`. `flash-kernels.txt` keeps the **three
kernels around the softmax**, which are also the ones where nixnan reports
exceptional values:

| kernel | when | what it computes | reports (3 training steps) | captured |
|---|---|---|---|---|
| `flash_fwd_kernel` (causal) | training, prefill | QKᵀ, mask, online softmax, PV, lse | 662: −inf (mask into the softmax's `FFMA`, `MUFU.EX2`), FP16 subnormals in `HMMA` | yes |
| `flash_fwd_kernel` (non-causal) | decode steps (1 query) | the same, no mask | not launched in training. In `generate.py` the two forward variants together give 1,288 reports (−inf, subnormal); nixnan's reports name the kernel without its template arguments, so they cannot be split by variant | yes |
| `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` | training | recomputes the softmax P = exp(S − lse), then dQ, dK, dV | 480: −inf, **+inf** (`FFMA`, `FMUL`), subnormals in `HMMA` | yes |
| `flash_bwd_dot_do_o_kernel` | training | rowsum(dO · O) for the softmax gradient | 0 | no |
| `flash_bwd_convert_dq_kernel` | training | FP32 dQ accumulator → FP16 | 0 | no |

Run times with these three captured (`SAMPLING=50`): `generate.py` 148 s,
`train.py --steps 3` 400 s.

## Files

| file | |
|---|---|
| `tinygpt.py` | model, tokens (`encode`/`decode`), `complete()`, `load()` |
| `gen_training_set.py`, `training-set.txt` | the training data |
| `train.py`, `tinygpt.pt` | training and the checkpoint |
| `generate.py` | completion |
| `test_tinygpt.py` | tests |
| `check_env.py`, `smoke_flash.py` | environment check; the smallest flash call that makes inf/NaN inside the kernel |
| `nixnan.sh`, `flash-kernels.txt` | nixnan wrapper and the kernels it instruments |
| `requirements.txt` | PyTorch pin |
