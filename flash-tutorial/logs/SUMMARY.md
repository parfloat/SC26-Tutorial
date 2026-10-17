## select

* whitelist: `flash-kernels-all.txt`, SAMPLING=50; exit 0, 401 s
* program: `step     1  loss 2.5254  (145.3s)`
* program: `train next-token accuracy 0.0849; words reproduced exactly 0/100`
* kernels launched: 60; **instrumented (4)**: `flash_bwd_convert_dq_kernel`, `flash_bwd_dot_do_o_kernel`, `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel`, `flash_fwd_kernel (causal)`
* exception reports: 1158

| kernel | instruction | value | type | reports |
|---|---|---|---|---:|
| `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` | `FFMA` | -infinity | f32 | 128 |
| `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` | `FMUL` | -infinity | f32 | 128 |
| `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` | `HMMA.16816.F32` | subnormal | f16 | 88 |
| `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` | `FFMA` | infinity | f32 | 64 |
| `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` | `MUFU.EX2` | -infinity | f32 | 64 |
| `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` | `FMUL` | infinity | f32 | 8 |
| `flash_fwd_kernel` | `FFMA` | -infinity | f32 | 236 |
| `flash_fwd_kernel` | `FMUL` | -infinity | f32 | 236 |
| `flash_fwd_kernel` | `MUFU.EX2` | -infinity | f32 | 118 |
| `flash_fwd_kernel` | `HMMA.16816.F32` | subnormal | f16 | 88 |

nixnan final report (nonzero): FP16 Subnormal: 176 sites (42448 repeats); FP32 Infinity: 72 sites (201528 repeats); FP32 -Infinity: 910 sites (1904690 repeats)

## smoke

* whitelist: `flash-kernels.txt`, SAMPLING=50; exit 0, 12 s
* program: `flash output: 32 NaN of 768`
* kernels launched: 5; **instrumented (1)**: `flash_fwd_kernel (causal)`
* exception reports: 1166

| kernel | instruction | value | type | reports |
|---|---|---|---|---:|
| `flash_fwd_kernel` | `FADD` | NaN | f32 | 272 |
| `flash_fwd_kernel` | `FFMA` | -infinity | f32 | 256 |
| `flash_fwd_kernel` | `FMUL` | -infinity | f32 | 256 |
| `flash_fwd_kernel` | `MUFU.EX2` | -infinity | f32 | 128 |
| `flash_fwd_kernel` | `HMMA.16816.F32` | NaN | f32 | 126 |
| `flash_fwd_kernel` | `FMUL` | NaN | f32 | 64 |
| `flash_fwd_kernel` | `FFMA` | infinity | f32 | 32 |
| `flash_fwd_kernel` | `FFMA` | NaN,-infinity | f32 | 8 |
| `flash_fwd_kernel` | `HMMA.16816.F32` | NaN | f16 | 8 |
| `flash_fwd_kernel` | `MUFU.EX2` | NaN,-infinity | f32 | 4 |
| `flash_fwd_kernel` | `MUFU.EX2` | NaN | f32 | 4 |
| `flash_fwd_kernel` | `HMMA.16816.F32` | NaN,infinity | f32 | 3 |
| `flash_fwd_kernel` | `HMMA.16816.F32` | infinity | f16 | 2 |
| `flash_fwd_kernel` | `FMUL` | infinity | f32 | 2 |
| `flash_fwd_kernel` | `FFMA` | NaN,infinity | f32 | 1 |

nixnan final report (nonzero): FP16 NaN: 8 sites (24 repeats); FP16 Infinity: 2 sites (6 repeats); FP32 NaN: 482 sites (1430 repeats); FP32 Infinity: 38 sites (0 repeats); FP32 -Infinity: 656 sites (9587 repeats)

## gen312

* whitelist: `flash-kernels.txt`, SAMPLING=50; exit 0, 148 s
* program: `completion: 312 -> 312323   (tokens A312 -> A312323)`
* kernels launched: 31; **instrumented (2)**: `flash_fwd_kernel (non-causal)`, `flash_fwd_kernel (causal)`
* exception reports: 1288

| kernel | instruction | value | type | reports |
|---|---|---|---|---:|
| `flash_fwd_kernel` | `FFMA` | -infinity | f32 | 512 |
| `flash_fwd_kernel` | `FMUL` | -infinity | f32 | 512 |
| `flash_fwd_kernel` | `MUFU.EX2` | -infinity | f32 | 256 |
| `flash_fwd_kernel` | `HMMA.16816.F32` | subnormal | f16 | 8 |

nixnan final report (nonzero): FP16 Subnormal: 8 sites (24 repeats); FP32 -Infinity: 1280 sites (19200 repeats)

## train3

* whitelist: `flash-kernels.txt`, SAMPLING=50; exit 0, 401 s
* program: `step     1  loss 2.5254  (151.0s)`
* program: `train next-token accuracy 0.0849; words reproduced exactly 0/100`
* kernels launched: 60; **instrumented (2)**: `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel`, `flash_fwd_kernel (causal)`
* flash kernels launched but not instrumented: `flash_bwd_convert_dq_kernel`, `flash_bwd_dot_do_o_kernel`
* exception reports: 1142

| kernel | instruction | value | type | reports |
|---|---|---|---|---:|
| `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` | `FFMA` | -infinity | f32 | 128 |
| `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` | `FMUL` | -infinity | f32 | 128 |
| `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` | `HMMA.16816.F32` | subnormal | f16 | 88 |
| `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` | `FFMA` | infinity | f32 | 64 |
| `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` | `MUFU.EX2` | -infinity | f32 | 64 |
| `flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel` | `FMUL` | infinity | f32 | 8 |
| `flash_fwd_kernel` | `FFMA` | -infinity | f32 | 236 |
| `flash_fwd_kernel` | `FMUL` | -infinity | f32 | 236 |
| `flash_fwd_kernel` | `MUFU.EX2` | -infinity | f32 | 118 |
| `flash_fwd_kernel` | `HMMA.16816.F32` | subnormal | f16 | 72 |

nixnan final report (nonzero): FP16 Subnormal: 160 sites (42432 repeats); FP32 Infinity: 72 sites (201528 repeats); FP32 -Infinity: 910 sites (1904690 repeats)

