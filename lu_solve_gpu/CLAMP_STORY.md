# The ε-clamp story: a silent failure, a defensive fix, a loud failure

Slides: `LU_Slides.pdf` pages 5, 6, 8, 10 and 11 are new. NEWLU.pdf's five slides are pages 2, 3, 4, 7 and 9 (unchanged), page 1 is the cover.
Reproduce on the machine: `./run_clamp_story.sh` (about 5 minutes; `step1`, `step2`, `fp32`, `n128` run one part).
Everything below was measured on an RTX 3090 (sm_86), CUDA 12.4, with plain `LD_PRELOAD=nixnan.so` and no other nixnan options
(except the one detail run in step 2). Logs are in `logs/clamp_story/`.

## The story

1. **Version 1, no clamp (FP64, 20×20).** `matrix.csv` solves cleanly (residual 9.7e-16). `bad_matrix.csv` returns finite numbers up to
   1.7e17 and a residual of **33.9**. The program exits 0. nixnan reports **nothing**: no NaN, Inf or division by zero was produced.
   The smallest pivot read back from the GPU is 6.9e-17, not zero, so the division gives a huge finite number.
2. **Version 2, the designer adds a clamp.** In `eliminate_kernel`, any value with `|val| < 10·DBL_EPSILON` (2.2e-15) is set to 0.0.
   That turns the 6.9e-17 pivot into exactly 0.0. Division by it gives ±Inf, Inf×0 gives NaN, and the whole solve becomes NaN.
   nixnan now reports FP64 NaN 316 (2171 repeats), Inf 4, division-by-zero 4 (the same numbers as the existing Stage 1 slide).
3. **Following nixnan back (step 2's detail run).** The first report is `div0` at `MUFU.RCP64H` in `compute_multipliers_kernel`,
   source line 216, the division `U[j][k] / U[k][k]`. The 331 reports then spread through `eliminate`, `find_pivot`, forward and
   backward substitution. nixnan shows where the zero is divided by. The cause (the clamp wrote the zero) is upstream of it.
4. **FP32 and 128×128.** The same two steps, with the same outcome in every configuration:

   | bad matrix | no clamp: pivot | residual | nixnan | clamp: pivot | residual | nixnan |
   |---|---|---|---|---|---|---|
   | FP64 20×20 | 6.9e-17 | 33.9 | none | 0.0 | NaN | NaN 316, Inf 4, Div0 4 |
   | FP32 20×20 | 3.0e-8 | 26.2 | none | 0.0 | NaN | NaN 253, Inf 8, Div0 2 |
   | FP64 128×128 | 1.1e-15 | 219 | none | 0.0 | NaN | NaN 142, Inf 19, −Inf 2, Div0 2 |
   | FP32 128×128 | 1.0e-6 | 92.8 | none | 0.0 | NaN | NaN 107, Inf 21, −Inf 2, Div0 1 |

   The well-conditioned matrices (`matrix.csv`, `matrix_128.csv`) are clean in all eight configurations.
5. **What it teaches.** nixnan finds *exceptions*, not ill-conditioning: version 1 is invisible to it. A residual check catches both
   versions; a pivot or `rcond` check would catch the problem before dividing. The clamp did not fix the matrix, it moved the failure
   from "silently wrong" to "divide by 0.0". A version 3 with a relative pivot test and an error return is **proposed on the slides
   and not implemented here**.

## Files

| file | what |
|---|---|
| `lu_solve_gpu_clamp_study.cu` | the solver, built four ways with `-DREAL=double\|float -DCLAMP=1\|0`; prints the GPU's smallest pivot |
| `run_clamp_story.sh`, `story_summary.py` | the reproduction script and its summary table |
| `gen_matrix_n.py` | makes `matrix_128.csv` and `bad_matrix_128.csv` (seed 1) and prints their condition numbers |
| `logs/clamp_story/` | `<fp64\|fp32>_clamp<on\|off>_<matrix>_{native,nixnan}.log`, and `fp64_clampon_bad_matrix_detail.log` |
| `make_LU_slides_pptx.py` → `LU_Slides.pptx` → `LU_Slides.pdf` | the deck; the new slides are the `slide_story_*` functions |

## Caveats

- **Clamp threshold luck at 128×128 / FP32.** The bad pivot (1.03e-6) is just below the FP32 threshold (1.19e-6). A different seed
  could give a pivot above it, and then the clamped run would behave like the unclamped one. One seed was run.
- **Condition numbers.** `bad_matrix_128.csv` has cond ≈ 3.05e17, computed in 80-digit arithmetic for the matrix as stored in double.
  `bad_matrix.csv`'s ≈ 2.3e17 comes from a double-precision SVD, which cannot resolve σ_min below ε·σ_max, so the two numbers are
  comparable in size only. The FP32 solver rounds the matrix to float first, so it solves a different matrix (condition number not computed).
- **The 6 FP32 NaNs in the FP64 clamp run** are the `FFMA R8, RZ, R5, R3` register-move artifact explained on the existing slide.
- **Residual** is computed on the host in double against the original double matrix, so FP32 residuals measure the FP32 answer honestly.
- **Single GPU.** Only sm_86 was run. nixnan counts can vary with the compiler's expansion of division; the ones above are from one machine.
