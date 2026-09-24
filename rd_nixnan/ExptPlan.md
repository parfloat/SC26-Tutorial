* All files and folders created for this experiment must go into Expts-9-23/ (make it)
  - Create shell-scripts, viz/ latex/ slides/ log/ here
  - [Begin New]
    + Include in the Latex these details in an earlier section talking about
    the overall NixNan features:
    + Meaning of {"count":128,"bf16":[],"f16":[[0,5],[-4,-1]],"f32":[],"f64":[] }
    + SAMPLING = 8 - Instrument a repeat kernel every SAMPLING times
    + How the parameters of exponential decay are specified in a NixNan run, and specific details
    + Summarize what NixNan settings generated the contents of 03_sampling_exp_decay_summary.tsv
    + A plot of how various precisions ended up generating exceptions (each precision in a different color; x-axis is step count. Since FP32 generated exceptions at step 1800, scale the plot to show 1800 as the full x-axis extent, and showing others in relation. E.g. in FP16, at step 200, we got this.
    + The CUDA statement where INF was first generated
    + Any reasoning wrt INF turning into NaN (diagnose if you can, showing the specific CUDA statement)
  - [End New]
  - [Done, latex/report.tex] All of the above is now in the report:
    + New early section "NixNan Feature Reference" (right after the "Straight
      nixnan" mechanics section, before the Sampling results): explains the
      BIN_SPEC_FILE JSON exactly (unbiased exponent ranges per format, the
      "count" report-frequency threshold, and that zero/infinity are always
      excluded -- the reason a hard jump to inf is invisible to this
      mechanism, used later in the diagnosis below); SAMPLING=8's semantics
      verbatim from nixnan's own banner; that there is no native nixnan
      setting for a decaying rate -- SAMPLING is read once per process, so
      the schedule lives entirely in sampling_schedule.json and is executed
      as 32 independent process launches (4 levels x 4 snaps x 2 cycles);
      and that those 32 runs are otherwise identical to Section 4's default
      exception-instrumentation baseline.
    + New subsection "Where Infinity Actually Comes From" (in the "Making
      BF16 and FP32 Fail Too" section): the requested max|u| vs step plot
      (viz/inf_onset_vs_step.{svg,pdf}), x-axis scaled to step 1800 as
      specified, built from the real 100-step checkpoints in
      Logs/05_extended_M2000.log -- FP16 breaks off the shared curve at step
      300; BF16/FP32/FP64 stay numerically indistinguishable through step
      1800 and BF16/FP32 are non-finite by step 1900 (FP64's apparent
      step-1900 non-finite in the same log turned out to be a diagnostic
      bug, not real -- see below).
    + The CUDA statement: rd_nixnan.cu was rebuilt with -g -lineinfo (00_build.sh,
      05_extended_run_all_fail.sh) specifically so nixnan's LINE_INFO=1 could
      resolve real source locations. Result: Infinity first appears at
      rd_nixnan.cu:96, the Laplacian (float lap = (u[i+1]+u[i-1]) - 2.f*u[i]),
      NOT the reaction term -- confirmed by nixnan itself for FP32 (BF16
      partially; FP16 not at all, since its __h* intrinsics don't resolve to
      a line in this build -- a real nixnan limitation, reported as such, with
      the FP16 conclusion drawn by structural analogy to FP32/BF16 instead).
    + INF -> NaN reasoning: nixnan's own instruction-level trace shows the
      exact FADD computing (u[i+1]+u[i-1]) - 2.f*u[i] as literally inf - inf,
      confirming the source file's own long-standing comment
      ("inf - 2*inf + inf -> NaN") to the instruction, not just by hand.
    + Bonus, found while doing this: FP64's max|u| at step 1800 is identical
      to FP32's (1.1675e38) and nowhere near FP64's real ceiling (~1.8e308);
      steady 1.05x/step growth cannot explain FP64 also going non-finite by
      step 1900. Reported as an open anomaly, not resolved here.
  - [Done] Not actually an anomaly: rd_nixnan.cu's own check narrowed a
    double into a float before calling isfinite() -- for FP64, once a real
    value passed float's max (~3.4e38) while still a finite double, that
    cast alone fabricated a false +Infinity. Fixed (double throughout;
    audited for similar casts, none found; commit 03c9ef3). With that
    fixed, added an off-by-default FP64_LONGRUN build mode that suppresses
    FP16/BF16/FP32 after step 3800 and steps FP64 alone until it, too,
    goes non-finite for real: step 14600, with nixnan now correctly
    classifying it (shell-scripts/06_fp64_longrun.sh, same Laplacian line
    and inf-inf->NaN mechanism as FP32). Details in the report.
* Make this entire set of steps repeatable thru a RunAll.sh
  - Do the experiments, writing all outputs to a suitable log file named wrt the expts
    + Save the logs in a Logs/ folder
    + Run the code on a reasonably sized grid using "straight nixnan"
      - LD_PRELOAD=<path>/nixnan.so with suitable env vars set
        + Goals
	  - show sampling
	    + every kernel call
	    + then once in 8 kernel calls
	    + then exponentially decaying sampling rate
	      - step 4 times, then double spacing, do like this 4 times, then reset
	        + write out the json file that was used for this, with comments at top
	  - show the histograms per precision bucket
	  - make BF16 and FP32 fail too (not just FP16)
	    + tried: domain enlargement (bigger N, dx held fixed) -- does NOT work.
	      The reaction term lambda*u[i] is a per-point multiplicative growth
	      factor, independent of N/dx entirely; already verified earlier this
	      session too (rd_nixnan_large.cu, N=1001/10x domain, still only FP16
	      failed at M=1600). Enlarging N just runs the same per-step growth
	      curve on more points -- it doesn't change when any point's exponent
	      is exhausted.
	    + works: run more time steps (M). Confirmed at M=2000 (up from 1600):
	      FP16 still fails earliest (step 300); BF16 and FP32 now also fail,
	      together, at step 1900, with real nixnan-classified NaN/Infinity
	      counts for all three. See Logs/05_extended_M2000.log.
	    + [Done] the FP64-at-step-1900 report above was a bug in the
	      program's own check (a double->float narrowing before
	      isfinite()), not a nixnan gap -- fixed in rd_nixnan.cu. FP64 does
	      overflow for real, just much later: step 14600, confirmed under
	      nixnan with real classified NaN/Infinity counts. See
	      shell-scripts/06_fp64_longrun.sh.
  - Write a Latex report of what was done (one concept/idea per section)
  - Make slides in latex/beamer
    + Explain example in one slide (very high level)
      - Include a discussion of how precision-sensitive this example is (two bullets)
    + 2-3 slides per concept/idea
  - Leave behind files that carry evidence (visualization, etc)
    + consult viz/ files so far
* Be able to show that nixnan helps understand many aspects of this example
