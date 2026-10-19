# Plan: nixnan walk-through deck (ordinary SASS, then HMMA)

Purpose: introduce how nixnan works at the SC26 tutorial, from loading until the
first HMMA is instrumented and its result reaches the CPU.

## Sources (read in full; nixnan branch print_histo_instrn, ce35185)
1. src/nixnan.cu – callbacks, instrument_function, recv thread, report
2. src/inject_funs.cu – device function nixnan_check_regs
3. src/instruction_info.cu / .cuh – get_reginfo, get_regs, reginsertion lambdas
4. src/instructions.h / instructions.json – opcode -> operand layout table (embedded as bytes)
5. src/common.cuh – exception flags, type codes, OPERANDS/OPBITS/EXCEBITS
6. src/exception_info.cuh – the packet sent GPU -> CPU
7. src/fp_utils.cuh – half2/bf162/float/double classifiers
8. src/recording.cu / .h – recorder: host table + device counters
9. src/reginfo.cuh – the 32-bit descriptor
+ nvbit_release/core/utils/channel.hpp (ChannelDev/ChannelHost, doorbell)

## Deck outline (latex/nixnan-walkthru-tc-rd_nixnan.tex)
0. Setup and the LD_LIBRARY_PATH pitfall (`env -u LD_LIBRARY_PATH`)
1. Callback architecture (diagram)
2. Lifecycle: at_init, tool_init, cuda_event entry/exit, ctx_term
3. First-launch instrumentation flow (diagram)
4. Compact encoding: instructions.json -> reginfo bitfield (diagram), reginsertion lambdas, recorder entry / inst_id
5. The two insert_call sites and the argument layout
6. Device side: va_arg, classify, shuffle-OR, atomicAdd record, first-occurrence push
7. Channel: push, flush, doorbell, recv; recv-thread state machine (diagrams); skip=1 duplicate; sentinel (design vs. current status)
8. End-of-run report
9. Part A: ordinary SASS (FFMA / HFMA2 from rd_nixnan), real output
10. Part B: HMMA.16816.F16 in rd_tc: register map (diagram), AFTER vs BEFORE, real report lines, F2FP blind spot

## Report (latex/nixnan-walkthru-report.tex): prose for every slide.

## Commands
    env -u LD_LIBRARY_PATH LINE_INFO=1 PRINT_ILL_INSTR=1 TOOL_VERBOSE=1 \
      LOGFILE=latex/data/<x>.log LD_PRELOAD=~/repos/nixnan/nixnan.so <binary>
    cuobjdump -sass <binary>
    cd latex && pdflatex <tex> (twice)

## Open points for refinement
- Sentinel flush kernel is commented out in this commit; deck states the status as-is.
- Whether to add histogram path (fp-histogram.cu) and memory instrumentation.
