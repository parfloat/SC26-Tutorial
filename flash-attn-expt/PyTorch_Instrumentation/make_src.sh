#!/usr/bin/env bash
# Regenerate src/ from PyTorch v2.6.0: copy the vendored flash-attention files TinyGPT's
# kernels need, rename namespace pytorch_flash -> instr_flash, make includes local,
# then apply instrumentation.patch (the dump hooks, the hdim-32-only dispatch, the binding).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
T=$(mktemp -d); trap 'rm -rf $T' EXIT
git clone -q --depth 1 --branch v2.6.0 --filter=blob:none --sparse https://github.com/pytorch/pytorch.git $T/pt
git -C $T/pt sparse-checkout set aten/src/ATen/native/transformers/cuda/flash_attn
F=$T/pt/aten/src/ATen/native/transformers/cuda/flash_attn
mkdir -p src
for f in alibi.h block_info.h dropout.h flash.h flash_fwd_kernel.h flash_fwd_launch_template.h kernel_traits.h \
         mask.h philox.cuh rotary.h softmax.h static_switch.h utils.h \
         kernels/flash_fwd_hdim32_fp16_sm80.cu kernels/flash_fwd_split_hdim32_fp16_sm80.cu; do cp $F/$f src/; done
cp $F/flash_api.cpp src/flash_api.cu
sed -i 's#<ATen/native/transformers/cuda/flash_attn/\([^>]*\)>#"\1"#; s/pytorch_flash/instr_flash/g; s/pytorch_fmha/instr_flash/g' src/*
patch -p0 < instrumentation.patch
