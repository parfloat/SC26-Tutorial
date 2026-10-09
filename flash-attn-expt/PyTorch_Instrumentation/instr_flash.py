"""Instrumented flash attention for TinyGPT, without nixnan.

src/ is PyTorch v2.6.0's vendored flash-attention
(aten/src/ATen/native/transformers/cuda/flash_attn), namespace renamed
pytorch_flash -> instr_flash so it cannot collide with libtorch's copy, with
dump hooks marked "INSTRUMENTATION" (see instrumentation.patch). Only the
FP16 head-dim-32 kernels TinyGPT uses are compiled. The host side is
PyTorch's own mha_fwd, so the split heuristic, tile sizes and kernel
template choices are the ones SDPA makes.

    import instr_flash
    instr_flash.install()          # tinygpt.flash_attention -> instrumented kernel
    ... run the model ...
    instr_flash.CALLS              # one dict per attention call, with the dumps
"""
import math
import os
import sys

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
sys.path.insert(0, os.path.dirname(HERE))           # flash-attn-expt/ (tinygpt.py)
import tinygpt  # noqa: E402

# Must match src/instr_dump.h
M, N, D = 128, 128, 32
OFF = {}
_o = 0
for _name, _size in (("Q", M * D), ("K", N * D), ("V", N * D), ("S_RAW", M * N), ("S_MASK", M * N),
                     ("P", M * N), ("O", M * D), ("ROW", M * 4)):
    OFF[_name] = (_o, _size)
    _o += _size
HEAD = _o
SHAPE = {"Q": (M, D), "K": (N, D), "V": (N, D), "S_RAW": (M, N), "S_MASK": (M, N), "P": (M, N),
         "O": (M, D), "ROW": (M, 4)}
SPLIT_FIELDS = ["written", "split", "head", "row", "valid", "row_max", "row_sum", "lse", "guard"]
COMB_FIELDS = ["written", "row", "valid", "lse_split0", "lse_split1", "lse_max", "lse_sum", "sum_is_zero",
               "sum_is_nan", "guard_fired", "unguarded_logsum", "lse_logsum", "nonfinite_O"]

_ext = None
CALLS = []          # filled by attention() while recording
RECORD = True


def build(verbose=False):
    global _ext
    if _ext is None:
        os.system(os.path.join(HERE, "get_cutlass.sh"))
        src = os.path.join(HERE, "src")
        os.makedirs(os.path.join(HERE, "build"), exist_ok=True)
        _ext = load(
            name="instr_flash",
            sources=[os.path.join(src, f) for f in ("binding.cpp", "flash_api.cu", "flash_fwd_hdim32_fp16_sm80.cu",
                                                    "flash_fwd_split_hdim32_fp16_sm80.cu")],
            extra_include_paths=[src, os.path.join(HERE, "third_party/cutlass/include")],
            extra_cflags=["-O3", "-std=c++17", "-DUSE_FLASH_ATTENTION"],
            extra_cuda_cflags=["-O3", "-std=c++17", "-DUSE_FLASH_ATTENTION", "--expt-relaxed-constexpr",
                               "--expt-extended-lambda",
                               "-gencode", "arch=compute_86,code=sm_86",
                               "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                               "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"],
            build_directory=os.path.join(HERE, "build"),
            verbose=verbose)
    return _ext


def flash(q, k, v, causal, instr=True):
    """q, k, v: FP16 (B, H, T, D) like F.scaled_dot_product_attention. Returns (out, lse, dumps)."""
    ext = build()
    o, lse, tile, split, comb, ns = ext.fwd(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                            causal, 1.0 / math.sqrt(q.shape[-1]), instr)
    return o.transpose(1, 2), lse, dict(tile=tile, split=split, comb=comb, num_splits=ns)


def attention(q, k, v, causal, layer):
    out, lse, d = flash(q, k, v, causal, instr=RECORD)
    if RECORD:
        CALLS.append(dict(layer=layer, q_len=q.shape[2], k_len=k.shape[2], causal=causal,
                          num_splits=d["num_splits"], lse=lse.float().cpu(),
                          out=out.float().cpu(),
                          tile=d["tile"].cpu() if d["tile"].numel() else None,
                          split=d["split"].cpu() if d["split"].numel() else None,
                          comb=d["comb"].cpu() if d["comb"].numel() else None))
    return out


def install():
    build()
    tinygpt.flash_attention = attention


def tile(call, head, name, bits=False):
    """One dumped tile of a non-split call as a (rows, cols) tensor. Q, K and V are stored as raw
    FP16 bit patterns: returned as float16 values, or as the int32 bit patterns with bits=True."""
    o, n = OFF[name]
    t = call["tile"][head, o:o + n].view(*SHAPE[name])
    if name in ("Q", "K", "V"):
        b = t.to(torch.int32)
        return b if bits else b.to(torch.int16).view(torch.float16)
    return t
