#!/usr/bin/env python3
"""Smallest flash-attention call that produces an inf and NaNs inside the
kernel: used by check_nixnan.sh to confirm nixnan sees flash's tensor-core
(HMMA) instructions. One key coordinate is +inf, so the score matrix gets
+-inf, and the softmax and P*V produce NaN."""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.manual_seed(0)
q = torch.randn(1, 4, 6, 32, device="cuda", dtype=torch.float16)
k, v = torch.randn_like(q), torch.randn_like(q)
k[0, 0, 2, 0] = float("inf")
with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
torch.cuda.synchronize()
print(f"flash output: {int(o.isnan().sum())} NaN of {o.numel()}")
