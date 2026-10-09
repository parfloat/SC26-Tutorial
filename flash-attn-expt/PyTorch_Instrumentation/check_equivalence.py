#!/usr/bin/env python3
"""Check that the instrumented kernels compute exactly what PyTorch's own flash
attention computes (aten::_scaled_dot_product_flash_attention, torch 2.6),
with the dumps on and off: bitwise-equal outputs and logsumexp on the valid
rows, for TinyGPT's shapes (non-split and split-KV paths)."""
import torch
import instr_flash

torch.manual_seed(0)
cases = [(4, 4, True), (1, 5, False), (1, 6, False), (16, 16, True), (100, 100, True),
         (301, 301, True), (1, 302, False), (1, 303, False)]
ok = True
for tq, tk, causal in cases:
    q = torch.randn(1, 4, tq, 32, device="cuda").half()
    k = torch.randn(1, 4, tk, 32, device="cuda").half()
    v = torch.randn(1, 4, tk, 32, device="cuda").half()
    ref_o, ref_lse = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, causal)[:2]
    for instr in (False, True):
        o, lse, d = instr_flash.flash(q, k, v, causal, instr=instr)
        same = torch.equal(o, ref_o) and torch.equal(lse, ref_lse)
        ok &= same
        print(f"q {tq:3d} k {tk:3d} causal {causal!s:5}  splits {d['num_splits']}  dumps {'on ' if instr else 'off'}"
              f"  bitwise equal to aten flash: {same}")
print("ALL EQUAL" if ok else "MISMATCH")
