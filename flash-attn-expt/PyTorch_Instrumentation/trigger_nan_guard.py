#!/usr/bin/env python3
"""Make the combine guard's `lse_sum != lse_sum` disjunct fire, and show what does not.

    lse_logsum = (lse_sum == 0.f || lse_sum != lse_sum) ? INFINITY : logf(lse_sum) + lse_max;

lse_sum = sum_s expf(lse_s - lse_max) is NaN only if some split's LSE is +inf
or NaN (inf - inf). softmax.h writes lse = -inf whenever the split's row sum
is 0 or NaN, so the split must end with a row sum of +inf, which needs
exp2f(x*s - max*s) > 2^128 for some score x. With s > 0 that argument is <= 0.
With s < 0 it is (max - x)*|s| >= 0, and a masked score (x = -inf) gives
exactly +inf.

One decode-shaped call (q_len 1, k_len 301 -> 2 KV splits, 4 heads) per case.
Every case is also run through PyTorch's own kernel
(aten::_scaled_dot_product_flash_attention, scale=...) and compared NaN-aware.

    python trigger_nan_guard.py      # -> traces/trigger/REPORT.txt
"""
import os

import torch

import instr_flash as IF

os.makedirs(os.path.join(IF.HERE, "traces", "trigger"), exist_ok=True)
rep = open(os.path.join(IF.HERE, "traces", "trigger", "REPORT.txt"), "w")


def out(s=""):
    print(s)
    rep.write(s + "\n")


def same(a, b):
    """Bitwise equal, NaN positions included."""
    return torch.equal(a.isnan(), b.isnan()) and torch.equal(torch.nan_to_num(a, 0.0, 1.0, -1.0),
                                                            torch.nan_to_num(b, 0.0, 1.0, -1.0))


torch.manual_seed(0)
H, Tk = 4, 301
q = torch.randn(1, H, 1, 32, device="cuda").half()
k = torch.randn(1, H, Tk, 32, device="cuda").half()
v = torch.randn(1, H, Tk, 32, device="cuda").half()
k_neg = k.clone()
k_neg[0, 0, :256] = -q[0, 0, 0]          # head 0: every key of split 0 scores -|q|^2 (about -37)

cases = [
    ("default scale 1/sqrt(32)", k, None),
    ("scale 1e30", k_neg, 1e30),
    ("scale 1e37: max*s overflows to -inf", k_neg, 1e37),
    ("scale -0.1", k, -0.1),
    ("scale -1", k, -1.0),
    ("scale -10", k, -10.0),
]
GUARD = {0: "ok", 1: "sum==0", 2: "sum NaN", 3: "empty"}
for name, kk, scale in cases:
    o, lse, d = IF.flash(q, kk, v, False, instr=True, scale=scale)
    ro, rl = torch.ops.aten._scaled_dot_product_flash_attention(q, kk, v, 0.0, False, False, scale=scale)[:2]
    sp = d["split"].cpu()
    sp = sp[(sp[:, 0] == 1) & (sp[:, 4] == 1)]
    cb = d["comb"].cpu()
    cb = cb[cb[:, 2] == 1]
    out(f"=== {name}   (num_splits {d['num_splits']}; output and LSE equal to PyTorch's kernel, NaN-aware: "
        f"{same(o, ro) and same(lse, rl)})")
    out("  split kernel (softmax.h), head: split -> row_max, row_sum, lse, guard")
    for h in range(H):
        rows = sorted((r for r in sp.tolist() if int(r[2]) == h), key=lambda r: r[1])
        out(f"    head {h}: " + ";  ".join(f"s{int(r[1])}: {r[5]:.4g}, {r[6]:.4g}, {r[7]:.4g}, {GUARD[int(r[8])]}"
                                       for r in rows))
    out("  combine kernel, head: lse_split0 lse_split1 lse_max lse_sum | sum==0 sum!=sum fired | stored  non-finite O")
    for r in cb.tolist():
        out(f"    head {int(r[1])}: {r[3]:.4g} {r[4]:.4g} {r[5]:.4g} {r[6]:.4g} | {int(r[7])} {int(r[8])} {int(r[9])} | "
            f"{r[11]:.4g} {int(r[12])}")
    out(f"  -> lse_sum != lse_sum fired on {int(cb[:, 8].sum())} of {len(cb)} rows; "
        f"lse_sum == 0 on {int(cb[:, 7].sum())}\n")
rep.close()
