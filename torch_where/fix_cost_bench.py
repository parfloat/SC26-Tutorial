"""Cost of the fix strategies discussed in pytorch/pytorch#68425, measured on this GPU.
Forward+backward of b = where(a<=0, exp(a), 1+a).sum() on N float32 elements.
Inputs are chosen finite so every variant does the same useful work; we time only overhead.
"""
import torch, time

torch.manual_seed(0)
dev = "cuda"
N = 1 << 24
iters = 50
a0 = (torch.rand(N, device=dev) * 20 - 10)          # values in [-10, 10]: finite exp


class SafeExp(torch.autograd.Function):
    """exp whose backward zeroes the gradient where the incoming grad is 0 (a 'fused' fix)."""
    @staticmethod
    def forward(ctx, x):
        y = x.exp()
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, g):
        (y,) = ctx.saved_tensors
        return torch.where(g == 0, torch.zeros_like(g), g * y)


def baseline(a):
    return torch.where(a <= 0, torch.exp(a), 1 + a).sum()


def double_where(a):
    m = a <= 0
    return torch.where(m, torch.where(m, a, torch.zeros_like(a)).exp(), 1 + a).sum()


def custom_function(a):
    return torch.where(a <= 0, SafeExp.apply(a), 1 + a).sum()


def sync_mask_check(a):
    m = a <= 0
    if (~m).any().item():            # data-dependent branch: forces a CPU-GPU sync
        pass
    return torch.where(m, torch.exp(a), 1 + a).sum()


def bench(f):
    for _ in range(5):
        a = a0.clone().requires_grad_(True); f(a).backward()
    torch.cuda.synchronize()
    a = a0.clone().requires_grad_(True)
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats()
    s.record()
    for _ in range(iters):
        a.grad = None
        f(a).backward()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters, torch.cuda.max_memory_allocated() / 2**20


print("torch", torch.__version__, torch.cuda.get_device_name(0), f"N={N}")
base_t, base_m = bench(baseline)
for name, f in [("baseline where", baseline), ("double where", double_where),
                ("custom Function fix", custom_function), ("mask check + .item() sync", sync_mask_check)]:
    t, m = bench(f)
    print(f"{name:28s} {t:8.3f} ms/iter  x{t/base_t:5.2f}   peak mem {m:8.1f} MiB  (+{m-base_m:6.1f})")
