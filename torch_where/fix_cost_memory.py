"""Where does the extra memory of each fix come from?  (companion to fix_cost_bench.py)
For each variant: (1) the tensors autograd saves for backward (via saved_tensors_hooks),
(2) memory live after forward, (3) peak during backward.  N = 2^24 float32 (64 MiB per float tensor,
16 MiB per bool tensor)."""
import torch
from fix_cost_bench import baseline, double_where, custom_function, sync_mask_check  # noqa  (re-uses variants)

dev = "cuda"
N = 1 << 24
MiB = 2**20
a0 = torch.rand(N, device=dev) * 20 - 10


def analyze(name, f):
    torch.cuda.synchronize()
    a = a0.clone().requires_grad_(True)
    saved = {}
    def pack(t):
        saved[(t.data_ptr(), t.dtype)] = (tuple(t.shape), t.dtype, t.numel() * t.element_size())
        return t
    def unpack(t):
        return t
    torch.cuda.synchronize(); base_alloc = torch.cuda.memory_allocated()
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        out = f(a)
    torch.cuda.synchronize()
    after_fwd = torch.cuda.memory_allocated() - base_alloc
    torch.cuda.reset_peak_memory_stats()
    pk0 = torch.cuda.max_memory_allocated()
    out.backward()
    torch.cuda.synchronize()
    peak_bwd = torch.cuda.max_memory_allocated() - base_alloc
    tot = sum(v[2] for v in saved.values())
    kinds = {}
    for shape, dt, nb in saved.values():
        kinds[str(dt).replace("torch.", "")] = kinds.get(str(dt).replace("torch.", ""), 0) + nb / MiB
    print(f"{name:26s} saved-for-backward {tot/MiB:7.1f} MiB {dict((k, round(v)) for k, v in kinds.items())}"
          f" | live after fwd {after_fwd/MiB:7.1f} | peak in bwd {peak_bwd/MiB:7.1f} (MiB above input a)")
    del out, a, saved


print("torch", torch.__version__, torch.cuda.get_device_name(0), "N=2^24; float tensor = 64 MiB, bool = 16 MiB")
for name, f in [("baseline where", baseline), ("double where", double_where),
                ("custom Function fix", custom_function), ("mask check + .item() sync", sync_mask_check)]:
    analyze(name, f)


# --- attribute backward allocations >= 16 MiB to ops (torch profiler, profile_memory) ---
from torch.profiler import profile, ProfilerActivity

def allocs(name, f):
    a = a0.clone().requires_grad_(True)
    out = f(a)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
        out.backward()
        torch.cuda.synchronize()
    rows = []
    for e in prof.events():
        m = getattr(e, "cuda_memory_usage", 0)
        if m and abs(m) >= 16 * MiB and e.device_type == torch.autograd.DeviceType.CPU:
            rows.append((e.name, m / MiB))
    print(f"-- backward allocations >= 16 MiB: {name}")
    tot = 0
    for n, m in rows:
        print(f"   {n:40s} {m:+8.1f} MiB")


print()
for name, f in [("baseline where", baseline), ("double where", double_where), ("custom Function fix", custom_function)]:
    allocs(name, f)
