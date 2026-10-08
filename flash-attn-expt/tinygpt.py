"""Step 3: TinyGPT, a decoder-only transformer over the digits 0..9.

    tokens 0..9 = digits, BOS = 10, EOS = 11
    4 blocks: x += Attn(LN(x)); x += MLP(LN(x))    (pre-LN, RoPE, no biases)
    d_model 128, 4 heads (head_dim 32), MLP 4x

Precision: parameters and Linear layers are FP32 with TF32 disabled, so the
GEMMs run on SIMT FFMA, not tensor cores. Attention casts Q/K/V to FP16 and
calls F.scaled_dot_product_attention with the FLASH backend forced. This is
the same dispatch HF models (BioMistral) use, landing in PyTorch's vendored
flash-attention (pytorch_flash::flash_fwd_kernel / flash_fwd_splitkv_kernel
+ flash_fwd_splitkv_combine_kernel). Flash exists only for FP16/BF16 on
tensor cores, so attention is the one place that uses HMMA.

Hooks for the experiments (all off by default):
  model.ext_embed  -- optional tensor (T, d_model) added to the token
                      embeddings: the "external input" of experiment_lse.py
  ORACLE           -- when set to a list, every attention call also appends
                      (layer, q_len, k_len, lse) using aten's flash op, which
                      returns flash's logsumexp (an +inf entry means the
                      combine guard in flash_fwd_kernel.h fired)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

VOCAB, BOS, EOS = 12, 10, 11
torch.backends.cuda.matmul.allow_tf32 = False      # FP32 GEMMs stay off tensor cores
torch.backends.cudnn.allow_tf32 = False

ORACLE = None                                      # list -> record flash LSE per call


def rope(x, pos0):
    """Rotary position embedding on (B, H, T, D) starting at position pos0."""
    B, H, T, D = x.shape
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D))
    ang = torch.arange(pos0, pos0 + T, device=x.device, dtype=torch.float32)[:, None] * inv
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)


def flash_attention(q, k, v, causal, layer):
    """q, k, v: FP16 (B, H, T, D). Forced flash backend (errors if ineligible)."""
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    if ORACLE is not None:
        with torch.no_grad():
            lse = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, causal)[1]
        ORACLE.append((layer, q.shape[2], k.shape[2], lse.float().cpu()))
    return out


class Block(nn.Module):
    def __init__(self, d, h, idx):
        super().__init__()
        self.h, self.idx = h, idx
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)

    def forward(self, x, cache=None, pos0=0):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q, pos0), rope(k, pos0)
        if cache is not None:                      # KV cache for autoregressive decoding
            if cache.get(self.idx) is not None:
                pk, pv = cache[self.idx]
                k, v = torch.cat((pk, k), 2), torch.cat((pv, v), 2)
            cache[self.idx] = (k, v)
        # q_len == k_len: causal (training / prefill); q_len == 1: all keys visible
        causal = q.shape[2] == k.shape[2] and q.shape[2] > 1
        y = flash_attention(q.half(), k.half(), v.half(), causal, self.idx).float()
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class TinyGPT(nn.Module):
    def __init__(self, d=128, h=4, n_layers=4):
        super().__init__()
        self.cfg = dict(d=d, h=h, n_layers=n_layers)
        self.emb = nn.Embedding(VOCAB, d)
        self.blocks = nn.ModuleList(Block(d, h, i) for i in range(n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)
        self.ext_embed = None
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, cache=None, pos0=0):
        # one-hot @ weight rather than self.emb(idx): bit-identical in FP32 (no TF32),
        # but avoids two kernels that hang NVBit (nixnan) at load time -- the
        # embedding/index-select forward and the CUB radix sort in its backward
        x = (idx[..., None] == torch.arange(VOCAB, device=idx.device)).float() @ self.emb.weight
        if self.ext_embed is not None:             # external input (experiment_lse.py)
            x = x + self.ext_embed[pos0:pos0 + idx.shape[1]].to(x.device)
        for b in self.blocks:
            x = b(x, cache, pos0)
        return self.head(self.ln_f(x))


def encode(word):
    return [BOS] + [int(c) for c in word]


@torch.no_grad()
def complete(model, prompt, length, device="cuda"):
    """Greedy autoregressive completion of the digit string `prompt` to
    `length` digits (KV cache: one prefill call, then one call per token).
    BOS/EOS are masked so the completion is digits only."""
    ids = torch.tensor([encode(prompt)], device=device)
    cache = {}
    logits = model(ids, cache, 0)
    out = list(prompt)
    pos = ids.shape[1]
    while len(out) < length:
        last = logits[0, -1].clone()
        last[BOS] = last[EOS] = -math.inf
        nxt = int(last.argmax())
        out.append(str(nxt))
        if len(out) < length:
            logits = model(torch.tensor([[nxt]], device=device), cache, pos)
            pos += 1
    return "".join(out)


def load(path="tinygpt.pt", device="cuda"):
    ck = torch.load(path, map_location=device)
    m = TinyGPT(**ck["cfg"]).to(device)
    m.load_state_dict(ck["state"])
    return m.eval()
