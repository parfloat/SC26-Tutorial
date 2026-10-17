#!/usr/bin/env python3
"""Train TinyGPT on the fixed training set and save tinygpt.pt.

Each word w is the sequence A w F (start symbol, digits, end symbol), e.g.
5881 -> A5881F. All 100 sequences form one batch, right-padded with F
(pads are ignored by the loss; with causal attention a pad
can never influence a real position, so no attention mask is needed and
the flash backend stays eligible).

    python3 train.py [--steps 1500] [--out tinygpt.pt]
"""
import argparse
import time

import torch
import torch.nn.functional as F

from tinygpt import TinyGPT, EOS, decode, encode

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=1500)
p.add_argument("--lr", type=float, default=3e-3)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="tinygpt.pt")
a = p.parse_args()

torch.manual_seed(a.seed)
words = open("training-set.txt").read().split()
seqs = [encode(w) + [EOS] for w in words]
T = max(map(len, seqs))
x = torch.full((len(seqs), T), EOS, dtype=torch.long)
y = torch.full((len(seqs), T), -100, dtype=torch.long)       # -100 = ignored by the loss
for i, s in enumerate(seqs):
    x[i, :len(s)] = torch.tensor(s)
    y[i, :len(s) - 1] = torch.tensor(s[1:])
x, y = x[:, :-1].cuda(), y[:, :-1].cuda()
print(f"{len(seqs)} sequences, e.g. {decode(seqs[0])}, {decode(seqs[1])}; batch {tuple(x.shape)}")

model = TinyGPT().cuda()
print(f"TinyGPT: {sum(p.numel() for p in model.parameters()):,} params, {model.cfg}")
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
t0 = time.time()
for step in range(1, a.steps + 1):
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    sched.step()
    if step % 100 == 0 or step == 1:
        print(f"step {step:5d}  loss {loss.item():.4f}  ({time.time() - t0:.1f}s)", flush=True)

model.eval()
with torch.no_grad():
    pred = model(x).argmax(-1)
mask = y != -100
acc = (pred[mask] == y[mask]).float().mean().item()
exact = sum(bool((pred[i][mask[i]] == y[i][mask[i]]).all()) for i in range(len(seqs)))
print(f"train next-token accuracy {acc:.4f}; words reproduced exactly {exact}/{len(seqs)}")
torch.save({"cfg": model.cfg, "state": model.state_dict(), "final_loss": loss.item(),
            "train_acc": acc}, a.out)
print(f"saved {a.out}")
