#!/usr/bin/env python3
"""Tile maps for report.tex / slides.tex from PyTorch_Instrumentation/traces/completion/dumps.npz.

Each map is one tile of call 11 (layer 3, last decode step: seqlen_q 1, seqlen_k 6), head 0,
one pixel block per element, coloured by value class. Writes figs/*.png (no matplotlib needed).

    python make_figures.py
"""
import os
import struct
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
Z = np.load(os.path.join(HERE, "..", "PyTorch_Instrumentation", "traces", "completion", "dumps.npz"))
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)

COL = {  # RGB
    "valid": (31, 119, 180),     # finite value inside the sequence
    "finite": (150, 150, 150),   # finite non-zero value in a padding lane
    "zero": (235, 235, 235),     # exactly 0
    "nan": (214, 39, 40),        # NaN
    "posinf": (255, 127, 14),    # +inf
    "neginf": (60, 60, 60),      # -inf (the mask)
    "sub": (148, 103, 189),      # subnormal
}


def png(path, rgb, scale):
    """Write an (H, W, 3) uint8 array as PNG, each element a scale x scale block."""
    img = np.repeat(np.repeat(rgb, scale, 0), scale, 1)
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[r].tobytes() for r in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def classify(x, valid, tiny):
    out = np.zeros(x.shape + (3,), np.uint8)
    cls = np.full(x.shape, "finite", object)
    cls[x == 0] = "zero"
    cls[(x != 0) & (np.abs(x) < tiny)] = "sub"
    cls[valid & np.isfinite(x)] = "valid"
    cls[np.isnan(x)] = "nan"
    cls[np.isposinf(x)] = "posinf"
    cls[np.isneginf(x)] = "neginf"
    for k, c in COL.items():
        out[cls == k] = c
    return out


n, h = 11, 0
layer, sq, sk, _ = Z[f"call{n}_meta"]
for name, tiny, rows_valid, cols_valid in (("Q", 2 ** -14, sq, None), ("K", 2 ** -14, sk, None),
                                           ("V", 2 ** -14, sk, None), ("O", 2 ** -126, sq, None),
                                           ("S_RAW", 2 ** -126, sq, sk), ("S_MASK", 2 ** -126, sq, sk),
                                           ("P", 2 ** -14, sq, sk)):
    x = Z[f"call{n}_{name}"][h]
    valid = np.zeros(x.shape, bool)
    if cols_valid is None:
        valid[:rows_valid] = True
    else:
        valid[:rows_valid, :cols_valid] = True
    png(os.path.join(HERE, "figs", f"tile_{name}.png"), classify(x, valid, tiny), 4 if x.shape[1] == 32 else 2)
    nan = np.isnan(x).sum()
    print(f"{name:7s} {x.shape}  NaN {nan}")
print("wrote figs/")
