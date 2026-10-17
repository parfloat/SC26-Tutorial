#!/usr/bin/env python3
"""Write training-set.txt -- 100 random words over the digits 0..9,
each of length 1..20 (one word per line), from a fixed seed."""
import random

SEED, N_WORDS, MIN_LEN, MAX_LEN = 2026, 100, 1, 20

rng = random.Random(SEED)
words = ["".join(rng.choice("0123456789") for _ in range(rng.randint(MIN_LEN, MAX_LEN)))
         for _ in range(N_WORDS)]
with open("training-set.txt", "w") as f:
    f.write("\n".join(words) + "\n")
print(f"wrote {len(words)} words, lengths {min(map(len, words))}..{max(map(len, words))}")
