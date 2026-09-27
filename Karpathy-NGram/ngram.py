#!/usr/bin/env python3
"""
Karpathy-style character-level n-gram language model.

A count-based (no neural net, no autograd) generalization of the bigram
model from Andrej Karpathy's "makemore" series
(https://github.com/karpathy/makemore) to arbitrary n-gram order. Trained
by simple counting + Laplace smoothing on a corpus of names, evaluated by
average negative log-likelihood, and able to sample new names from the
learned distribution.

Every n-gram model IS a (stochastic) finite-state machine: a state is the
last (n-1) characters of context, and each state has one outgoing edge per
possible next character, weighted by its learned probability. Rather than
plotting that automaton (graphviz / matplotlib are both out of scope for
this benchmark), it is rendered as text arrows:

    S --prob--> S1

For readability, each state's fan-out (up to 27 edges for this alphabet)
is collapsed to the two edges that matter: the single most likely next
character, and an aggregate "everything else" edge. Because the two
probabilities are complementary (p and 1-p), they are always printed as
an adjacent pair:

    S --p--> S1
    S --(1-p)--> S2
"""

import argparse
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

BOUNDARY = "."  # start/end-of-word token, following Karpathy's makemore convention

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_words(path):
    words = [w.strip().lower() for w in Path(path).read_text().splitlines()]
    return [w for w in words if w]


def build_vocab(words):
    chars = sorted(set("".join(words)))
    if BOUNDARY in chars:
        raise ValueError(f"corpus already contains reserved boundary char {BOUNDARY!r}")
    itos = [BOUNDARY] + chars
    stoi = {c: i for i, c in enumerate(itos)}
    return stoi, itos


def split_words(words, seed, train_frac=0.8, dev_frac=0.1):
    rng = random.Random(seed)
    shuffled = words[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_frac)
    n_dev = int(n * dev_frac)
    train = shuffled[:n_train]
    dev = shuffled[n_train : n_train + n_dev]
    test = shuffled[n_train + n_dev :]
    return train, dev, test


# ---------------------------------------------------------------------------
# n-gram counting
#
# n = total characters spanned by one training example: (n-1) characters of
# context followed by the 1 character being predicted. n=2 is a bigram
# model (Karpathy's original), n=3 a trigram model, etc. Each word is
# padded with BOUNDARY tokens so that word-start and word-end are just
# ordinary, learnable states of the chain.
# ---------------------------------------------------------------------------


def iter_ngrams(word, n):
    chs = [BOUNDARY] * (n - 1) + list(word) + [BOUNDARY]
    for i in range(len(chs) - (n - 1)):
        context = tuple(chs[i : i + n - 1])
        target = chs[i + n - 1]
        yield context, target


def train_counts(words, n):
    counts = defaultdict(Counter)
    for w in words:
        for ctx, tgt in iter_ngrams(w, n):
            counts[ctx][tgt] += 1
    return counts


def context_distribution(counts, ctx, vocab_chars, smoothing):
    row = counts.get(ctx, Counter())
    total = sum(row.values()) + smoothing * len(vocab_chars)
    return {c: (row.get(c, 0) + smoothing) / total for c in vocab_chars}


# ---------------------------------------------------------------------------
# Evaluation: average negative log-likelihood, the standard n-gram loss
# ---------------------------------------------------------------------------


def evaluate_nll(words, n, counts, vocab_chars, smoothing):
    nll_sum = 0.0
    n_examples = 0
    for w in words:
        for ctx, tgt in iter_ngrams(w, n):
            p = context_distribution(counts, ctx, vocab_chars, smoothing)[tgt]
            nll_sum += -math.log(p)
            n_examples += 1
    return nll_sum / n_examples, n_examples


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_word(counts, n, vocab_chars, smoothing, rng, max_len=40):
    ctx = tuple([BOUNDARY] * (n - 1))
    out = []
    for _ in range(max_len):
        dist = context_distribution(counts, ctx, vocab_chars, smoothing)
        chars, weights = zip(*dist.items())
        nxt = rng.choices(chars, weights=weights, k=1)[0]
        if nxt == BOUNDARY:
            break
        out.append(nxt)
        ctx = (ctx + (nxt,))[1:] if n > 1 else ctx
    return "".join(out)


# ---------------------------------------------------------------------------
# Text rendering of the n-gram state machine (no graphviz, no plotting)
# ---------------------------------------------------------------------------


def state_label(ctx):
    return "".join(ctx) if ctx else "ε"  # epsilon for the empty context (n=1)


def render_state_machine(counts, vocab_chars, smoothing, top_states):
    """
    Print each visited state's two outgoing moves as an adjacent pair:

        S --p--> S1          (most probable next character)
        S --(1-p)--> other   (everything else, aggregated)

    States are ranked by total visit count (most-trained-on first) and
    truncated to `top_states` (0 = no truncation) to keep the output
    readable for higher-order models, whose state count grows as
    27^(n-1).
    """
    ranked = sorted(counts.items(), key=lambda kv: -sum(kv[1].values()))
    if top_states > 0:
        ranked = ranked[:top_states]

    lines = []
    for ctx, row in ranked:
        dist = context_distribution(counts, ctx, vocab_chars, smoothing)
        best_char, best_p = max(dist.items(), key=lambda kv: kv[1])
        rest_p = 1.0 - best_p
        s = state_label(ctx)
        s1 = best_char if best_char != BOUNDARY else "<END>"
        n_seen = sum(row.values())
        lines.append(f"  [{n_seen:>6d}x] {s} --{best_p:.3f}--> {s1}")
        lines.append(f"             {s} --{rest_p:.3f}--> <other {len(vocab_chars) - 1} chars>")
    return lines


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_order(words_train, words_dev, words_test, n, vocab_chars, smoothing, args, rng):
    print(f"\n{'=' * 70}")
    print(f"n-gram order n={n}  ({n - 1}-character context -> 1 predicted character)")
    print(f"{'=' * 70}")

    t0 = time.perf_counter()
    counts = train_counts(words_train, n)
    train_time = time.perf_counter() - t0
    n_states = len(counts)
    n_edges = sum(len(row) for row in counts.values())
    print(f"trained in {train_time * 1e3:.2f} ms  "
          f"({n_states} distinct states, {n_edges} distinct (state,next) edges observed)")

    nll_by_split = {}
    for split_name, split_words_ in (("train", words_train), ("dev", words_dev), ("test", words_test)):
        t0 = time.perf_counter()
        nll, n_ex = evaluate_nll(split_words_, n, counts, vocab_chars, smoothing)
        eval_time = time.perf_counter() - t0
        nll_by_split[split_name] = nll
        print(f"  {split_name:>5} NLL/char = {nll:.4f}   "
              f"({n_ex} examples, {eval_time * 1e3:.2f} ms)")

    print(f"\nsampled names:")
    samples = [sample_word(counts, n, vocab_chars, smoothing, rng) for _ in range(args.num_samples)]
    for w in samples:
        print(f"  {w}")

    print(f"\nstate machine "
          f"(top {args.top_states if args.top_states > 0 else 'all'} states by visit count; "
          f"'S --p--> S1 / S --(1-p)--> other' pairs):")
    for line in render_state_machine(counts, vocab_chars, smoothing, args.top_states):
        print(line)

    return {"n": n, "train_time_ms": train_time * 1e3, **{f"{k}_nll": v for k, v in nll_by_split.items()}}


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(here / "names.txt"), help="training corpus, one word per line")
    ap.add_argument("--orders", default="2,3,4",
                     help="comma-separated n-gram orders to train (n=2 is bigram, n=3 trigram, ...)")
    ap.add_argument("--smoothing", type=float, default=1.0, help="Laplace (add-k) smoothing constant")
    ap.add_argument("--num-samples", type=int, default=10, help="names to sample per order")
    ap.add_argument("--top-states", type=int, default=15,
                     help="states shown per order's state-machine dump (0 = show all)")
    ap.add_argument("--seed", type=int, default=2147483647, help="RNG seed (train/dev/test split + sampling)")
    args = ap.parse_args()

    words = load_words(args.data)
    stoi, itos = build_vocab(words)
    vocab_chars = itos  # includes BOUNDARY
    words_train, words_dev, words_test = split_words(words, args.seed)

    print(f"corpus: {args.data}")
    print(f"{len(words)} words total -> {len(words_train)} train / {len(words_dev)} dev / {len(words_test)} test")
    print(f"vocabulary ({len(vocab_chars)} symbols incl. boundary {BOUNDARY!r}): {''.join(itos)}")

    rng = random.Random(args.seed)
    orders = [int(x) for x in args.orders.split(",")]
    summary = []
    for n in orders:
        summary.append(run_order(words_train, words_dev, words_test, n, vocab_chars, args.smoothing, args, rng))

    print(f"\n{'=' * 70}")
    print("summary (lower dev/test NLL = better fit; higher n overfits without more smoothing/data)")
    print(f"{'=' * 70}")
    for s in summary:
        print(f"  n={s['n']}: train NLL/char={s['train_nll']:.4f}  dev NLL/char={s['dev_nll']:.4f}  "
              f"test NLL/char={s['test_nll']:.4f}  train_time={s['train_time_ms']:.2f} ms")


if __name__ == "__main__":
    main()
