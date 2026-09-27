# Karpathy-NGram

A standalone, dependency-free benchmark implementing the character-level
n-gram language model from the first lecture of Andrej Karpathy's
["makemore"](https://github.com/karpathy/makemore) series, generalized
from a bigram (n=2) to arbitrary n-gram order.

The model is the count-based one, not the neural one: no PyTorch, no
autograd, no GPU. It learns by counting `(context, next-character)` pairs
over a name corpus, normalizes counts into probabilities (with Laplace
smoothing), and can then be scored (negative log-likelihood) or sampled
from to generate new, name-like strings one character at a time.

## Why it's here

This directory holds a plain-CPU reference benchmark alongside the
CUDA/`nixnan` case studies elsewhere in this tutorial (`tc_matmult/`,
`lu_solve_gpu/`, `mpir_gpu/`, `rd_nixnan/`, `hec_atan2/`, `hec_fluidSim/`,
`pytorch_slogdet/`) -- small, fast, and legible enough to run anywhere
with just `python3`.

## Data

`names.txt` is Karpathy's own training corpus, copied verbatim from
[`karpathy/makemore`](https://github.com/karpathy/makemore/blob/master/names.txt)
(MIT licensed) -- 32,032 English first names, one per line, lowercase.

## Usage

```bash
./ngram.py                          # default: train n=2,3,4 on names.txt
./ngram.py --orders 2                # bigram only (Karpathy's original)
./ngram.py --orders 5 --smoothing 0.1 --num-samples 20
./ngram.py --data /path/to/words.txt --seed 42
```

Key flags:

| Flag | Meaning |
|---|---|
| `--orders` | comma-separated n-gram orders to train, e.g. `2,3,4` (n=2 is a bigram model: 1 character of context predicts the next) |
| `--smoothing` | Laplace (add-k) smoothing constant applied to every count |
| `--num-samples` | names to sample per order |
| `--top-states` | states shown in each order's state-machine dump (`0` = show all) |
| `--seed` | RNG seed for the train/dev/test split and for sampling |

For each order, the script reports: training time, number of distinct
states and observed transitions, average per-character negative
log-likelihood on train/dev/test splits, a batch of sampled names, and a
text dump of the learned state machine (see below).

## The model as a state machine, in text

An n-gram model *is* a finite-state machine: a state is the trailing
`n-1` characters of context, and each state has one outgoing,
probability-weighted edge per possible next character (27 of them here:
`a`-`z` plus a boundary symbol `.` marking word start/end). Rather than
plotting that automaton (this benchmark deliberately avoids both
`graphviz` and any plotting library), each state's fan-out is collapsed
to the two edges that matter and printed as adjacent text arrows:

```
S --p--> S1
S --(1-p)--> other
```

`S1` is the single most probable next character out of state `S`;
`other` aggregates every remaining character into one complementary edge,
so the two probabilities always sum to 1 and the two moves out of any
given state are always printed next to each other. For example, from the
bigram model:

```
  [ 27099x] a --0.196--> <END>
             a --0.804--> <other 26 chars>
  [ 25626x] . --0.137--> a
             . --0.863--> <other 26 chars>
```

States are ranked by how often they were visited during training (most
common context first) and truncated to `--top-states` (default 15) so
that higher-order models -- whose state count grows as up to `27^(n-1)`
-- stay readable; pass `--top-states 0` to dump every visited state.

## Sample output (n=2,3, `--top-states 4`)

```
n-gram order n=2  (1-character context -> 1 predicted character)
trained in 110.90 ms  (27 distinct states, 613 distinct (state,next) edges observed)
  train NLL/char = 2.4555   (182546 examples, 1091.67 ms)
    dev NLL/char = 2.4552   (22840 examples, 135.40 ms)
   test NLL/char = 2.4514   (22760 examples, 134.76 ms)

sampled names:
  erauirabenerey
  o
  ki
  r
  mi

state machine (top 4 states by visit count):
  [ 27099x] a --0.196--> <END>
             a --0.804--> <other 26 chars>
  [ 25626x] . --0.137--> a
             . --0.863--> <other 26 chars>
  [ 16338x] e --0.193--> <END>
             e --0.807--> <other 26 chars>
  [ 14728x] n --0.366--> <END>
             n --0.634--> <other 26 chars>

n-gram order n=3  (2-character context -> 1 predicted character)
trained in 105.50 ms  (588 distinct states, 5801 distinct (state,next) edges observed)
  train NLL/char = 2.2154   (182546 examples, 1093.99 ms)
    dev NLL/char = 2.2465   (22840 examples, 136.99 ms)
   test NLL/char = 2.2312   (22760 examples, 138.87 ms)

sampled names:
  jayton
  izah
  bryedengugbayan
  jire
  ani

summary
  n=2: train NLL/char=2.4555  dev NLL/char=2.4552  test NLL/char=2.4514
  n=3: train NLL/char=2.2154  dev NLL/char=2.2465  test NLL/char=2.2312
```

NLL/char drops as n grows (more context -> better fit), while the dev/test
gap widens -- the expected bias/variance tradeoff of raw count-based
n-gram models as the state space grows and per-state training data
thins out.

## Attribution

Model design and training corpus: Andrej Karpathy,
[`karpathy/makemore`](https://github.com/karpathy/makemore) (MIT
License). This implementation (`ngram.py`) is a from-scratch, pure
standard-library Python rewrite generalized to arbitrary n-gram order,
with the graphviz/plotting-based visualizations replaced by the text
state-machine renderer described above.
