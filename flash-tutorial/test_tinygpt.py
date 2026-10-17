#!/usr/bin/env python3
"""Tests for TinyGPT's token alphabet (0..9, start symbol A, end symbol F) and
the trained checkpoint.

    python3 test_tinygpt.py          # tokenizer tests on CPU; model tests if CUDA is up
"""
import os
import unittest

import torch

from tinygpt import BOS, EOS, STOI, TOKENS, VOCAB, decode, encode

HERE = os.path.dirname(os.path.abspath(__file__))


def training_words():
    with open(os.path.join(HERE, "training-set.txt")) as f:
        return f.read().split()


class Tokens(unittest.TestCase):
    def test_alphabet(self):
        self.assertEqual(TOKENS, "0123456789AF")
        self.assertEqual(VOCAB, 12)
        self.assertEqual((BOS, EOS), (10, 11))          # ids unchanged: old checkpoints load as is
        self.assertEqual((STOI["A"], STOI["F"]), (10, 11))
        for d in range(10):
            self.assertEqual(STOI[str(d)], d)

    def test_encode_decode(self):
        self.assertEqual(encode("312"), [10, 3, 1, 2])
        self.assertEqual(encode("A312"), [10, 3, 1, 2])
        self.assertEqual(decode([10, 3, 1, 2, 11]), "A312F")
        self.assertEqual(decode(encode("5881") + [EOS]), "A5881F")
        with self.assertRaises(KeyError):
            encode("31B")

    def test_training_sequences(self):
        words = training_words()
        self.assertEqual(len(words), 100)
        for w in words:
            s = decode(encode(w) + [EOS])
            self.assertEqual(s, "A" + w + "F")
            self.assertEqual(s.count("A") + s.count("F"), 2)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA (flash attention)")
class Model(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tinygpt import load
        cls.model = load(os.path.join(HERE, "tinygpt.pt"))

    def test_shapes(self):
        self.assertEqual(self.model.cfg, dict(d=128, h=4, n_layers=4))
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 791_808)
        self.assertEqual(tuple(self.model.emb.weight.shape), (VOCAB, 128))
        self.assertEqual(tuple(self.model.head.weight.shape), (VOCAB, 128))

    def test_completion(self):
        from tinygpt import complete
        self.assertEqual(complete(self.model, "312", 6), "312323")

    def test_completion_is_digits(self):
        from tinygpt import complete
        out = complete(self.model, "9", 20)
        self.assertEqual(len(out), 20)
        self.assertTrue(out.isdigit(), out)               # A and F are masked

    def test_kv_cache_matches_full_forward(self):
        ids = torch.tensor([encode("31245")], device="cuda")
        with torch.no_grad():
            full = self.model(ids)[0, -1]
            cache = {}
            self.model(ids[:, :-1], cache, 0)
            step = self.model(ids[:, -1:], cache, ids.shape[1] - 1)[0, -1]
        torch.testing.assert_close(step, full, rtol=1e-3, atol=1e-3)

    def test_memorised_words(self):
        """Once a word's prefix is unique in the training set, the model predicts
        the rest of the word and the final F (the earlier digits are ambiguous)."""
        words = training_words()
        for w in words:
            k = next((k for k in range(1, len(w) + 1)
                      if sum(v.startswith(w[:k]) for v in words) == 1), None)
            if k is None:                                 # w is a prefix of another word
                continue
            ids = torch.tensor([encode(w) + [EOS]], device="cuda")
            with torch.no_grad():
                pred = self.model(ids[:, :-1]).argmax(-1)[0]
            self.assertEqual(decode(pred[k:].tolist()), w[k:] + "F", w)

if __name__ == "__main__":
    unittest.main(verbosity=2)
