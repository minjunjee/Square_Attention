#!/usr/bin/env python3
"""Does a trained tied layer follow the chain, or answer the end digit of the longest chain?

Loads saved models of the interference test (n = 16 lines), asks 2,000 training-distribution programs
(4 questions each) at r = 6, and splits the questions by whether the right answer is the end digit of
the program's longest chain. A model that follows chains answers the other questions correctly; a model
that answers the longest chain's digit gets them wrong.

usage: python3 tools/check_trick.py > results/check_trick_e8.txt      (CPU only)
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import poc_fair as P  # noqa: E402

MODELS = [("names 512, d=64, seed 0", "ckpt/e8_tied_d64_n16_s0.pt", 512, 64),
          ("names 512, d=64, seed 1", "ckpt/e8_tied_d64_n16_s1.pt", 512, 64),
          ("names 512, d=64, seed 2", "ckpt/e8_tied_d64_n16_s2.pt", 512, 64),
          ("names 48, d=64, seed 0", "ckpt/e8b_tied_d64_names48_s0.pt", 48, 64)]


def load(path, n_names, d, n=16):
    P.NAMES = np.concatenate([np.arange(P.LET), np.arange(40, 40 + n_names - P.LET)]) if n_names > P.LET else np.arange(P.LET)
    P.V = 40 + max(0, n_names - P.LET)
    m = P.Tied(n, 4, 64, d, read="square", gate_ctx=True, hard_taps=0, tap_attn=8, line=4)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    m.tap_scale = 10.0                                               # value reached at the end of training (--tap_sharp 10)
    return m


def longest_chain_digit(tok, n=16):
    lhs, rhs = tok[:, 0:4 * n:4], tok[:, 2:4 * n:4]
    out = np.zeros(len(tok), np.int64)
    for b in range(len(tok)):
        val, dep = {}, {}
        for x, y in zip(lhs[b], rhs[b]):
            if P.LET <= y < P.LET + P.DIG:
                val[x], dep[x] = y, 1
            else:
                val[x], dep[x] = val[y], dep[y] + 1
        out[b] = val[max(dep, key=dep.get)]
    return out


def main():
    torch.set_num_threads(8)
    for name, path, n_names, d in MODELS:
        m = load(path, n_names, d)
        tok, lab, qd, _ = P.make_batch(np.random.default_rng(123), 2000, 16, 4, 4)
        with torch.no_grad():
            pred = m(torch.from_numpy(tok), r=6).argmax(-1).numpy()
        big = longest_chain_digit(tok)[:, None]
        acc, other = pred == lab, lab != big
        print(f"{name}: accuracy by depth " + ", ".join(f"{k}: {100 * acc[qd == k].mean():.0f}%" for k in (1, 2, 3, 4)))
        print(f"  questions whose answer is not the longest chain's end digit: {100 * other.mean():.0f}% of questions; "
              f"the model answers that digit anyway {100 * (pred == big)[other].mean():.0f}%, correct {100 * acc[other].mean():.0f}%")
        print(f"  questions whose answer is that digit: correct {100 * acc[~other].mean():.0f}%")


if __name__ == "__main__":
    main()
