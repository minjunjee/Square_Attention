#!/usr/bin/env python3
"""Do the remaining errors of the repaired models sit on rarely seen names?

Models trained with --name_zipf 1 see name i with probability proportional to 1 / i. For each saved model
of the given runs this script asks two-chain questions (uniform names, r = 6) and splits them by the rarest
name on the queried chain (largest rank among the names the chain passes through). If rarely seen names
caused the errors, accuracy would fall with that rank; if every bin has the same accuracy, they do not.

usage: CUDA_VISIBLE_DEVICES='' .venv/bin/python tools/err_by_rank.py > results/e9b_err_by_rank.txt   (CPU)
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import poc_fair as P  # noqa: E402

V0, N_NAMES, n = P.V, 512, 16
BINS = [(0, 63), (64, 127), (128, 255), (256, 511)]


def chain_rank(tok):
    """Largest name rank on the chain that the first query follows."""
    rank = {int(t): i for i, t in enumerate(P.NAMES)}
    out = np.zeros(len(tok), np.int64)
    for b in range(len(tok)):
        val = {int(tok[b, 4 * i]): int(tok[b, 4 * i + 2]) for i in range(n)}
        x, worst = int(tok[b, 4 * n + 1]), 0
        while x in val:
            worst = max(worst, rank[x])
            x = val[x]
        out[b] = worst
    return out


def main():
    torch.set_num_threads(8)
    P.NAMES = np.concatenate([np.arange(P.LET), np.arange(V0, V0 + N_NAMES - P.LET)])
    P.V = V0 + N_NAMES - P.LET
    for arm in ("orth_zipf_r6", "orth_zipf", "orth_curr_r6"):
        tot = {D: np.zeros((len(BINS), 2)) for D in (2, 4, 8)}
        for seed in range(10):
            m = P.Tied(n, 4, 64, 64, read="square", gate_ctx=True, tap_attn=8, line=4)
            m.load_state_dict(torch.load(os.path.join(ROOT, f"ckpt/e9b_{arm}_s{seed}.pt"), map_location="cpu"))
            m.eval()
            m.tap_scale = 10.0
            for D in tot:
                tok, lab, _, _ = P.make_batch(np.random.default_rng(777 + seed), 2000, n, 4, 4, twin_depth=D)
                with torch.no_grad():
                    pred = m(torch.from_numpy(tok), r=6)[:, 0].argmax(-1).numpy()
                rk, good = chain_rank(tok), pred == lab[:, 0]
                for i, (lo, hi) in enumerate(BINS):
                    sel = (rk >= lo) & (rk <= hi)
                    tot[D][i] += [good[sel].sum(), sel.sum()]
        print(f"== {arm}, 10 seeds pooled; rarest name on the queried chain has rank ...")
        for D in tot:
            print(f"  depth {D}: " + " | ".join(f"{lo + 1}-{hi + 1}: {100 * c / max(1, t):.1f}% of {int(t)}" for (lo, hi), (c, t) in zip(BINS, tot[D])))


if __name__ == "__main__":
    main()
