#!/usr/bin/env python3
"""The intended solution of the squared read, computed without training, to separate what a d x d state
can represent from what training finds.

Every name and digit gets a fixed code. Digits take the first 10 coordinates, so they are exact fixed
points of the state. Names get orthonormal codes in the other d - 10 coordinates when there is room, and
random unit codes there otherwise. Each line "x=y" writes key code(x) and value code(y) with the delta rule
at full strength (beta = 1) into a state that starts as the identity; the query reads code(x) through
S^(2^r) with the same Frobenius rescaling as the model, and the answer is the digit coordinate with the
largest value. Programs come from the same generator as training (poc_fair.make_batch), two-chain test.

usage: python3 interference_oracle.py [--out results/oracle_interference.jsonl]   (CPU only)
"""
import argparse
import json
import math
import time

import numpy as np
import torch

import poc_fair as P


def set_names(n_names):
    P.NAMES = np.arange(P.LET) if n_names == P.LET else np.concatenate([np.arange(P.LET), np.arange(P.V, P.V + n_names - P.LET)])


def make_codes(d, n_names, rng):
    C = np.zeros((P.V + max(0, n_names - P.LET), d))
    C[P.LET + np.arange(P.DIG), np.arange(P.DIG)] = 1.0               # digits: the first 10 coordinates
    k = d - P.DIG
    if n_names <= k:                                                  # room for exactly orthogonal name codes
        X = np.linalg.qr(rng.standard_normal((k, n_names)))[0].T
    else:
        X = rng.standard_normal((n_names, k))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    C[P.NAMES, P.DIG:] = X
    return torch.tensor(C, dtype=torch.float32)


def two_chain_acc(C, d, n, D, r, B, rng):
    tok, lab, _, _ = P.make_batch(rng, B, n, 4, 4, twin_depth=D)
    tok = torch.from_numpy(tok)
    K, Vv, q = C[tok[:, 0:4 * n:4]], C[tok[:, 2:4 * n:4]], C[tok[:, 4 * n + 1]]
    S = torch.eye(d).expand(B, d, d).clone()
    for i in range(n):                                                # delta rule, beta = 1
        k, v = K[:, i], Vv[:, i]
        S += torch.einsum("bi,bj->bij", v - torch.einsum("bij,bj->bi", S, k), k)
    M = S
    for _ in range(r):
        M = M @ M
        M = M / (M.flatten(1).norm(dim=1)[:, None, None] / math.sqrt(d) + 1e-6)
    y = torch.einsum("bij,bj->bi", M, q)
    pred = y[:, :P.DIG].argmax(dim=1).numpy() + P.LET
    return float((pred == lab[:, 0]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/oracle_interference.jsonl")
    ap.add_argument("--B", type=int, default=500)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    grid = [(d, k, n) for d in (32, 64, 128, 256) for k in (26, 48, 128, 512) for n in (16, 32, 64, 128, 256) if n <= k]
    with open(args.out, "w") as f:
        for d, k, n in grid:
            t0 = time.time()
            set_names(k)
            rng = np.random.default_rng(0)
            C = make_codes(d, k, rng)
            acc = {D: round(two_chain_acc(C, d, n, D, 6, args.B, rng), 4) for D in (1, 2, 4, 8)}
            by_r = {}
            if n == 16:                                               # does squaring more than needed amplify interference?
                by_r = {r: round(two_chain_acc(C, d, n, 4, r, args.B, rng), 4) for r in (2, 3, 4, 6, 8)}
            row = dict(d=d, n_names=k, n=n, r=6, B=args.B, acc_two_chain=acc, acc_depth4_by_r=by_r, seconds=round(time.time() - t0, 1))
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(row, flush=True)


if __name__ == "__main__":
    main()
