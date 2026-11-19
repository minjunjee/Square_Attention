#!/usr/bin/env python3
"""The intended solution computed without training, under four ways of placing the codes. (CPU only)

Each line "x=y" writes key code(x) and value code(y) at full strength, the query reads code(x) through
S^(2^r), and the answer is the digit whose code is nearest. Same programs as interference_oracle.py (n = 16).
  ideal   digits own 10 coordinates, names are random unit vectors in the other d - 10 (interference_oracle.py)
  random  every token gets a random unit vector in d dimensions
  orth    N(0, 1) embeddings of width 64 through an orthogonal code map (--code_init orth)
  init    N(0, 1) embeddings of width 64 through the default nn.Linear code map (how the models of
          experiments 1 to 8 start)

usage: .venv/bin/python tools/oracle_code_init.py > results/oracle_code_init.txt
"""
import os, sys, math
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import poc_fair as P
torch.set_num_threads(8)
def set_names(k):
    P.NAMES = np.arange(P.LET) if k == P.LET else np.concatenate([np.arange(P.LET), np.arange(40, 40 + k - P.LET)])
def codes(kind, d, k, seed):
    g = torch.Generator().manual_seed(seed)
    Vn = 40 + max(0, k - P.LET)
    if kind == "ideal":                      # digits own 10 coordinates, names random in the rest
        C = torch.zeros(Vn, d); C[P.LET + torch.arange(10), torch.arange(10)] = 1.0
        X = torch.randn(k, d - 10, generator=g); C[torch.from_numpy(P.NAMES), 10:] = X / X.norm(dim=1, keepdim=True)
        return C
    E = torch.randn(Vn, 64, generator=g)     # nn.Embedding init
    if kind == "random":                     # isotropic random unit codes for every token
        X = torch.randn(Vn, d, generator=g); return X / X.norm(dim=1, keepdim=True)
    if kind == "init":                       # the model's initial codes: default nn.Linear(64, d) on N(0,1) embeddings
        W = (torch.rand(d, 64, generator=g) * 2 - 1) / math.sqrt(64)
    elif kind == "orth":                     # orthogonal code map on the same embeddings
        W = torch.linalg.qr(torch.randn(max(d, 64), max(d, 64), generator=g))[0][:d, :64]
    X = E @ W.T
    return X / X.norm(dim=1, keepdim=True)
def acc(C, d, n, D, r, B, rng, beta=1.0):
    tok, lab, _, _ = P.make_batch(rng, B, n, 4, 4, twin_depth=D)
    tok = torch.from_numpy(tok)
    K, Vv, q = C[tok[:, 0:4 * n:4]], C[tok[:, 2:4 * n:4]], C[tok[:, 4 * n + 1]]
    S = torch.eye(d).expand(B, d, d).clone()
    for i in range(n):
        k, v = K[:, i], Vv[:, i]
        S += beta * torch.einsum("bi,bj->bij", v - torch.einsum("bij,bj->bi", S, k), k)
    M = S
    for _ in range(r):
        M = M @ M; M = M / (M.flatten(1).norm(dim=1)[:, None, None] / math.sqrt(d) + 1e-6)
    y = torch.einsum("bij,bj->bi", M, q)
    pred = (y @ C[P.LET:P.LET + 10].T).argmax(1).numpy() + P.LET       # nearest digit code
    return float((pred == lab[:, 0]).mean())
print("two-chain accuracy of the intended solution, n=16 lines; cells: depth 1 / 4 / 8 (r = 2 / 3 / 6 where noted)")
for d in (64, 128):
    for k in (26, 512):
        set_names(k)
        for kind in ("ideal", "random", "orth", "init"):
            out = []
            for D, r in ((1, 6), (4, 6), (8, 6), (1, 2), (4, 3)):
                a = np.mean([acc(codes(kind, d, k, s), d, 16, D, r, 400, np.random.default_rng(s)) for s in range(3)])
                out.append(f"D{D} r{r}: {100*a:.0f}")
            print(f"d={d} names={k} codes={kind:6s} | " + " | ".join(out))
