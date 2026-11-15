#!/usr/bin/env python3
"""Shows what a trained proposed layer (poc_fair.py --model tied) has actually learned.

usage: python3 inspect_fair.py fair_tied_s1.pt [--gate_ctx 1] [--tap_scale 10] [--hard_taps 0]

Prints
  1. the learned mixing weights over the last 4 tokens for key, value and read vector
  2. the mean write gate at every kind of position (left variable, '=', right side, newline, '#', query variable, ':')
  3. for one program: where the state sends each token's code (nearest token code and cosine)
  4. for one query: the nearest token after 1, 2, 3, ... multiplications by the state
The intended solution would show: key = token 3 back, value = token 1 back, gate on at the newline,
and the state sending each variable to the token on the right side of its line.
"""
import argparse

import numpy as np
import torch

import poc_fair as P

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--gate_ctx", type=int, default=1)
ap.add_argument("--tap_scale", type=float, default=10.0)
ap.add_argument("--hard_taps", type=int, default=0)
ap.add_argument("--depth", type=int, default=6)
args = ap.parse_args()

names = [chr(97 + i) for i in range(26)] + [str(i) for i in range(10)] + ["=", "\\n", "#", ":"]
m = P.Tied(16, 4, read="square", gate_ctx=bool(args.gate_ctx), hard_taps=args.hard_taps)
m.load_state_dict(torch.load(args.path, map_location="cpu"))
m.tap_scale = args.tap_scale
m.eval()
tok, lab, qd, _ = P.make_batch(np.random.default_rng(3), 512, twin_depth=args.depth)
t = torch.from_numpy(tok)
with torch.no_grad():
    e = m.emb(t)
    k, v, q = m.roles(e)
    beta = (torch.sigmoid(m.gate(e.transpose(1, 2))[:, 0, :e.shape[1]]) if m.gate_ctx
            else torch.sigmoid(m.gate(e)).squeeze(-1)).numpy()
    taps = torch.softmax(m.taps * m.tap_scale, dim=1).numpy().round(2)
    print("mixing weights over [current token, 1 back, 2 back, 3 back]")
    for name, row in zip(("key  ", "value", "read "), taps):
        print("  ", name, row)
    print("mean write gate: left variable %.3f | '=' %.3f | right side %.3f | newline %.3f | '#' %.3f | query variable %.3f | ':' %.3f"
          % (beta[:, 0:64:4].mean(), beta[:, 1:64:4].mean(), beta[:, 2:64:4].mean(), beta[:, 3:64:4].mean(),
             beta[:, 64::3].mean(), beta[:, 65::3].mean(), beta[:, 66::3].mean()))
    C = P.unit(m.code(m.emb.weight))
    U = P.delta_updates(k, v, torch.from_numpy(beta), 1.0)
    p, b = 66, 0                                               # first ':' position, first program
    S = torch.eye(m.d) + U[b, :p + 1].T @ k[b, :p + 1]
    print("program:", "".join(names[x] if names[x] != "\\n" else " | " for x in tok[b]))
    SC = C @ S.T
    near = (P.unit(SC) @ C.T).argmax(-1)
    cos = (P.unit(SC) * C[near]).sum(-1)
    print("state sends:", ", ".join(f"{names[x]}->{names[int(near[x])]}({cos[x]:.2f})" for x in sorted(set(tok[b].tolist()))))
    y, path = q[b, p], []
    for step in range(1, 65):
        y = P.unit(S @ y)
        if step in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64):
            j = int((y @ C.T).argmax())
            path.append(f"{step}:{names[j]}({float(y @ C[j]):.2f})")
    print("query", names[tok[b, 65]], "answer", names[lab[b, 0]], "| nearest token after n multiplications:", " ".join(path))
