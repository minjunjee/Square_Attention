#!/usr/bin/env python3
"""What did a proposed layer (attention choice) learn on the vt task, and where does it fail?

usage: python3 tools/inspect_vt.py ckpt/vt_square_D_s0.pt --name_min 1 --name_max 3 --prog_len 256

Prints
  1. the mean write gate by token type,
  2. at positions where the gate is open, how the key and the value attention is spread over token types,
  3. at the query ':' positions, how the read vector attention is spread over token types,
  4. two-chain accuracy (depth 6) by the length of the queried name, and by whether another variable
     in the program has the same letters in another order (an anagram), which a key made of a
     weighted sum of letter codes could confuse.
"""
import argparse
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/tools/", 1)[0])
import poc_fair as P  # noqa: E402

TYPES = ["VAR", "left name", "=", "right side", ";", "noise", "pad", "#", "query name", ":"]


def token_types(tok, n, m, name_max, prog_len):
    """Type index (into TYPES) of every position of one program."""
    T, _ = P.vt_layout(n, m, name_max, prog_len)
    P_len = T - m * (2 + name_max)
    ty = np.zeros(len(tok), np.int64)
    state = None
    for i, t in enumerate(tok[:P_len]):
        if t == P.VAR_T:
            ty[i], state = 0, "left"
        elif t == P.EQ:
            ty[i], state = 2, "right"
        elif t == P.NL:
            ty[i], state = 4, None
        elif t == P.PAD:
            ty[i] = 6
        elif t > P.PAD:
            ty[i] = 5
        else:
            ty[i] = 1 if state == "left" else 3
    for q in range(m):
        o = P_len + q * (2 + name_max)
        ty[o] = 7
        for j in range(o + 1, o + 1 + name_max):
            ty[j] = 6 if tok[j] == P.PAD else 8
        ty[o + 1 + name_max] = 9
    return ty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--name_min", type=int, default=1)
    ap.add_argument("--name_max", type=int, default=3)
    ap.add_argument("--prog_len", type=int, default=256)
    ap.add_argument("--n_noise", type=int, default=50)
    ap.add_argument("--read", default="square")
    ap.add_argument("--B", type=int, default=1000)
    args = ap.parse_args()
    torch.set_num_threads(8)
    P.V = P.PAD + 1 + args.n_noise
    T, pos = P.vt_layout(16, 4, args.name_max, args.prog_len)
    m = P.Tied(16, 4, read=args.read, gate_ctx=True, tap_attn=8, pos=pos)
    m.load_state_dict(torch.load(args.path, map_location="cpu"))
    m.tap_scale = 10.0
    m.eval()
    kw = dict(name_min=args.name_min, name_max=args.name_max, prog_len=args.prog_len, n_noise=args.n_noise)
    tok, lab, qd, _ = P.make_vt(np.random.default_rng(7), args.B, 16, 4, 4, twin_depth=6, **kw)
    with torch.no_grad():
        t = torch.from_numpy(tok)
        e = m.emb(t)
        m.roles(e)
        att = m.last_att.numpy()                                     # 3 x B x T x W
        beta = torch.sigmoid(m.gate(e.transpose(1, 2))[:, 0, :e.shape[1]]).numpy()
        pred = m(t, r=6)[:, 0].argmax(-1).numpy()
    types = np.stack([token_types(x, 16, 4, args.name_max, args.prog_len) for x in tok])
    W = att.shape[-1]
    print("1. mean write gate by token type")
    print("   " + " | ".join(f"{TYPES[i]} {beta[types == i].mean():.3f}" for i in range(len(TYPES)) if (types == i).any()))
    on = beta > 0.5
    print(f"2. where the gate is open (beta > 0.5, {on.sum()} positions), share of key / value attention by token type")
    for role, name in ((0, "key"), (1, "value")):
        acc = Counter()
        b_idx, t_idx = np.nonzero(on)
        for o in range(W):
            src = t_idx - o
            ok = src >= 0
            for ty_, w in zip(types[b_idx[ok], src[ok]], att[role, b_idx[ok], t_idx[ok], o]):
                acc[ty_] += w
        tot = sum(acc.values())
        print(f"   {name:5s}: " + " | ".join(f"{TYPES[k]} {v / tot:.2f}" for k, v in sorted(acc.items(), key=lambda kv: -kv[1]) if v / tot >= 0.01))
    print("3. at the query ':' positions, share of the read vector attention by token type")
    acc = Counter()
    for p in pos:
        for o in range(W):
            if p - o < 0:
                continue
            for ty_, w in zip(types[:, p - o], att[2, :, p, o]):
                acc[ty_] += w
    tot = sum(acc.values())
    print("   " + " | ".join(f"{TYPES[k]} {v / tot:.2f}" for k, v in sorted(acc.items(), key=lambda kv: -kv[1]) if v / tot >= 0.01))
    ok = pred == lab[:, 0]
    T_, _ = P.vt_layout(16, 4, args.name_max, args.prog_len)
    P_len = T_ - 4 * (2 + args.name_max)
    qname = [tuple(x for x in tok[b, P_len + 1:P_len + 1 + args.name_max] if x != P.PAD) for b in range(len(tok))]
    print(f"4. two-chain accuracy at depth 6: {ok.mean():.3f}")
    for L in range(args.name_min, args.name_max + 1):
        sel = np.array([len(x) == L for x in qname])
        if sel.any():
            print(f"   queried name of {L} letter(s): {ok[sel].mean():.3f} ({sel.sum()} programs)")
    if args.name_max > 1:
        anagram = np.zeros(len(tok), bool)
        for b in range(len(tok)):
            names = set()
            for i in np.nonzero(tok[b, :P_len] == P.VAR_T)[0]:
                j = i + 1
                nm = []
                while tok[b, j] != P.EQ:
                    nm.append(tok[b, j]); j += 1
                names.add(tuple(nm))
            q = qname[b]
            anagram[b] = any(sorted(x) == sorted(q) and x != q for x in names)
        if anagram.any():
            print(f"   another variable is an anagram of the queried name: {ok[anagram].mean():.3f} ({anagram.sum()} programs); "
                  f"otherwise {ok[~anagram].mean():.3f}")


if __name__ == "__main__":
    main()
