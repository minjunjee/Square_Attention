#!/usr/bin/env python3
"""How do saved models of the interference test place their codes? (CPU only, no training)

For each model: accuracy on depth-1 questions of the training distribution, the number of directions the
name codes effectively use (participation ratio of their singular values), the cosine between two name codes
and between two digit codes, the share of a name code that lies inside the span of the digit codes, and the
mean write gate at each kind of position. The last line is an untrained model with 512 names.

usage: .venv/bin/python tools/code_geometry.py > results/code_geometry_e8.txt
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import poc_fair as P
torch.set_num_threads(8)
V0 = 40
def load(path, n_names, d):
    P.NAMES = np.concatenate([np.arange(P.LET), np.arange(40, 40 + n_names - P.LET)]) if n_names > P.LET else np.arange(P.LET)
    P.V = V0 + max(0, n_names - P.LET)
    m = P.Tied(16, 4, 64, d, read="square", gate_ctx=True, hard_taps=0, tap_attn=8, line=4)
    m.load_state_dict(torch.load(path, map_location="cpu")); m.eval(); m.tap_scale = 10.0
    return m
def stats(name, path, n_names, d):
    m = load(path, n_names, d)
    with torch.no_grad():
        C = P.unit(m.code(m.emb.weight))
        names, digs = C[P.NAMES], C[P.LET:P.LET + P.DIG]
        nn_ = (names @ names.T); nn_off = nn_[~torch.eye(len(names), dtype=torch.bool)]
        dd = digs @ digs.T; dd_off = dd[~torch.eye(10, dtype=torch.bool)]
        nd = names @ digs.T
        # share of name-code energy inside the span of the digit codes
        Q = torch.linalg.qr(digs.T)[0]
        in_dig = ((names @ Q) ** 2).sum(1)
        sv = torch.linalg.svdvals(names); pr = float((sv ** 2).sum() ** 2 / (sv ** 4).sum())
        tok, lab, qd, _ = P.make_batch(np.random.default_rng(1), 500, 16, 4, 4)
        pred = m(torch.from_numpy(tok), r=6).argmax(-1).numpy()
        acc1 = (pred == lab)[qd == 1].mean()
        # gate by position type
        e = m.emb(torch.from_numpy(tok)); beta = torch.sigmoid(m.gate(e.transpose(1, 2))[:, 0, :e.shape[1]])
        g = [float(beta[:, o:64:4].mean()) for o in range(4)] + [float(beta[:, 64 + o::3].mean()) for o in range(3)]
    print(f"{name}: depth-1 acc {acc1:.2f} | name-name |cos| mean {nn_off.abs().mean():.3f} rms {nn_off.pow(2).mean().sqrt():.3f} | "
          f"digit-digit |cos| {dd_off.abs().mean():.3f} | name-digit rms {nd.pow(2).mean().sqrt():.3f} | name energy in digit span {in_dig.mean():.3f} "
          f"| name participation ratio {pr:.1f}\n    gate at lhs,=,rhs,NL: {[round(x,2) for x in g[:4]]}  at #,name,':' : {[round(x,2) for x in g[4:]]}")
rows = [("names48 s0 (success)", "ckpt/e8b_tied_d64_names48_s0.pt", 48, 64),
        ("names64 s0 (success)", "ckpt/e8b_tied_d64_names64_s0.pt", 64, 64),
        ("names256 s2 (success)", "ckpt/e8b_tied_d64_names256_s2.pt", 256, 64),
        ("names256 s0 (trick)", "ckpt/e8b_tied_d64_names256_s0.pt", 256, 64),
        ("names512 s0 (trick)", "ckpt/e8_tied_d64_n16_s0.pt", 512, 64),
        ("names512 s1 (trick)", "ckpt/e8_tied_d64_n16_s1.pt", 512, 64),
        ("names512 s2 (1 hop only)", "ckpt/e8_tied_d64_n16_s2.pt", 512, 64),
        ("names512 d128 s0 (success)", "ckpt/e8b_tied_d128_names512_s0.pt", 512, 128),
        ("names512 d128 s1 (trick)", "ckpt/e8b_tied_d128_names512_s1.pt", 512, 128)]
for r in rows: stats(*r)
# reference: untrained models with 512 names, seeds 0 to 4
for seed in range(5):
    torch.manual_seed(seed)
    P.NAMES = np.concatenate([np.arange(P.LET), np.arange(40, 40 + 512 - P.LET)]); P.V = V0 + 512 - P.LET
    m = P.Tied(16, 4, 64, 64, read="square", gate_ctx=True, tap_attn=8)
    print(f"untrained, 512 names, d=64, seed {seed}: {P.code_geom(m)}")
