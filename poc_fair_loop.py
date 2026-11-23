#!/usr/bin/env python3
"""Fair test of idea 1 on a task taken from another paper.

Task: variable binding in symbolic programs, following Wu, Geiger and Milliere (ICML 2025).
A program has 16 assignment lines "x=5" or "x=y" (y was assigned on an earlier line) and
query lines "#x:". The answer is the digit that x resolves to. Character-level tokens, no
extra input features. A line gets a digit with probability 0.3, otherwise it refers to an
earlier variable chosen with probability proportional to its depth cubed. Training programs
contain chains of depth 1 to 4 only, as in the paper. Details the paper leaves open and
that were chosen here: every variable is assigned once, and each program has 4 queries
whose answers are not fed back into the sequence.

Tests
  in-distribution   programs from the training generator, by depth of the query (1 to 4)
  deeper chains     two chains of the same depth D (1 to 8) that end in different digits,
                    mixed with ordinary lines; the query asks for the head of one of them

Models (all read the same token sequence and get the same training budget)
  tied    one fast-weight layer whose keys, values and queries come from one shared code map.
          Which of the last 4 tokens is the key, the value or the query is learned.
          The state starts as the identity, so an entity that was never written maps to itself.
            --read square   the state is squared r times before it is read (proposed)
            --read once     the state is read once (same layer, standard read)
            --read iter     the state is read 2^r times in sequence (same result, more steps)
  delta   standard DeltaNet-style blocks (separate key/value/query maps, short convolutions)
            --loops n       one block applied n times with shared weights
  tf      causal Transformer with learned positions
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gpu_guard

LET, DIG = 26, 10
EQ, NL, HASH, COL = 36, 37, 38, 39
V = 40


# ----------------------------------------------------------------------------- data
def gen_lines(rng, B, n, D_cap, plan=None):
    """Assignment lines in program order. plan (B, n): 0 ordinary line, 1 chain A, 2 chain B."""
    ar = np.arange(B)
    lhs = np.argsort(rng.random((B, LET)), axis=1)[:, :n]            # every variable is assigned once
    rhs = np.zeros((B, n), np.int64)
    dep = np.zeros((B, n), np.int64)
    val = np.zeros((B, n), np.int64)
    last = {1: np.full(B, -1), 2: np.full(B, -1)}
    cA = rng.integers(0, DIG, size=B)
    cB = (cA + 1 + rng.integers(0, DIG - 1, size=B)) % DIG            # a different digit
    for i in range(n):
        digit = rng.integers(0, DIG, size=B)
        if i > 0:
            wgt = (dep[:, :i].astype(np.float64) ** 3) * (dep[:, :i] <= D_cap - 1)
            tot = wgt.sum(axis=1)
            j = (np.cumsum(wgt, axis=1) > (rng.random(B) * tot)[:, None]).argmax(axis=1)
            use_var = (tot > 0) & (rng.random(B) < 0.7)
        else:
            j = np.zeros(B, np.int64)
            use_var = np.zeros(B, bool)
        if plan is not None:
            for c, const in ((1, cA), (2, cB)):
                on = plan[:, i] == c
                first, cont = on & (last[c] < 0), on & (last[c] >= 0)
                digit = np.where(first, const, digit)
                use_var = np.where(first, False, np.where(cont, True, use_var))
                j = np.where(cont, last[c], j)
                last[c] = np.where(on, i, last[c])
        rhs[:, i] = np.where(use_var, lhs[ar, j], LET + digit)
        dep[:, i] = np.where(use_var, dep[ar, j] + 1, 1)
        val[:, i] = np.where(use_var, val[ar, j], digit)
    return lhs, rhs, dep, val, last


def make_batch(rng, B, n=16, m=4, D_cap=4, twin_depth=None, q0_depth=None):
    """Returns tokens (B, 4n + 3m), labels (B, m) as token ids, depth of each query (B, m),
    and the number of distinct digits that end a chain (for the guessing level)."""
    ar = np.arange(B)
    plan = None
    if twin_depth is not None:
        D = twin_depth
        order = np.argsort(rng.random((B, n)), axis=1)
        plan = np.zeros((B, n), np.int64)
        np.put_along_axis(plan, order[:, :D], 1, axis=1)
        np.put_along_axis(plan, order[:, D:2 * D], 2, axis=1)
        D_cap = max(D_cap, D)
    lhs, rhs, dep, val, last = gen_lines(rng, B, n, D_cap, plan)
    target = rng.integers(1, D_cap + 1, size=(B, m))                 # wanted depth of each query
    if q0_depth is not None:
        target[:, 0] = q0_depth
    qline = np.zeros((B, m), np.int64)
    for q in range(m):
        qline[:, q] = (rng.random((B, n)) + 10.0 * (dep == target[:, q:q + 1])).argmax(axis=1)
    if twin_depth is not None:
        qline[:, 0] = np.where(rng.random(B) < 0.5, last[1], last[2])
    T = 4 * n + 3 * m
    tok = np.zeros((B, T), np.int64)
    tok[:, 0:4 * n:4] = lhs
    tok[:, 1:4 * n:4] = EQ
    tok[:, 2:4 * n:4] = rhs
    tok[:, 3:4 * n:4] = NL
    tok[:, 4 * n::3] = HASH
    tok[:, 4 * n + 1::3] = np.take_along_axis(lhs, qline, axis=1)
    tok[:, 4 * n + 2::3] = COL
    labels = LET + np.take_along_axis(val, qline, axis=1)
    qdep = np.take_along_axis(dep, qline, axis=1)
    ends = np.sort(np.where(dep == 1, val, -1), axis=1)
    n_end = ((ends[:, 1:] != ends[:, :-1]) & (ends[:, 1:] >= 0)).sum(axis=1) + (ends[:, 0] >= 0)
    return tok, labels, qdep, n_end


def read_positions(n, m):
    return [4 * n + 3 * q + 2 for q in range(m)]


# ----------------------------------------------------------------------------- models
def unit(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


class MLPHead(nn.Module):
    def __init__(self, dm):
        super().__init__()
        self.ln1 = nn.LayerNorm(dm)
        self.mlp = nn.Sequential(nn.Linear(dm, 4 * dm), nn.GELU(), nn.Linear(4 * dm, dm))
        self.ln2 = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, V)

    def forward(self, h):
        h = h + self.mlp(self.ln1(h))
        return self.head(self.ln2(h))


class Tied(nn.Module):
    """One fast-weight layer. Keys, values and queries are codes of recent tokens under one
    shared code map; the mixing weights over the last w tokens are learned for each of the
    three roles. Delta-rule write into a state that starts as the identity."""

    def __init__(self, n, m, dm=64, d=64, w=4, read="square"):
        super().__init__()
        self.n, self.m, self.d, self.w, self.read = n, m, d, w, read
        self.emb = nn.Embedding(V, dm)
        self.code = nn.Linear(dm, d, bias=False)                     # shared code map
        self.taps = nn.Parameter(0.5 * torch.randn(3, w))            # rows: key, value, query
        self.gate = nn.Linear(dm, 1)                                 # when to write
        self.out = nn.Linear(d, dm, bias=False)
        self.top = MLPHead(dm)

    def roles(self, e):
        c = self.code(e)                                             # B x T x d
        B, T, d = c.shape
        pad = torch.cat([c.new_zeros(B, self.w - 1, d), c], dim=1)
        win = torch.stack([pad[:, self.w - 1 - o:self.w - 1 - o + T] for o in range(self.w)], dim=2)  # offset o
        mix = torch.einsum("rw,btwd->rbtd", torch.softmax(self.taps, dim=1), win)
        return unit(mix[0]), unit(mix[1]), unit(mix[2])              # key, value, query

    def forward(self, tok, r=3):
        e = self.emb(tok)
        B, T, _ = e.shape
        k, v, q = self.roles(e)
        beta = torch.sigmoid(self.gate(e)).squeeze(-1)
        S = torch.eye(self.d, device=tok.device).expand(B, -1, -1)
        pos = read_positions(self.n, self.m)
        ys = []
        for t in range(T):
            Sk = torch.einsum("bij,bj->bi", S, k[:, t])
            S = S + (beta[:, t, None] * (v[:, t] - Sk))[:, :, None] * k[:, t, None, :]
            if t in pos:
                ys.append(self.read_state(S, q[:, t], r))
        y = torch.stack(ys, dim=1)                                   # B x m x d
        return self.top(e[:, pos] + self.out(y))

    def read_state(self, S, q, r):
        if self.read == "once":
            return unit(torch.einsum("bij,bj->bi", S, q))
        if self.read == "iter":
            y = q
            for _ in range(2 ** r):
                y = unit(torch.einsum("bij,bj->bi", S, y))
            return y
        M = S
        for _ in range(r):                                           # M = S^(2^r), rescaled to stay bounded
            M = M @ M
            M = M / (M.flatten(1).norm(dim=1)[:, None, None] / math.sqrt(self.d) + 1e-6)
        return unit(torch.einsum("bij,bj->bi", M, q))


class CausalDW(nn.Module):
    """Short causal depthwise convolution, as used on q, k, v in DeltaNet implementations."""

    def __init__(self, ch, w):
        super().__init__()
        self.w = w
        self.conv = nn.Conv1d(ch, ch, w, groups=ch, padding=w - 1, bias=False)

    def forward(self, x):                                            # B x T x ch
        return self.conv(x.transpose(1, 2))[:, :, :x.shape[1]].transpose(1, 2)


class DeltaBlock(nn.Module):
    def __init__(self, dm, H, w=4):
        super().__init__()
        self.H, self.dh = H, dm // H
        self.ln1 = nn.LayerNorm(dm)
        self.q, self.k, self.v = (nn.Linear(dm, dm, bias=False) for _ in range(3))
        self.cq, self.ck, self.cv = (CausalDW(dm, w) for _ in range(3))
        self.b = nn.Linear(dm, H)
        self.o = nn.Linear(dm, dm, bias=False)
        self.ln2 = nn.LayerNorm(dm)
        self.mlp = nn.Sequential(nn.Linear(dm, 4 * dm), nn.GELU(), nn.Linear(4 * dm, dm))

    def forward(self, h):
        B, T, dm = h.shape
        x = self.ln1(h)
        split = lambda z: z.view(B, T, self.H, self.dh)
        q = unit(split(self.cq(self.q(x))))
        k = unit(split(self.ck(self.k(x))))
        v = split(self.cv(self.v(x)))
        beta = torch.sigmoid(self.b(x))                              # B x T x H
        S = h.new_zeros(B, self.H, self.dh, self.dh)
        outs = []
        for t in range(T):
            Sk = torch.einsum("bhij,bhj->bhi", S, k[:, t])
            S = S + (beta[:, t, :, None] * (v[:, t] - Sk))[..., None] * k[:, t, :, None, :]
            outs.append(torch.einsum("bhij,bhj->bhi", S, q[:, t]))
        h = h + self.o(torch.stack(outs, dim=1).reshape(B, T, dm))
        return h + self.mlp(self.ln2(h))


class Delta(nn.Module):
    def __init__(self, n, m, dm=64, L=2, H=4, loops=0):
        super().__init__()
        self.n, self.m, self.loops = n, m, loops
        self.emb = nn.Embedding(V, dm)
        self.blocks = nn.ModuleList(DeltaBlock(dm, H) for _ in range(1 if loops else L))
        self.ln = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, V)

    def forward(self, tok, r=None, loops=None):
        h = self.emb(tok)
        if self.loops:
            for _ in range(loops or self.loops):
                h = self.blocks[0](h)
        else:
            for blk in self.blocks:
                h = blk(h)
        return self.head(self.ln(h[:, read_positions(self.n, self.m)]))


class TF(nn.Module):
    def __init__(self, n, m, dm=64, L=4, H=4):
        super().__init__()
        self.n, self.m = n, m
        T = 4 * n + 3 * m
        self.emb = nn.Embedding(V, dm)
        self.pos = nn.Embedding(T, dm)
        layer = nn.TransformerEncoderLayer(dm, H, 4 * dm, dropout=0.0, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, V)

    def forward(self, tok, r=None, loops=None):
        T = tok.shape[1]
        x = self.emb(tok) + self.pos(torch.arange(T, device=tok.device))
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tok.device), 1)
        h = self.ln(self.enc(x, mask=mask))
        return self.head(h[:, read_positions(self.n, self.m)])


# ----------------------------------------------------------------------------- train / eval
def evaluate(model, args, rng, dev, r_eval, loops_eval=None, n_eval=2000):
    """In-distribution accuracy by query depth, and accuracy on two competing chains of depth D."""
    model.eval()
    ind, deep, guess = {}, {}, {}
    kw = dict(r=r_eval) if loops_eval is None else dict(r=r_eval, loops=loops_eval)
    with torch.no_grad():
        for D in range(1, args.D_train + 1):
            tok, lab, qd, _ = make_batch(rng, n_eval, args.n, args.m, args.D_train, q0_depth=D)
            pred = model(torch.from_numpy(tok).to(dev), **kw)[:, 0].argmax(-1).cpu().numpy()
            ok = qd[:, 0] == D
            ind[D] = round(float((pred[ok] == lab[ok, 0]).mean()), 4)
        for D in range(1, args.n // 2 + 1):
            tok, lab, qd, n_end = make_batch(rng, n_eval, args.n, args.m, args.D_train, twin_depth=D)
            pred = model(torch.from_numpy(tok).to(dev), **kw)[:, 0].argmax(-1).cpu().numpy()
            deep[D] = round(float((pred == lab[:, 0]).mean()), 4)
            guess[D] = round(float((1.0 / n_end).mean()), 4)
    model.train()
    return ind, deep, guess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tied", choices=["tied", "delta", "tf"])
    ap.add_argument("--read", default="square", choices=["square", "once", "iter"])
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--loops", type=int, default=0, help="delta only: apply one shared block this many times")
    ap.add_argument("--loops_eval", type=int, default=None)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--D_train", type=int, default=4)
    ap.add_argument("--dm", type=int, default=64)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--r_lo", type=int, default=2)
    ap.add_argument("--r_hi", type=int, default=4)
    ap.add_argument("--r_eval", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--max_seconds", type=float, default=1e9)
    ap.add_argument("--eval_every", type=int, default=0, help="also record test accuracy every this many steps")
    ap.add_argument("--save", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    gpu_guard.check()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model == "tied":
        model = Tied(args.n, args.m, args.dm, args.d, read=args.read)
        lr = args.lr or 3e-3
    elif args.model == "delta":
        model = Delta(args.n, args.m, args.dm, args.L, loops=args.loops)
        lr = args.lr or 2e-3
    else:
        model = TF(args.n, args.m, args.dm, args.L)
        lr = args.lr or 1e-3
    model.to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    warm = 300
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    t0 = time.time()
    log, curve = [], []
    step = 0
    for step in range(1, args.steps + 1):
        tok, lab, _, _ = make_batch(rng, args.bs, args.n, args.m, args.D_train)
        r = int(rng.integers(args.r_lo, args.r_hi + 1))
        logits = model(torch.from_numpy(tok).to(dev), r=r)
        loss = F.cross_entropy(logits.reshape(-1, V), torch.from_numpy(lab).to(dev).reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(1, args.steps // 20) == 0 or step == 1:
            acc = (logits.argmax(-1).cpu().numpy() == lab).mean()
            log.append((step, round(loss.item(), 4), round(float(acc), 4), round(time.time() - t0, 1)))
            print(f"step {step} loss {loss.item():.4f} acc {acc:.3f} t {time.time() - t0:.0f}s", flush=True)
        if args.eval_every and step % args.eval_every == 0:
            ind, deep, _ = evaluate(model, args, np.random.default_rng(5), dev, args.r_eval, args.loops_eval, n_eval=500)
            curve.append((step, ind, deep))
        if time.time() - t0 > args.max_seconds:
            print("time cap reached at step", step, flush=True)
            break
    train_seconds = round(time.time() - t0, 1)
    if args.save:
        torch.save(model.state_dict(), args.save)
    ind, deep, guess = evaluate(model, args, np.random.default_rng(10_000 + args.seed), dev, args.r_eval, args.loops_eval)
    summary = dict(model=args.model, read=args.read if args.model == "tied" else None,
                   L=args.L if args.model != "tied" else 1, loops=args.loops, loops_eval=args.loops_eval,
                   seed=args.seed, params=n_params, steps=step, bs=args.bs, lr=lr, train_seconds=train_seconds,
                   n=args.n, m=args.m, D_train=args.D_train, dm=args.dm, d=args.d,
                   r_train=[args.r_lo, args.r_hi], r_eval=args.r_eval, tag=args.tag,
                   acc_in_dist=ind, acc_deep=deep, guess_deep=guess, log=log, curve=curve)
    if args.model == "tied":
        with torch.no_grad():
            summary["taps"] = [[round(x, 3) for x in row] for row in torch.softmax(model.taps, dim=1).tolist()]
            if args.read == "square":                                  # how many squarings does each depth need?
                by_r = {}
                for rr in range(0, 6):
                    _, dd, _ = evaluate(model, args, np.random.default_rng(20_000 + args.seed), dev, rr, n_eval=1000)
                    by_r[rr] = dd
                summary["acc_deep_by_squarings"] = by_r
    print("RESULT", json.dumps(summary))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
