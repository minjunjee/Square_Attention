#!/usr/bin/env python3
"""PoC for idea 1: reading a delta-rule linear-attention state by squaring it.

Task (variable dereference): the context is a shuffled list of assignments
"lhs rhs" (rhs is another variable or a value). Some variables are assigned
twice and only the later assignment counts. A query names a variable and the
answer is the value reached by following assignments. Training only contains
chains of depth <= 4; tests use deeper chains.

Models
  power : one delta-rule state S (d x d). At read time S is decoded into a
          relation matrix R over tokens, sinks get a self loop, and R is
          squared r times. Modes:
            full    squaring + sharpening after every squaring (proposed)
            nosharp squaring without sharpening (ablation)
            r0      no squaring, one hop per read (plain linear attention read)
            iter    one hop at a time with sharpening, n_iter sequential steps
            resolvent  dense variant: read with I + S + S^2 + ... built by squaring
  tf    : causal transformer baseline
  lstm  : LSTM baseline
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


# ----------------------------------------------------------------------------- data
def make_batch(rng, B, Vv, Vc, n_st, m, D_lo, D_hi, D_cap, head_query, twin=False, info=False):
    """Returns tokens, roles, labels, query depths (numpy int64 arrays).
    twin=True adds a second chain of the same depth that ends in a different value and asks for
    the head of one of the two chains, so that the answer cannot be guessed without following a chain.
    info=True also returns the number of distinct values that end some chain (the guessing level)."""
    Q = Vv + Vc
    ar = np.arange(B)
    idx = np.tile(np.arange(Vv), (B, 1))
    perm = np.argsort(rng.random((B, Vv)), axis=1)
    D = rng.integers(D_lo, D_hi + 1, size=B)
    # processing order: main chain from its tail to its head, then the other variables
    rev = np.where(idx < D[:, None], D[:, None] - 1 - idx, idx)
    proc = np.take_along_axis(perm, rev, axis=1)
    rhs_p = np.zeros((B, Vv), np.int64)
    dep_p = np.zeros((B, Vv), np.int64)
    val_p = np.zeros((B, Vv), np.int64)
    for i in range(Vv):
        rand_val = rng.integers(Vv, Vv + Vc, size=B)
        if i == 0:
            rhs_p[:, 0] = rand_val
            dep_p[:, 0] = 1
            val_p[:, 0] = rand_val
            continue
        on_chain = i < D
        elig = dep_p[:, :i] <= D_cap - 1
        score = rng.random((B, i))
        score[~elig] = -1.0
        pick = score.argmax(1)
        use_var = elig.any(1) & (rng.random(B) < 0.6)
        tgt = np.where(on_chain, i - 1, pick)
        use_var = np.where(on_chain, True, use_var)
        if twin:
            start2 = i == D                                   # tail of the second chain
            on_chain2 = (i > D) & (i < 2 * D)
            other_val = Vv + (val_p[:, 0] - Vv + 1 + rng.integers(0, Vc - 1, size=B)) % Vc
            tgt = np.where(on_chain2, i - 1, tgt)
            use_var = np.where(on_chain2, True, np.where(start2, False, use_var))
            rand_val = np.where(start2, other_val, rand_val)
        rhs_p[:, i] = np.where(use_var, proc[ar, tgt], rand_val)
        dep_p[:, i] = np.where(use_var, dep_p[ar, tgt] + 1, 1)
        val_p[:, i] = np.where(use_var, val_p[ar, tgt], rand_val)
    rhs = np.zeros((B, Vv), np.int64)
    dep = np.zeros((B, Vv), np.int64)
    val = np.zeros((B, Vv), np.int64)
    np.put_along_axis(rhs, proc, rhs_p, axis=1)
    np.put_along_axis(dep, proc, dep_p, axis=1)
    np.put_along_axis(val, proc, val_p, axis=1)
    head = proc[ar, D - 1]
    # stale assignments that a later assignment overwrites
    st_vars = np.argsort(rng.random((B, Vv)), axis=1)[:, :n_st]
    st_rhs = rng.integers(0, Vv + Vc, size=(B, n_st))
    key_final = rng.random((B, Vv))
    key_stale = rng.random((B, n_st)) * np.take_along_axis(key_final, st_vars, axis=1)
    lhs_all = np.concatenate([idx, st_vars], axis=1)
    rhs_all = np.concatenate([rhs, st_rhs], axis=1)
    order = np.argsort(np.concatenate([key_final, key_stale], axis=1), axis=1)
    lhs_s = np.take_along_axis(lhs_all, order, axis=1)
    rhs_s = np.take_along_axis(rhs_all, order, axis=1)
    n = Vv + n_st
    T = 2 * n + 2 * m
    tokens = np.zeros((B, T), np.int64)
    roles = np.zeros((B, T), np.int64)
    tokens[:, 0:2 * n:2] = lhs_s
    tokens[:, 1:2 * n:2] = rhs_s
    roles[:, 1:2 * n:2] = 1
    qv = rng.integers(0, Vv, size=(B, m))
    if head_query:
        qv[:, 0] = np.where(rng.random(B) < 0.5, head, proc[ar, np.minimum(2 * D, Vv) - 1]) if twin else head
    tokens[:, 2 * n::2] = Q
    tokens[:, 2 * n + 1::2] = qv
    roles[:, 2 * n::2] = 2
    roles[:, 2 * n + 1::2] = 3
    labels = np.take_along_axis(val, qv, axis=1)
    qdep = np.take_along_axis(dep, qv, axis=1)
    if info:
        ends = np.sort(np.where(dep == 1, val, -1), axis=1)
        n_end_values = ((ends[:, 1:] != ends[:, :-1]) & (ends[:, 1:] >= 0)).sum(1) + (ends[:, 0] >= 0)
        return tokens, roles, labels, qdep, n_end_values
    return tokens, roles, labels, qdep


# ----------------------------------------------------------------------------- models
class PowerRead(nn.Module):
    """Delta-rule state written from the token stream, read by repeated squaring."""

    def __init__(self, V, n_edge_tokens, dm=64, d=32, mode="full", p_infl=2.0, tied=True,
                 tau=0.05, theta=0.5, learn_temp=False):
        super().__init__()
        self.V, self.n_edge, self.d, self.mode, self.p, self.tied = V, n_edge_tokens, d, mode, p_infl, tied
        self.hard = False                           # inference only: pointer jumping on hard pointers
        self.tok = nn.Embedding(V, dm)
        self.role = nn.Embedding(4, dm)
        self.Wk = nn.Linear(dm, d, bias=False)      # token -> entity code (used as key, and as value when tied)
        if not tied:
            self.Wv = nn.Linear(dm, d, bias=False)  # token -> value code
            self.Wb = nn.Parameter(torch.eye(d) + 0.01 * torch.randn(d, d))  # value space -> key space
        self.gate = nn.Linear(dm, 1)                # when to write
        if mode == "resolvent":
            self.head = nn.Linear(d, V)             # reads the value out of the summed path
        if learn_temp:
            self.theta = nn.Parameter(torch.tensor(float(theta)))       # self-loop threshold
            self.logtau = nn.Parameter(torch.tensor(math.log(tau)))
        else:
            self.register_buffer("theta", torch.tensor(float(theta)))
            self.register_buffer("logtau", torch.tensor(math.log(tau)))

    def write(self, tokens, roles):
        B = tokens.shape[0]
        E = self.tok.weight
        K = F.normalize(self.Wk(E), dim=1)          # V x d
        Vcode = K if self.tied else self.Wv(E)      # V x d
        beta = torch.sigmoid(self.gate(self.tok(tokens) + self.role(roles))).squeeze(-1)
        S = torch.zeros(B, self.d, self.d, device=tokens.device)
        for t in range(1, self.n_edge):
            k = K[tokens[:, t - 1]]
            v = Vcode[tokens[:, t]]
            Sk = torch.einsum("bij,bj->bi", S, k)
            S = S + (beta[:, t, None] * (v - Sk))[:, :, None] * k[:, None, :]
        return S, K

    def relation(self, S, K):
        A = S if self.tied else torch.einsum("ij,bjk->bik", self.Wb, S)   # key -> key
        G = torch.einsum("vi,bij,wj->bvw", K, A, K)                 # G[b, target, source]
        eye = torch.eye(self.V, device=S.device)
        return torch.softmax((G + self.theta * eye) / self.logtau.exp(), dim=1)

    def sharpen(self, M):
        M = M.clamp_min(1e-12) ** self.p
        return M / M.sum(dim=1, keepdim=True)

    def forward(self, tokens, roles, r=3, n_iter=None):
        S, K = self.write(tokens, roles)
        qtok = tokens[:, self.n_edge + 1::2]                         # B x m
        B, m = qtok.shape
        if self.mode == "resolvent":
            # dense read: P = I + S + S^2 + ... + S^(2^r - 1), built with r squarings, no entity decode
            P = torch.eye(self.d, device=S.device).expand(B, -1, -1)
            M = S
            for _ in range(r):
                P = P + M @ P
                M = M @ M
            return self.head(torch.einsum("bij,bmj->bmi", P, K[qtok]))
        R = self.relation(S, K)
        if self.hard:
            # every column of R is close to one-hot, so squaring is pointer jumping: ptr <- ptr[ptr].
            # This costs O(V) per squaring instead of a V x V matrix product.
            ptr = R.argmax(dim=1)                                    # ptr[b, source] = target
            for _ in range(r):
                ptr = ptr.gather(1, ptr)
            return torch.log(F.one_hot(ptr.gather(1, qtok), self.V).float() + 1e-8)
        if self.mode == "iter":
            y = F.one_hot(qtok, self.V).float().transpose(1, 2)      # B x V x m
            for _ in range(n_iter if n_iter is not None else 2 ** r):
                y = self.sharpen(R @ y)
        else:
            steps = 0 if self.mode == "r0" else r
            for _ in range(steps):
                R = R @ R
                if self.mode == "full":
                    R = self.sharpen(R)
            y = R.gather(2, qtok[:, None, :].expand(B, self.V, m))   # B x V x m
        return torch.log(y.transpose(1, 2) + 1e-8)                   # B x m x V


class TF(nn.Module):
    def __init__(self, V, T, n_edge_tokens, dm=64, L=4, H=4):
        super().__init__()
        self.n_edge = n_edge_tokens
        self.tok = nn.Embedding(V, dm)
        self.role = nn.Embedding(4, dm)
        self.pos = nn.Embedding(T, dm)
        layer = nn.TransformerEncoderLayer(dm, H, 4 * dm, dropout=0.0, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, L, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, V)

    def forward(self, tokens, roles, **_):
        T = tokens.shape[1]
        x = self.tok(tokens) + self.role(roles) + self.pos(torch.arange(T, device=tokens.device))
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tokens.device), 1)
        h = self.ln(self.enc(x, mask=mask))
        return self.head(h[:, self.n_edge + 1::2])


class LSTM(nn.Module):
    def __init__(self, V, n_edge_tokens, dm=64, hid=128, L=2):
        super().__init__()
        self.n_edge = n_edge_tokens
        self.tok = nn.Embedding(V, dm)
        self.role = nn.Embedding(4, dm)
        self.rnn = nn.LSTM(dm, hid, L, batch_first=True)
        self.head = nn.Linear(hid, V)

    def forward(self, tokens, roles, **_):
        h, _ = self.rnn(self.tok(tokens) + self.role(roles))
        return self.head(h[:, self.n_edge + 1::2])


# ----------------------------------------------------------------------------- train / eval
def to_t(arrs, dev):
    return [torch.from_numpy(a).to(dev) for a in arrs]


def evaluate(model, args, rng, dev, depths, r_eval, twin, n=2000):
    """Accuracy on the head of a chain of depth D. Also returns the accuracy of guessing
    uniformly among the values that end some chain in the context.
    twin=True: two chains of depth D that end in different values (the strict test)."""
    model.eval()
    out, guess = {}, {}
    with torch.no_grad():
        for D in depths:
            *arrs, n_end = make_batch(rng, n, args.Vv, args.Vc, args.n_st, args.m, D, D,
                                      max(D, args.D_train), True, twin=twin, info=True)
            tok, rol, lab, _ = to_t(arrs, dev)
            logits = model(tok, rol, r=r_eval, n_iter=args.Vv)
            out[D] = round((logits[:, 0].argmax(-1) == lab[:, 0]).float().mean().item(), 4)
            guess[D] = round(float((1.0 / n_end).mean()), 4)
    model.train()
    return out, guess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="power", choices=["power", "tf", "lstm"])
    ap.add_argument("--mode", default="full", choices=["full", "nosharp", "r0", "iter", "resolvent"])
    ap.add_argument("--Vv", type=int, default=12)
    ap.add_argument("--Vc", type=int, default=6)
    ap.add_argument("--n_st", type=int, default=3)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--D_train", type=int, default=4)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--dm", type=int, default=64)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--r_lo", type=int, default=2)
    ap.add_argument("--r_hi", type=int, default=5)
    ap.add_argument("--untied", action="store_true")
    ap.add_argument("--learn_temp", action="store_true")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max_seconds", type=float, default=1e9)
    ap.add_argument("--depths", default="1,2,3,4,5,6,7,8,10,12,16", help="depths of the two-chain test")
    ap.add_argument("--depths_single", default="1,2,3,4,5,6,8,10,12", help="depths of the one-chain test")
    ap.add_argument("--save", default=None, help="file for the trained weights")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    gpu_guard.check()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    V = args.Vv + args.Vc + 1
    n_edge = 2 * (args.Vv + args.n_st)
    T = n_edge + 2 * args.m
    if args.model == "power":
        model = PowerRead(V, n_edge, args.dm, args.d, args.mode, tied=not args.untied,
                          tau=args.tau, theta=args.theta, learn_temp=args.learn_temp)
        lr = args.lr or 3e-3
    elif args.model == "tf":
        model = TF(V, T, n_edge, args.dm, args.L)
        lr = args.lr or 1e-3
    else:
        model = LSTM(V, n_edge, args.dm)
        lr = args.lr or 2e-3
    model.to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0 if args.model == "power" else 0.01)
    warm = 300
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    depths = [int(x) for x in args.depths.split(",") if 2 * int(x) <= args.Vv]
    depths_single = [int(x) for x in args.depths_single.split(",") if int(x) <= args.Vv]
    r_eval = max(args.r_hi, math.ceil(math.log2(args.Vv)) + 1)
    t0 = time.time()
    log = []
    step = 0
    for step in range(1, args.steps + 1):
        tok, rol, lab, _ = to_t(make_batch(rng, args.bs, args.Vv, args.Vc, args.n_st, args.m,
                                           1, args.D_train, args.D_train, False), dev)
        r = int(rng.integers(args.r_lo, args.r_hi + 1))       # random number of squarings
        logits = model(tok, rol, r=r, n_iter=int(rng.integers(args.D_train, 2 * args.D_train + 1)))
        loss = F.cross_entropy(logits.reshape(-1, V), lab.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(1, args.steps // 10) == 0 or step == 1:
            acc = (logits.argmax(-1) == lab).float().mean().item()
            log.append((step, round(loss.item(), 4), round(acc, 4), round(time.time() - t0, 1)))
            print(f"step {step} loss {loss.item():.4f} acc {acc:.3f} t {time.time() - t0:.0f}s", flush=True)
        if time.time() - t0 > args.max_seconds:
            print("time cap reached at step", step, flush=True)
            break
    train_seconds = round(time.time() - t0, 1)
    if args.save:
        torch.save(model.state_dict(), args.save)
    # one-chain test first (same random stream as the first version of this script), then the strict two-chain test
    res_single, guess_single = evaluate(model, args, np.random.default_rng(10_000 + args.seed), dev,
                                        depths_single, r_eval, twin=False)
    res, guess = evaluate(model, args, np.random.default_rng(20_000 + args.seed), dev, depths, r_eval, twin=True)
    res_hard = None
    if args.model == "power" and args.mode == "full":
        model.hard = True
        res_hard, _ = evaluate(model, args, np.random.default_rng(20_000 + args.seed), dev, depths, r_eval, twin=True)
        model.hard = False
    summary = dict(model=args.model, mode=args.mode if args.model == "power" else None,
                   L=args.L if args.model == "tf" else None, seed=args.seed, params=n_params,
                   steps=step, bs=args.bs, train_seconds=train_seconds, r_train=[args.r_lo, args.r_hi],
                   r_eval=r_eval, tied=not args.untied, tag=args.tag, Vv=args.Vv, Vc=args.Vc, n_st=args.n_st,
                   d=args.d, D_train=args.D_train, acc_by_depth=res, guess_by_depth=guess,
                   acc_single=res_single, guess_single=guess_single, acc_by_depth_hard=res_hard, log=log)
    if args.model == "power":
        summary["theta"] = round(float(model.theta), 3)
        summary["tau"] = round(float(model.logtau.exp()), 4)
        with torch.no_grad():
            Kc = F.normalize(model.Wk(model.tok.weight), dim=1)
            summary["max_code_overlap"] = round((Kc @ Kc.T - torch.eye(V, device=Kc.device)).abs().max().item(), 3)
    print("RESULT", json.dumps(summary))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
