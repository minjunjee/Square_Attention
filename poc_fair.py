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
            --tap_entropy x add x times the entropy of the mixing weights to the loss
            --tap_attn w    choose each role's token by attention over the last w tokens
  delta   standard DeltaNet-style blocks (separate key/value/query maps, short convolutions)
            --loops n       one block applied n times with shared weights
  tf      causal Transformer with learned positions
            --loops n       one block applied n times with shared weights
  lstm    stacked LSTM
Second task (--task lego), adapted from Zhang et al. 2022 (LEGO, task 1): n clauses "a=+b;" or "a=-b;"
forming one chain that starts at "x=+1;" or "x=-1;", in random order, followed by queries "#a:" whose
answer is the sign of a (+ or -). Training queries ask only for the first n_tr variables of the chain;
the test asks for every position. --signed 1 lets the tied layer multiply each value by a learned sign.
For looped models, --loops_lo/--loops_hi draw the number of applications at every training
step, and the deeper-chain test is repeated for several numbers of applications.
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
ONE, PLUS, MINUS = LET + 1, LET + 4, LET + 5                     # lego task: digit ids reused for '1', '+', '-'
VAR_T, PAD = 40, 41                                              # vt task: 'VAR' and padding; noise words follow, V grows


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


def make_lego(rng, B, n=12, m=4, n_tr=6, q0_depth=None):
    """LEGO task 1 with queries at the end. Clause i of the chain (depth i + 1) is "a_i = g_i a_(i-1) ;"
    with a_0 = 1. Returns tokens (B, 5n + 3m), labels (B, m) as PLUS/MINUS, and the depth of each query."""
    ar = np.arange(B)[:, None]
    var = np.argsort(rng.random((B, LET)), axis=1)[:, :n]           # chain order, all different
    y = np.where(rng.random((B, n)) < 0.5, 1, -1)                   # value of each variable
    g = y * np.concatenate([np.ones((B, 1), np.int64), y[:, :-1]], axis=1)
    operand = np.concatenate([np.full((B, 1), ONE), var[:, :-1]], axis=1)
    order = np.argsort(rng.random((B, n)), axis=1)                   # clause order in the sentence
    tok = np.zeros((B, 5 * n + 3 * m), np.int64)
    tok[:, 0:5 * n:5] = var[ar, order]
    tok[:, 1:5 * n:5] = EQ
    tok[:, 2:5 * n:5] = np.where(g[ar, order] > 0, PLUS, MINUS)
    tok[:, 3:5 * n:5] = operand[ar, order]
    tok[:, 4:5 * n:5] = NL
    qpos = rng.integers(0, n_tr, size=(B, m))                       # chain positions asked about
    if q0_depth is not None:
        qpos[:, 0] = q0_depth - 1
    tok[:, 5 * n::3] = HASH
    tok[:, 5 * n + 1::3] = np.take_along_axis(var, qpos, axis=1)
    tok[:, 5 * n + 2::3] = COL
    labels = np.where(np.take_along_axis(y, qpos, axis=1) > 0, PLUS, MINUS)
    return tok, labels, qpos + 1


def vt_layout(n, m, name_max, prog_len):
    """Sequence length and query read positions of the vt task."""
    P_len = prog_len if prog_len > 0 else n * (3 + 2 * name_max)
    q_len = 2 + name_max
    return P_len + m * q_len, [P_len + q * q_len + q_len - 1 for q in range(m)]


def make_vt(rng, B, n=16, m=4, D_cap=4, twin_depth=None, q0_depth=None, name_min=1, name_max=3, prog_len=256, n_noise=50):
    """Variable tracking in a less regular format. Statements "VAR <name> = <digit or name> ;" where a name
    has name_min to name_max letters, separated by random noise words so that the program has prog_len
    tokens (prog_len 0: no noise, padded at the end). Queries "# <name, left-padded> :". Chains, depths
    and the two tests are the same as in make_batch. Returns tokens, labels, query depths, and the
    number of distinct chain-end digits (for the guessing level)."""
    plan = None
    if twin_depth is not None:
        D = twin_depth
        order = np.argsort(rng.random((B, n)), axis=1)
        plan = np.zeros((B, n), np.int64)
        np.put_along_axis(plan, order[:, :D], 1, axis=1)
        np.put_along_axis(plan, order[:, D:2 * D], 2, axis=1)
        D_cap = max(D_cap, D)
    lhs, rhs, dep, val, last = gen_lines(rng, B, n, D_cap, plan)
    target = rng.integers(1, D_cap + 1, size=(B, m))
    if q0_depth is not None:
        target[:, 0] = q0_depth
    qline = np.zeros((B, m), np.int64)
    for q in range(m):
        qline[:, q] = (rng.random((B, n)) + 10.0 * (dep == target[:, q:q + 1])).argmax(axis=1)
    if twin_depth is not None:
        qline[:, 0] = np.where(rng.random(B) < 0.5, last[1], last[2])
    T, _ = vt_layout(n, m, name_max, prog_len)
    P_len = T - m * (2 + name_max)
    tok = np.full((B, T), PAD, np.int64)
    lens = rng.integers(name_min, name_max + 1, size=(B, LET))
    letters = rng.integers(0, LET, size=(B, LET, name_max))
    for b in range(B):
        names, seen = [], set()
        for v in range(LET):                                         # one distinct name per variable
            nm = tuple(int(x) for x in letters[b, v, :lens[b, v]])
            while nm in seen:
                nm = tuple(int(x) for x in rng.integers(0, LET, size=lens[b, v]))
            seen.add(nm)
            names.append(list(nm))
        stmts = [[VAR_T] + names[lhs[b, i]] + [EQ] + (names[rhs[b, i]] if rhs[b, i] < LET else [int(rhs[b, i])]) + [NL]
                 for i in range(n)]
        body = sum(len(x) for x in stmts)
        if prog_len > 0:
            assert body <= P_len, "prog_len is too small for the statements"
            cuts = np.sort(rng.integers(0, P_len - body + 1, size=n))
            gaps = np.diff(np.concatenate([[0], cuts]))               # noise before each statement
            noise = (PAD + 1 + rng.integers(0, n_noise, size=P_len - body)).tolist()
            seq, k = [], 0
            for i in range(n):
                seq += noise[k:k + gaps[i]] + stmts[i]
                k += gaps[i]
            seq += noise[k:]
        else:
            seq = [t for x in stmts for t in x]
        tok[b, :len(seq)] = seq
        for q in range(m):
            nm = names[lhs[b, qline[b, q]]]
            o = P_len + q * (2 + name_max)
            tok[b, o] = HASH
            tok[b, o + 1 + name_max - len(nm):o + 1 + name_max] = nm
            tok[b, o + 1 + name_max] = COL
    labels = LET + np.take_along_axis(val, qline, axis=1)
    qdep = np.take_along_axis(dep, qline, axis=1)
    ends = np.sort(np.where(dep == 1, val, -1), axis=1)
    n_end = ((ends[:, 1:] != ends[:, :-1]) & (ends[:, 1:] >= 0)).sum(axis=1) + (ends[:, 0] >= 0)
    return tok, labels, qdep, n_end


def read_positions(n, m, line=4):
    return [line * n + 3 * q + 2 for q in range(m)]


# ----------------------------------------------------------------------------- models
def unit(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def delta_updates(K, Vv, beta, s0):
    """Delta rule S_t = S_{t-1} + u_t k_t^T with u_t = beta_t (v_t - S_{t-1} k_t), S_0 = s0 * I.
    Since S_{t-1} k_t = s0 k_t + sum_{s<t} u_s (k_s . k_t), all u_t solve one lower-triangular system,
    so no loop over time is needed. K, Vv: (N, T, d), beta: (N, T). Returns U: (N, T, d)."""
    T = K.shape[1]
    L = torch.tril(K @ K.transpose(1, 2), diagonal=-1)
    A = torch.eye(T, device=K.device) + beta[:, :, None] * L
    return torch.linalg.solve_triangular(A, beta[:, :, None] * (Vv - s0 * K), upper=False, unitriangular=True)


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

    def __init__(self, n, m, dm=64, d=64, w=4, read="square", gate_ctx=False, hard_taps=0, tap_attn=0, line=4, signed=False, pos=None, pre=0, T=None, pre_kind="tf"):
        super().__init__()
        self.n, self.m, self.d, self.w, self.read, self.line = n, m, d, w, read, line
        self.pos_list = pos if pos is not None else read_positions(n, m, line)
        self.emb = nn.Embedding(V, dm)
        self.code = nn.Linear(dm, d, bias=False)                     # shared code map
        self.taps = nn.Parameter(0.5 * torch.randn(3, w))            # rows: key, value, query
        self.tap_scale = 1.0                                         # raised during training: each role ends up copying one token
        self.hard_taps = hard_taps
        self.gate_ctx = gate_ctx
        if gate_ctx:                                                 # the write gate sees the last w tokens
            self.gate = nn.Conv1d(dm, 1, w, padding=w - 1)
        else:                                                        # the write gate sees the current token only
            self.gate = nn.Linear(dm, 1)
        self.out = nn.Linear(d, dm, bias=False)
        self.top = MLPHead(dm)
        if hard_taps == 2:                                           # created last, so other settings keep their initial weights
            self.pair = nn.Parameter(0.5 * torch.randn(w, w))
        self.tap_attn = tap_attn
        if tap_attn:                                                 # also created last; replaces the fixed mixing weights
            dk = 16
            self.att_q = nn.Linear(dm, 3 * dk)
            self.att_k = nn.Linear(dm, 3 * dk, bias=False)
            self.att_pos = nn.Parameter(0.5 * torch.randn(tap_attn, 3, dk))
        self.signed = signed
        if signed:                                                   # also created last: a sign in [-1, 1] for each value
            self.sign = nn.Conv1d(dm, 1, w, padding=w - 1)
        self.pre_n, self.pre_kind = pre, pre_kind
        if pre and pre_kind == "tf":                                 # also created last: standard causal layers before the
            T = T or line * n + 3 * m                                # fast-weight layer, so that a name of several tokens
            self.pre_pos = nn.Embedding(T, dm)                       # can be gathered into one vector
            layer = nn.TransformerEncoderLayer(dm, 4, 4 * dm, dropout=0.0, batch_first=True, norm_first=True, activation="gelu")
            self.pre = nn.TransformerEncoder(layer, pre, enable_nested_tensor=False)
        elif pre and pre_kind == "delta":                            # DeltaNet blocks instead
            self.pre = nn.ModuleList(DeltaBlock(dm, 4) for _ in range(pre))
        elif pre and pre_kind == "conv":                             # causal convolutions over the last 4 tokens, with an MLP
            self.pre = nn.ModuleList(nn.ModuleDict(dict(ln=nn.LayerNorm(dm), conv=CausalDW(dm, 4), mix=nn.Linear(dm, dm),
                                                        mlp=nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, 4 * dm), nn.GELU(),
                                                                          nn.Linear(4 * dm, dm)))) for _ in range(pre))

    def attn_roles(self, e, c):
        """Each role mixes the codes of the last tap_attn tokens with attention weights. The score
        of offset o at position t compares the current token with token t - o plus an offset
        embedding, so the choice can depend on content as well as on distance."""
        W, dk = self.tap_attn, self.att_pos.shape[-1]
        B, T, _ = c.shape
        window = lambda x: torch.stack(
            [torch.cat([x.new_zeros(B, o, x.shape[-1]), x[:, :T - o]], dim=1) for o in range(W)], dim=2)  # B x T x W x ch
        qa = self.att_q(e).view(B, T, 3, dk)
        ka = self.att_k(window(e)).view(B, T, W, 3, dk) + self.att_pos
        s = torch.einsum("btrk,btwrk->rbtw", qa, ka) / math.sqrt(dk) * self.tap_scale
        reach = torch.arange(T, device=e.device)[:, None] >= torch.arange(W, device=e.device)[None, :]
        p = torch.softmax(s.masked_fill(~reach, -1e9), dim=-1)      # 3 x B x T x W
        self.last_att = p.detach()
        mix = torch.einsum("rbtw,btwd->rbtd", p, window(c))
        return unit(mix[0]), unit(mix[1]), unit(mix[2])

    def att_summary(self):
        """Mean attention over offsets 0..W-1 for key, value and read vector, at the newlines and at
        the ':' of the queries, from the last forward pass."""
        p, n = self.last_att, self.n
        rows = lambda x: [[round(v, 2) for v in row] for row in x.tolist()]
        return {"newline": rows(p[:, :, 3:4 * n:4].mean(dim=(1, 2))), "colon": rows(p[:, :, 4 * n + 2::3].mean(dim=(1, 2)))}

    def roles(self, e):
        c = self.code(e)                                             # B x T x d
        if self.tap_attn:
            return self.attn_roles(e, c)
        B, T, d = c.shape
        pad = torch.cat([c.new_zeros(B, self.w - 1, d), c], dim=1)
        win = torch.stack([pad[:, self.w - 1 - o:self.w - 1 - o + T] for o in range(self.w)], dim=2)  # offset o
        p = torch.softmax(self.taps * self.tap_scale, dim=1)
        if self.hard_taps == 1:                                      # each role copies exactly one of the last w tokens;
            hard = F.one_hot(p.argmax(dim=1), self.w).float()        # the gradient is taken as if the choice were soft
            p = hard + p - p.detach()
        elif self.hard_taps == 2:
            # as above, but key and value are chosen as a pair and may not be the same token
            # (a write whose key equals its value changes nothing when the state starts as the identity)
            logits = self.pair.masked_fill(torch.eye(self.w, dtype=torch.bool, device=c.device), -1e9)
            joint = torch.softmax(logits.flatten(), dim=0).view(self.w, self.w)
            i, j = divmod(int(joint.argmax()), self.w)
            pk, pv = joint.sum(dim=1), joint.sum(dim=0)
            hk = F.one_hot(torch.tensor(i, device=c.device), self.w).float() + pk - pk.detach()
            hv = F.one_hot(torch.tensor(j, device=c.device), self.w).float() + pv - pv.detach()
            hq = F.one_hot(p[2].argmax(), self.w).float() + p[2] - p[2].detach()
            p = torch.stack([hk, hv, hq])
        mix = torch.einsum("rw,btwd->rbtd", p, win)
        return unit(mix[0]), unit(mix[1]), unit(mix[2])              # key, value, query

    def forward(self, tok, r=3):
        e = self.emb(tok)
        if self.pre_n and self.pre_kind == "tf":
            T = tok.shape[1]
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tok.device), 1)
            e = self.pre(e + self.pre_pos(torch.arange(T, device=tok.device)), mask=mask)
        elif self.pre_n and self.pre_kind == "delta":
            for blk in self.pre:
                e = blk(e)
        elif self.pre_n:
            for blk in self.pre:
                e = e + blk["mix"](F.gelu(blk["conv"](blk["ln"](e))))
                e = e + blk["mlp"](e)
        k, v, q = self.roles(e)
        if self.gate_ctx:
            beta = torch.sigmoid(self.gate(e.transpose(1, 2))[:, 0, :e.shape[1]])
        else:
            beta = torch.sigmoid(self.gate(e)).squeeze(-1)
        if self.signed:
            v = v * torch.tanh(self.sign(e.transpose(1, 2))[:, 0, :e.shape[1]])[:, :, None]
        U = delta_updates(k, v, beta, 1.0)                           # the state starts as the identity
        pos = self.pos_list
        ys = [self.read_state(U[:, :p + 1], k[:, :p + 1], q[:, p], r) for p in pos]
        return self.top(e[:, pos] + self.out(torch.stack(ys, dim=1)))

    def read_state(self, U, K, q, r):
        """Reads the state S = I + U^T K (everything written so far) with the query q."""
        apply = lambda y: y + torch.einsum("btd,bt->bd", U, torch.einsum("btd,bd->bt", K, y))   # S y
        if self.read == "once":
            return unit(apply(q))
        if self.read == "iter":
            y = q
            for _ in range(2 ** r):
                y = unit(apply(y))
            return y
        eye = torch.eye(self.d, device=q.device)
        M = eye + U.transpose(1, 2) @ K
        rescale = lambda X: X / (X.flatten(1).norm(dim=1)[:, None, None] / math.sqrt(self.d) + 1e-6)
        if self.read == "sum":                                       # P = I + S + S^2 + ... + S^(2^r - 1), by r squarings
            Pm = eye.expand_as(M)
            for _ in range(r):
                Pm = Pm + M @ Pm
                M = rescale(M @ M)
            return unit(torch.einsum("bij,bj->bi", Pm, q))
        for _ in range(r):                                           # M = S^(2^r), rescaled to stay bounded
            M = rescale(M @ M)
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
        split = lambda z: z.view(B, T, self.H, self.dh).transpose(1, 2).reshape(B * self.H, T, self.dh)
        q = unit(split(self.cq(self.q(x))))
        k = unit(split(self.ck(self.k(x))))
        v = split(self.cv(self.v(x)))
        beta = torch.sigmoid(self.b(x)).transpose(1, 2).reshape(B * self.H, T)
        U = delta_updates(k, v, beta, 0.0)
        o = torch.tril(q @ k.transpose(1, 2)) @ U                    # o_t = S_t q_t, the write at t included
        o = o.view(B, self.H, T, self.dh).transpose(1, 2).reshape(B, T, dm)
        h = h + self.o(o)
        return h + self.mlp(self.ln2(h))


class Delta(nn.Module):
    def __init__(self, n, m, dm=64, L=2, H=4, loops=0, line=4, pos=None):
        super().__init__()
        self.n, self.m, self.loops, self.line = n, m, loops, line
        self.pos_list = pos if pos is not None else read_positions(n, m, line)
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
        return self.head(self.ln(h[:, self.pos_list]))


class TF(nn.Module):
    def __init__(self, n, m, dm=64, L=4, H=4, loops=0, line=4, pos=None, T=None):
        super().__init__()
        self.n, self.m, self.loops, self.line = n, m, loops, line
        self.pos_list = pos if pos is not None else read_positions(n, m, line)
        T = T or line * n + 3 * m
        self.emb = nn.Embedding(V, dm)
        self.pos = nn.Embedding(T, dm)
        layer = nn.TransformerEncoderLayer(dm, H, 4 * dm, dropout=0.0, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, 1 if loops else L, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(dm)
        self.head = nn.Linear(dm, V)

    def forward(self, tok, r=None, loops=None):
        T = tok.shape[1]
        x = self.emb(tok) + self.pos(torch.arange(T, device=tok.device))
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tok.device), 1)
        if self.loops:
            for _ in range(loops or self.loops):
                x = self.enc.layers[0](x, src_mask=mask)
            h = self.ln(x)
        else:
            h = self.ln(self.enc(x, mask=mask))
        return self.head(h[:, self.pos_list])


class LSTMModel(nn.Module):
    def __init__(self, n, m, dm=64, hid=128, L=2, line=4, pos=None):
        super().__init__()
        self.n, self.m, self.line = n, m, line
        self.pos_list = pos if pos is not None else read_positions(n, m, line)
        self.emb = nn.Embedding(V, dm)
        self.rnn = nn.LSTM(dm, hid, L, batch_first=True)
        self.head = nn.Linear(hid, V)

    def forward(self, tok, r=None, loops=None):
        h, _ = self.rnn(self.emb(tok))
        return self.head(h[:, self.pos_list])


# ----------------------------------------------------------------------------- train / eval
def evaluate_lego(model, args, rng, dev, kw, n_eval):
    """Accuracy at every chain position. The first n_tr positions are the training distribution."""
    acc = {}
    for D in range(1, args.n + 1):
        tok, lab, _ = make_lego(rng, n_eval, args.n, args.m, args.n_tr, q0_depth=D)
        pred = model(torch.from_numpy(tok).to(dev), **kw)[:, 0].argmax(-1).cpu().numpy()
        acc[D] = round(float((pred == lab[:, 0]).mean()), 4)
    return {D: acc[D] for D in range(1, args.n_tr + 1)}, acc, {D: 0.5 for D in acc}


def evaluate(model, args, rng, dev, r_eval, loops_eval=None, n_eval=2000):
    """In-distribution accuracy by query depth, and accuracy on two competing chains of depth D."""
    model.eval()
    ind, deep, guess = {}, {}, {}
    kw = dict(r=r_eval) if loops_eval is None else dict(r=r_eval, loops=loops_eval)
    if getattr(args, "task", "bind") == "lego":
        with torch.no_grad():
            out = evaluate_lego(model, args, rng, dev, kw, n_eval)
        model.train()
        return out
    mk = make_batch
    if getattr(args, "task", "bind") == "vt":
        mk = lambda *a, **k: make_vt(*a, name_min=args.name_min, name_max=args.name_max, prog_len=args.prog_len,
                                     n_noise=args.n_noise, **k)
    with torch.no_grad():
        for D in range(1, args.D_train + 1):
            tok, lab, qd, _ = mk(rng, n_eval, args.n, args.m, args.D_train, q0_depth=D)
            pred = model(torch.from_numpy(tok).to(dev), **kw)[:, 0].argmax(-1).cpu().numpy()
            ok = qd[:, 0] == D
            ind[D] = round(float((pred[ok] == lab[ok, 0]).mean()), 4)
        for D in range(1, args.n // 2 + 1):
            tok, lab, qd, n_end = mk(rng, n_eval, args.n, args.m, args.D_train, twin_depth=D)
            pred = model(torch.from_numpy(tok).to(dev), **kw)[:, 0].argmax(-1).cpu().numpy()
            deep[D] = round(float((pred == lab[:, 0]).mean()), 4)
            guess[D] = round(float((1.0 / n_end).mean()), 4)
    model.train()
    return ind, deep, guess


def main():
    global V
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tied", choices=["tied", "delta", "tf", "lstm"])
    ap.add_argument("--read", default="square", choices=["square", "sum", "once", "iter"])
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--H", type=int, default=4, help="attention heads (tf) or state heads (delta)")
    ap.add_argument("--loops", type=int, default=0, help="delta and tf: apply one shared block this many times")
    ap.add_argument("--loops_eval", type=int, default=None)
    ap.add_argument("--loops_lo", type=int, default=None,
                    help="delta and tf with --loops: draw the number of applications from [loops_lo, loops_hi] at every training step")
    ap.add_argument("--loops_hi", type=int, default=None)
    ap.add_argument("--hid", type=int, default=128, help="lstm only: hidden size")
    ap.add_argument("--task", default="bind", choices=["bind", "lego", "vt"])
    ap.add_argument("--name_min", type=int, default=1, help="vt only: shortest variable name, in letters")
    ap.add_argument("--name_max", type=int, default=3, help="vt only: longest variable name, in letters")
    ap.add_argument("--prog_len", type=int, default=256, help="vt only: program length in tokens, filled with noise words; 0 = no noise")
    ap.add_argument("--n_noise", type=int, default=50, help="vt only: number of distinct noise words")
    ap.add_argument("--n_tr", type=int, default=6, help="lego only: training queries ask for the first n_tr chain positions")
    ap.add_argument("--signed", type=int, default=0, help="tied only: 1 = multiply each value by a learned sign")
    ap.add_argument("--pre", type=int, default=0, help="tied only: this many standard causal layers before the fast-weight layer")
    ap.add_argument("--pre_kind", default="tf", choices=["tf", "delta", "conv"], help="tied only: kind of the layers given by --pre")
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
    ap.add_argument("--tap_sharp", type=float, default=1.0,
                    help="tied only: final sharpness of the role selection, reached after --tap_ramp of training")
    ap.add_argument("--tap_ramp", type=float, default=0.5)
    ap.add_argument("--hard_taps", type=int, default=0,
                    help="tied only: 1 = every role copies exactly one token; 2 = same, key and value chosen as a pair of different tokens")
    ap.add_argument("--gate_ctx", type=int, default=0, help="tied only: 1 = the write gate sees the last 4 tokens")
    ap.add_argument("--tap_entropy", type=float, default=0.0,
                    help="tied only: add this times the mean entropy of the three mixing weight rows to the loss")
    ap.add_argument("--tap_attn", type=int, default=0,
                    help="tied only: choose each role's token by attention over this many recent tokens instead of fixed weights")
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
    line = 5 if args.task == "lego" else 4
    pos, T = None, None
    if args.task == "vt":
        V = PAD + 1 + args.n_noise
        T, pos = vt_layout(args.n, args.m, args.name_max, args.prog_len)
    if args.model == "tied":
        model = Tied(args.n, args.m, args.dm, args.d, read=args.read, gate_ctx=bool(args.gate_ctx), hard_taps=args.hard_taps,
                     tap_attn=args.tap_attn, line=line, signed=bool(args.signed), pos=pos, pre=args.pre, T=T, pre_kind=args.pre_kind)
        lr = args.lr or 3e-3
    elif args.model == "delta":
        model = Delta(args.n, args.m, args.dm, args.L, H=args.H, loops=args.loops, line=line, pos=pos)
        lr = args.lr or 2e-3
    elif args.model == "tf":
        model = TF(args.n, args.m, args.dm, args.L, H=args.H, loops=args.loops, line=line, pos=pos, T=T)
        lr = args.lr or 1e-3
    else:
        model = LSTMModel(args.n, args.m, args.dm, args.hid, args.L, line=line, pos=pos)
        lr = args.lr or 2e-3
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
        if args.task == "lego":
            tok, lab, _ = make_lego(rng, args.bs, args.n, args.m, args.n_tr)
        elif args.task == "vt":
            tok, lab, _, _ = make_vt(rng, args.bs, args.n, args.m, args.D_train, name_min=args.name_min, name_max=args.name_max,
                                     prog_len=args.prog_len, n_noise=args.n_noise)
        else:
            tok, lab, _, _ = make_batch(rng, args.bs, args.n, args.m, args.D_train)
        r = int(rng.integers(args.r_lo, args.r_hi + 1))
        if args.model == "tied":
            model.tap_scale = 1.0 + (args.tap_sharp - 1.0) * min(1.0, step / (args.tap_ramp * args.steps))
        kw = {}
        if args.loops and args.loops_lo is not None:
            kw["loops"] = int(rng.integers(args.loops_lo, args.loops_hi + 1))
        logits = model(torch.from_numpy(tok).to(dev), r=r, **kw)
        loss = F.cross_entropy(logits.reshape(-1, V), torch.from_numpy(lab).to(dev).reshape(-1))
        if args.model == "tied" and args.tap_entropy:                  # push each role towards a single token
            logp = torch.log_softmax(model.taps * model.tap_scale, dim=1)
            loss = loss + args.tap_entropy * -(logp.exp() * logp).sum(dim=1).mean()
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
            if args.model == "tied" and args.tap_attn:
                taps = model.att_summary()
            elif args.model == "tied":
                taps = [[round(x, 2) for x in row] for row in torch.softmax(model.taps * model.tap_scale, dim=1).tolist()]
            else:
                taps = None
            curve.append((step, ind, deep, taps))
            print("   eval", step, "in-dist", list(ind.values()), "deep", list(deep.values()), "taps", taps, flush=True)
        if time.time() - t0 > args.max_seconds:
            print("time cap reached at step", step, flush=True)
            break
    train_seconds = round(time.time() - t0, 1)
    if args.save:
        torch.save(model.state_dict(), args.save)
    ind, deep, guess = evaluate(model, args, np.random.default_rng(10_000 + args.seed), dev, args.r_eval, args.loops_eval)
    summary = dict(model=args.model, read=args.read if args.model == "tied" else None,
                   L=args.L if args.model != "tied" else 1, H=args.H, loops=args.loops, loops_eval=args.loops_eval,
                   loops_train=[args.loops_lo, args.loops_hi] if args.loops_lo is not None else None,
                   hid=args.hid if args.model == "lstm" else None,
                   seed=args.seed, params=n_params, steps=step, bs=args.bs, lr=lr, train_seconds=train_seconds,
                   n=args.n, m=args.m, D_train=args.D_train, dm=args.dm, d=args.d,
                   r_train=[args.r_lo, args.r_hi], r_eval=args.r_eval, tap_sharp=args.tap_sharp, hard_taps=args.hard_taps, gate_ctx=args.gate_ctx, tag=args.tag,
                   tap_entropy=args.tap_entropy, tap_attn=args.tap_attn, task=args.task, n_tr=args.n_tr if args.task == "lego" else None,
                   vt=dict(name_min=args.name_min, name_max=args.name_max, prog_len=args.prog_len, n_noise=args.n_noise) if args.task == "vt" else None,
                   signed=args.signed, pre=args.pre, pre_kind=args.pre_kind if args.pre else None,
                   acc_in_dist=ind, acc_deep=deep, guess_deep=guess, log=log, curve=curve)
    if args.model == "tied":
        with torch.no_grad():
            if args.tap_attn:
                summary["taps"] = model.att_summary()                  # from the last batch of the final evaluation
            else:
                summary["taps"] = [[round(x, 3) for x in row] for row in torch.softmax(model.taps * model.tap_scale, dim=1).tolist()]
            if args.hard_taps == 2:
                lg = model.pair.masked_fill(torch.eye(model.w, dtype=torch.bool, device=model.pair.device), -1e9)
                i, j = divmod(int(lg.argmax()), model.w)
                summary["chosen_offsets"] = dict(key=i, value=j, query=int(model.taps[2].argmax()))
            if args.read in ("square", "sum"):                         # how many squarings does each depth need?
                by_r = {}
                for rr in range(0, 9):
                    _, dd, _ = evaluate(model, args, np.random.default_rng(20_000 + args.seed), dev, rr, n_eval=1000)
                    by_r[rr] = dd
                summary["acc_deep_by_squarings"] = by_r
    if args.loops:                                                     # how many applications does each depth need?
        by_loops = {}
        for ll in (1, 2, 4, 6, 8, 12, 16):
            _, dd, _ = evaluate(model, args, np.random.default_rng(20_000 + args.seed), dev, args.r_eval, ll, n_eval=1000)
            by_loops[ll] = dd
        summary["acc_deep_by_loops"] = by_loops
    print("RESULT", json.dumps(summary))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
