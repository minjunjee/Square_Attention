#!/usr/bin/env python3
"""Experiment 7: time per training step and peak GPU memory, by model and program length.

usage: CUDA_VISIBLE_DEVICES=<allowed GPU UUID> python3 cost_bench.py --lines 16 64 256 --out results/cost.jsonl

Timing does not depend on the content of the program, so the input is random tokens of the
right length (4 tokens per line and 4 queries). The proposed layer is timed twice: reading only
at the 4 query positions, as in all experiments, and reading at every position, as a language
model would. Numbers are only meaningful when nothing else runs on the GPU.
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

import gpu_guard
import poc_fair as P


class TiedAllPositions(P.Tied):
    """The proposed layer, reading the state at every position: S_t holds the writes up to t."""

    def forward(self, tok, r=3):
        e = self.emb(tok)
        k, v, q = self.roles(e)
        beta = torch.sigmoid(self.gate(e.transpose(1, 2))[:, 0, :e.shape[1]]) if self.gate_ctx \
            else torch.sigmoid(self.gate(e)).squeeze(-1)
        U = P.delta_updates(k, v, beta, 1.0)
        B, T, d = k.shape
        M = torch.eye(d, device=tok.device) + torch.cumsum(U[:, :, :, None] * k[:, :, None, :], dim=1)   # B x T x d x d
        rescale = lambda X: X / (X.flatten(2).norm(dim=2)[:, :, None, None] / math.sqrt(d) + 1e-6)
        if self.read == "once":
            y = torch.einsum("btij,btj->bti", M, q)
        else:
            for _ in range(r):
                M = rescale(M @ M)
            y = torch.einsum("btij,btj->bti", M, q)
        y = P.unit(y)
        pos = P.read_positions(self.n, self.m)
        return self.top(e[:, pos] + self.out(y[:, pos]))


def build(spec, n, m):
    kind = spec["kind"]
    if kind == "tied":
        return P.Tied(n, m, read=spec["read"], gate_ctx=True)
    if kind == "tied_all":
        return TiedAllPositions(n, m, read=spec["read"], gate_ctx=True)
    if kind == "tf":
        return P.TF(n, m, spec["dm"], spec["L"], H=spec["H"], loops=spec.get("loops", 0))
    if kind == "delta":
        return P.Delta(n, m, 64, spec["L"], loops=spec.get("loops", 0))
    return P.LSTMModel(n, m, 64, 128, 2)


SPECS = [
    dict(name="tied square, r=5", kind="tied", read="square", r=5),
    dict(name="tied iter, 2^5 reads", kind="tied", read="iter", r=5),
    dict(name="tied once", kind="tied", read="once", r=5),
    dict(name="tied square, r=5, read at every position", kind="tied_all", read="square", r=5),
    dict(name="tied once, read at every position", kind="tied_all", read="once", r=5),
    dict(name="tf L4 d64", kind="tf", L=4, dm=64, H=4),
    dict(name="tf L12 d512 h8", kind="tf", L=12, dm=512, H=8),
    dict(name="tf d64, one block x4", kind="tf", L=1, dm=64, H=4, loops=4),
    dict(name="tf d64, one block x16", kind="tf", L=1, dm=64, H=4, loops=16),
    dict(name="delta L2", kind="delta", L=2),
    dict(name="delta L4", kind="delta", L=4),
    dict(name="delta L8", kind="delta", L=8),
    dict(name="delta, one block x8", kind="delta", L=1, loops=8),
    dict(name="lstm L2", kind="lstm"),
]


def bench(spec, n, m, bs, steps, warm, dev):
    torch.manual_seed(0)
    model = build(spec, n, m).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    T = 4 * n + 3 * m
    rng = np.random.default_rng(0)
    tok = torch.from_numpy(rng.integers(0, P.V, size=(bs, T))).to(dev)
    lab = torch.from_numpy(rng.integers(P.LET, P.LET + P.DIG, size=(bs, m))).to(dev)
    kw = dict(r=spec.get("r", 3))
    if spec.get("loops"):
        kw["loops"] = spec["loops"]
    torch.cuda.reset_peak_memory_stats(dev)
    times = []
    for i in range(warm + steps):
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        loss = F.cross_entropy(model(tok, **kw).reshape(-1, P.V), lab.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        torch.cuda.synchronize(dev)
        if i >= warm:
            times.append(time.perf_counter() - t0)
    return dict(model=spec["name"], lines=n, tokens=T, bs=bs, params=sum(p.numel() for p in model.parameters()),
                ms_per_step=round(1000 * float(np.median(times)), 3), ms_p90=round(1000 * float(np.percentile(times, 90)), 3),
                peak_mem_mb=round(torch.cuda.max_memory_allocated(dev) / 2 ** 20, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=int, nargs="+", default=[16, 64, 256])
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--only", default=None, help="substring of model names to run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    gpu_guard.check()
    assert torch.cuda.is_available(), "cost_bench needs a GPU"
    dev = torch.device("cuda")
    torch.set_num_threads(1)
    for n in args.lines:
        for spec in SPECS:
            if args.only and args.only not in spec["name"]:
                continue
            try:
                res = bench(spec, n, 4, args.bs, args.steps, args.warm, dev)
            except torch.cuda.OutOfMemoryError:
                res = dict(model=spec["name"], lines=n, error="out of memory")
            torch.cuda.empty_cache()
            res["gpu"] = torch.cuda.get_device_name(dev)
            print(json.dumps(res), flush=True)
            if args.out:
                with open(args.out, "a") as f:
                    f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
