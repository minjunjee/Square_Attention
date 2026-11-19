#!/usr/bin/env python3
"""Re-tests saved models of experiment 9 with fewer squarings at test time.

The verdict of experiment 9 uses r = 6 squarings at test time, one more than the training range (2 to 5).
This script loads every saved model of the given tag and measures the training-distribution accuracy by
depth at r = 4, 5 and 6, on the same 2,000 programs per depth as the final evaluation (so the r = 6 column
must reproduce the recorded numbers). A diagnostic only; the verdict rule stays at r = 6.

usage: CUDA_VISIBLE_DEVICES=<GPU UUID or ''> .venv/bin/python tools/eval_by_r.py [--tag e9] > results/e9_by_r.txt
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import gpu_guard  # noqa: E402
import poc_fair as P  # noqa: E402

V0 = P.V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="e9")
    ap.add_argument("--rs", default="4,5,6")
    a = ap.parse_args()
    rs_test = [int(x) for x in a.rs.split(",")]
    gpu_guard.check()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cmds = {}
    for line in open(os.path.join(ROOT, f"jobs/{a.tag}.txt")):
        if "--save" in line:
            w = line.split()
            cmds[w[w.index("--save") + 1]] = w
    rows = {}
    for line in open(os.path.join(ROOT, "results/res_server.jsonl")):
        r = json.loads(line)
        if r["tag"] == a.tag:
            rows[(json.dumps([r[k] for k in ("lr", "d", "dm", "m", "code_init", "freeze_code", "name_curr", "name_zipf", "r_train")]), r["seed"])] = r
    out = defaultdict(list)
    for path, w in sorted(cmds.items()):
        opt = lambda k, default=None: w[w.index(k) + 1] if k in w else default
        cfg = dict(lr=float(opt("--lr")), d=int(opt("--d", 64)), dm=int(opt("--dm", 64)), m=int(opt("--m", 4)), code_init=opt("--code_init", "default"),
                   freeze_code=int(opt("--freeze_code", 0)), name_curr=opt("--name_curr"), name_zipf=float(opt("--name_zipf", 0.0)),
                   r_train=[2, int(opt("--r_hi", 4))])
        seed, n, n_names = int(opt("--seed")), int(opt("--n")), int(opt("--n_names"))
        rec = rows[(json.dumps([cfg[k] for k in ("lr", "d", "dm", "m", "code_init", "freeze_code", "name_curr", "name_zipf", "r_train")]), seed)]
        P.NAMES = np.concatenate([np.arange(P.LET), np.arange(V0, V0 + n_names - P.LET)])
        P.V = V0 + n_names - P.LET
        model = P.Tied(n, cfg["m"], cfg["dm"], cfg["d"], read="square", gate_ctx=True, tap_attn=8, line=4)
        model.load_state_dict(torch.load(os.path.join(ROOT, path), map_location="cpu"))
        model.to(dev)
        model.tap_scale = float(opt("--tap_sharp"))
        args = SimpleNamespace(task="bind", n=n, m=cfg["m"], D_train=4, deep_max=None, eval_chunk=None)
        name = os.path.basename(path)[len(a.tag) + 1:-3].rsplit("_s", 1)[0]
        res = {}
        for r in rs_test:
            ind, deep, _ = P.evaluate(model, args, np.random.default_rng(10_000 + seed), dev, r)
            res[r] = (ind, deep)
        same = all(abs(res[6][0][D] - rec["acc_in_dist"][str(D)]) < 1e-9 for D in range(1, 5)) if 6 in res else None
        out[name].append((seed, res, same))
        print(f"{name} seed {seed}: " + " | ".join(
            f"r={r}: in-dist {[round(100 * v, 1) for v in res[r][0].values()]} two-chain 5-8 {100 * np.mean([res[r][1][D] for D in (5, 6, 7, 8)]):.1f}"
            for r in rs_test) + f" | r=6 equals the recorded result: {same}", flush=True)
    print()
    for name, xs in out.items():
        line = f"{name}: runs {len(xs)}"
        for r in rs_test:
            okr = [all(v >= 0.99 for v in res[r][0].values()) for _, res, _ in xs]
            d58 = [float(np.mean([res[r][1][D] for D in (5, 6, 7, 8)])) for _, res, _ in xs]
            line += f" | r={r}: all depths >= 99% in {sum(okr)}, two-chain 5-8 >= 90% in {sum(x >= 0.90 for x in d58)}"
        print(line)


if __name__ == "__main__":
    main()
