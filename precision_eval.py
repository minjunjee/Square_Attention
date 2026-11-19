#!/usr/bin/env python3
"""Does the squared read survive low-precision matrix products? Tensor cores need TF32 or bf16.

usage: CUDA_VISIBLE_DEVICES=<allowed GPU UUID> python3 precision_eval.py ckpt/e1fix_square_lr1e-3_s*.pt --out results/precision.json

Evaluates saved proposed-layer models (attention choice, as in e1fix) on the two-chain test at r = 6
with every matrix product in fp32, in TF32, and under bf16 autocast.
"""
import argparse
import glob
import json
import re
import types

import numpy as np
import torch

import gpu_guard
import poc_fair as P


def two_chain(model, seed, dev, n_eval=1000):
    args = types.SimpleNamespace(D_train=4, n=16, m=4, task="bind")
    _, deep, _ = P.evaluate(model, args, np.random.default_rng(30_000 + seed), dev, 6, n_eval=n_eval)
    return deep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    gpu_guard.check()
    dev = "cuda"
    out = {}
    for path in sorted(p for g in args.paths for p in glob.glob(g)):
        seed = int(re.search(r"_s(\d+)\.pt", path).group(1))
        m = P.Tied(16, 4, read="square", gate_ctx=True, tap_attn=8).to(dev)
        m.load_state_dict(torch.load(path, map_location=dev))
        m.tap_scale = 10.0
        res = {}
        for mode in ("fp32", "tf32", "bf16"):
            torch.backends.cuda.matmul.allow_tf32 = mode == "tf32"
            torch.backends.cudnn.allow_tf32 = mode == "tf32"
            if mode == "bf16":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    res[mode] = two_chain(m, seed, dev)
            else:
                res[mode] = two_chain(m, seed, dev)
        torch.backends.cuda.matmul.allow_tf32 = False
        out[path] = res
        print(path, {k: round(float(np.mean([v[str(d)] if str(d) in v else v[d] for d in range(2, 9)])), 4) for k, v in res.items()},
              flush=True)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
