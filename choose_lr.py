#!/usr/bin/env python3
"""Prints the learning rate of one setting chosen by mean in-distribution accuracy (plan section 5,
rule 3), and how many of its seeds reach 99% at every training depth.
usage: python3 choose_lr.py results/res_server.jsonl --tag e1 --model tied --read square"""
import argparse
import json
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--tag", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--read", default=None)
ap.add_argument("--hard_taps", type=int, default=0)
args = ap.parse_args()
by_lr = defaultdict(list)
for line in open(args.path):
    r = json.loads(line)
    if r["tag"] == args.tag and r["model"] == args.model and r.get("read") == args.read and r.get("hard_taps", 0) == args.hard_taps \
            and not r.get("tap_entropy") and not r.get("tap_attn"):
        by_lr[r["lr"]].append(r)
score = {lr: np.mean([np.mean(list(r["acc_in_dist"].values())) for r in rs]) for lr, rs in by_lr.items()}
best = max(score, key=score.get)
ok = sum(all(v >= 0.99 for v in r["acc_in_dist"].values()) for r in by_lr[best])
print(f"{best:g} {ok} {len(by_lr[best])}")
