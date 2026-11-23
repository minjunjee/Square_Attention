#!/usr/bin/env python3
"""Experiment 4: picks the fix for the seed problem (plan section 6, experiment 4).

Every (variant, learning rate) of tag e4 is scored by the number of seeds that reach 99% at every
training depth, then by mean in-distribution accuracy. The two-chain test is not used. The best one
is adopted only if it has more such seeds than the original setting at the e1 learning rate on the
same seeds. Prints the flags of the variant (without the learning rate), or "none".
usage: python3 choose_fix.py results/res_server.jsonl --e1_lr 0.001"""
import argparse
import json
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--e1_lr", type=float, required=True)
args = ap.parse_args()
runs = [json.loads(line) for line in open(args.path)]
ok = lambda r: all(v >= 0.99 for v in r["acc_in_dist"].values())
cand = defaultdict(list)
for r in runs:
    if r["tag"] == "e4":
        if r.get("hard_taps") == 2:
            flags = "--hard_taps 2"
        elif r.get("tap_entropy"):
            flags = f"--tap_entropy {r['tap_entropy']:g}"
        else:
            flags = f"--tap_attn {r['tap_attn']}"
        cand[(flags, r["lr"])].append(r)
seeds = sorted({r["seed"] for rs in cand.values() for r in rs})
base = [r for r in runs if r["tag"] == "e1" and r["model"] == "tied" and r["read"] == "square" and r["lr"] == args.e1_lr
        and r["seed"] in seeds and not r.get("hard_taps") and not r.get("tap_entropy") and not r.get("tap_attn")]
score = lambda rs: (sum(ok(r) for r in rs), float(np.mean([np.mean(list(r["acc_in_dist"].values())) for r in rs])))
for (flags, lr), rs in sorted(cand.items()):
    print(f"# {flags} lr {lr:g}: {score(rs)[0]}/{len(rs)} seeds reach 99%, in-dist mean {100 * score(rs)[1]:.1f}")
print(f"# original setting, lr {args.e1_lr:g}, seeds {seeds}: {score(base)[0]}/{len(base)} seeds reach 99%, in-dist mean {100 * score(base)[1]:.1f}")
best = max(cand, key=lambda c: score(cand[c]))
print(best[0] if score(cand[best])[0] > score(base)[0] else "none")
