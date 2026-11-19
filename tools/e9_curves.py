#!/usr/bin/env python3
"""Experiment 9: when does each run leave the trick, and where are its codes at that time?

For every finished run with tag e9 (results/res_server.jsonl): the first evaluation step at which depth 1 of
the training distribution reaches 90% (the trick stays near 30%), and two numbers about the codes at step
2,000 and at the end: the share of a name code inside the span of the digit codes, and the number of
directions the name codes use. Evaluations during training use 500 programs per depth.

usage: .venv/bin/python tools/e9_curves.py > results/e9_curves.txt      (CPU only, reads results only)
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from make_status import E9_ARMS, e9_arm, ok  # noqa: E402


def main():
    g = defaultdict(list)
    for line in open(os.path.join(ROOT, "results/res_server.jsonl")):
        r = json.loads(line)
        if r["tag"] == "e9":
            g[e9_arm(r)].append(r)
    for arm in E9_ARMS:
        rs = sorted(g.get(arm, []), key=lambda r: r["seed"])
        if not rs:
            continue
        print(f"== {arm}")
        early_hi, early_lo = [], []
        for r in rs:
            left = next((step for step, ind, _, _ in r["curve"] if ind["1"] >= 0.90), None)
            gc = {step: x for step, x in r["geom_curve"]}
            first, last = gc[min(gc)], r["geom"]
            d1 = r["acc_in_dist"]["1"]
            (early_hi if d1 >= 0.90 else early_lo).append(first["name_in_digit_span"])
            print(f"  seed {r['seed']}: success {'yes' if ok(r) else 'no '} | depth 1 at the end {100 * d1:.0f}% | "
                  f"depth 1 first at 90%: step {left if left else '-'} | "
                  f"name code in digit span at step {min(gc)}: {first['name_in_digit_span']:.2f}, at the end: {last['name_in_digit_span']:.2f} | "
                  f"directions used at step {min(gc)}: {first['name_dims']:.0f}, at the end: {last['name_dims']:.0f}")
        med = lambda xs: f"{float(np.median(xs)):.2f} (n={len(xs)})" if xs else "none"
        print(f"  name code in digit span at step 2,000, median: runs that end with depth 1 >= 90%: {med(early_hi)}; other runs: {med(early_lo)}")


if __name__ == "__main__":
    main()
