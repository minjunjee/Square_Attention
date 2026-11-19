#!/usr/bin/env python3
"""Experiment 2 against the benefit criterion of plan section 7.

usage: python3 judge_e2.py results/res_server.jsonl --tags e2 e2fix [--out results/judge_e2.md]

For every setting at its chosen learning rate (mean in-distribution accuracy, plan section 5 rule 3):
  - seeds that reach 99% at every training depth (rule 5: the others count as undertrained),
  - two-chain accuracy at depths 5-8, averaged over those seeds and over all seeds,
  - the first evaluation step at which a seed reaches 99% at every training depth, from the
    curve recorded every 2,000 steps on 500 programs per depth (a noisier test than the final one).
The benefit criterion compares the proposed layer with the standard models that reach 99%.
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from summarize import name, setting


def ok(r):
    return all(v >= 0.99 for v in r["acc_in_dist"].values())


def deep(r, lo=5, hi=8):
    return float(np.mean([r["acc_deep"][str(d)] for d in range(lo, hi + 1)]))


def first_ok_step(r):
    for step, ind, _, _ in r.get("curve", []):
        if all(v >= 0.99 for v in ind.values()):
            return step
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--tags", nargs="+", default=["e2"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs = [r for r in map(json.loads, open(args.path)) if r.get("tag") in args.tags]
    groups = defaultdict(lambda: defaultdict(list))
    for r in runs:
        groups[(r["tag"],) + setting(r)][r["lr"]].append(r)
    rows = []
    for key, by_lr in groups.items():
        lr = max(by_lr, key=lambda x: np.mean([np.mean(list(r["acc_in_dist"].values())) for r in by_lr[x]]))
        rs = by_lr[lr]
        good = [r for r in rs if ok(r)]
        steps = [s for s in (first_ok_step(r) for r in good) if s is not None]
        rows.append(dict(tag=key[0], name=name(rs[0]), lr=lr, n=len(rs), n_ok=len(good), params=rs[0]["params"],
                         deep_ok=np.mean([deep(r) for r in good]) if good else None,
                         deep_all=np.mean([deep(r) for r in rs]),
                         step99=int(np.median(steps)) if steps else None, proposed=rs[0]["model"] == "tied"))
    rows.sort(key=lambda x: (not x["proposed"], x["name"]))
    pct = lambda x: "" if x is None else f"{100 * x:.1f}"
    out = ["| tag | setting | params | lr | seeds | seeds at 99% | two-chain 5-8, seeds at 99% | two-chain 5-8, all seeds | median first step at 99% |",
           "|---|---|---|---|---|---|---|---|---|"]
    for x in rows:
        out.append(f"| {x['tag']} | {x['name']} | {x['params']:,} | {x['lr']:g} | {x['n']} | {x['n_ok']} | {pct(x['deep_ok'])} | "
                   f"{pct(x['deep_all'])} | {x['step99'] if x['step99'] is not None else ''} |")
    out.append("")
    std = [x for x in rows if not x["proposed"] and x["n_ok"] > 0]
    for p in [x for x in rows if x["proposed"]]:
        if p["deep_ok"] is None:
            out.append(f"{p['tag']} {p['name']}: no seed reaches 99%, no comparison.")
            continue
        if not std:
            out.append(f"{p['tag']} {p['name']}: no standard model reaches 99% yet, no comparison.")
            continue
        best = max(std, key=lambda x: x["deep_ok"])
        gap = 100 * (p["deep_ok"] - best["deep_ok"])
        out.append(f"{p['tag']} {p['name']}: two-chain 5-8 {pct(p['deep_ok'])}% against the best standard model that reaches 99%, "
                   f"{best['name']} at {pct(best['deep_ok'])}%: difference {gap:+.1f} points "
                   f"({'at least' if gap >= 20 else 'less than'} the 20 points of the first benefit condition).")
    text = "\n".join(out) + "\n"
    if args.out:
        open(args.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
