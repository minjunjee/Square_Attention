#!/usr/bin/env python3
"""Experiment 10: which kind of solution did each run learn?

Uses the two-chain accuracy by number of test squarings r that every run records (1,000 programs per cell).
  trick       depth 1 of the training distribution below 40% (answers the longest chain's end digit)
  one step    two-chain depth 4 at r = 2 is at least 90%: S^4 already follows 4 assignments,
              so one multiplication by S moves one assignment
  slow        depth 4 reaches 90% only with more squarings (r = 3 to 6)
  partial     none of the above
For each run the script also prints the best two-chain accuracy over r = 2..8 at depths 8, 16 and 32,
to separate "r = 6 was not enough squarings" from "the chain cannot be followed at any r".

usage: .venv/bin/python tools/e10_solutions.py > results/e10_solutions.txt     (reads results only)
"""
import json
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def kind(r):
    by = r["acc_deep_by_squarings"]
    if r["acc_in_dist"]["1"] < 0.40:
        return "trick"
    if by["2"]["4"] >= 0.90:
        return "one step"
    if max(by[str(k)]["4"] for k in (3, 4, 5, 6)) >= 0.90:
        return "slow"
    return "partial"


def main():
    g = defaultdict(list)
    for line in open(os.path.join(ROOT, "results/res_server.jsonl")):
        r = json.loads(line)
        if r["model"] != "tied":
            continue
        if r["tag"] == "e10" or (r["tag"] == "e9b" and r["name_zipf"] and r["r_train"][1] == 6):
            g[(r["d"], r["n"])].append(r)
    print("| state | lines | runs | stuck on the trick | depth 8 >= 90% at r=6 | depth 8 >= 90% at the best r (2..8) | depth 16 >= 90% at the best r |")
    print("|---|---|---|---|---|---|---|")
    for (d, n), rs in sorted(g.items()):
        best = lambda r, D: max(r["acc_deep_by_squarings"][str(k)][str(D)] for k in range(2, 9))
        has16 = "16" in rs[0]["acc_deep_by_squarings"]["6"]
        print(f"| {d}x{d} | {n} | {len(rs)} | {sum(kind(r) == 'trick' for r in rs)} | {sum(r['acc_deep_by_squarings']['6']['8'] >= 0.90 for r in rs)} | "
              f"{sum(best(r, 8) >= 0.90 for r in rs)} | {sum(best(r, 16) >= 0.90 for r in rs) if has16 else ''} |")
    print()
    for (d, n), rs in sorted(g.items()):
        rs = sorted(rs, key=lambda r: r["seed"])
        c = Counter(kind(r) for r in rs)
        print(f"== state {d}x{d}, {n} lines: " + ", ".join(f"{k} {c[k]}" for k in ("one step", "slow", "partial", "trick")))
        for r in rs:
            by = r["acc_deep_by_squarings"]
            best = lambda D: max(by[str(k)][str(D)] for k in range(2, 9)) if str(D) in by["6"] else None
            arg = lambda D: max(range(2, 9), key=lambda k: by[str(k)][str(D)])
            cells = []
            for D in (4, 8, 16, 32):
                if str(D) in by["6"]:
                    cells.append(f"depth {D}: r=6 {100 * by['6'][str(D)]:.0f}%, best {100 * best(D):.0f}% at r={arg(D)}")
            ind = ", ".join(f"{100 * v:.0f}" for v in r["acc_in_dist"].values())
            print(f"  seed {r['seed']}: {kind(r):8s} | training distribution {ind} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
