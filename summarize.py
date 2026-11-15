#!/usr/bin/env python3
"""Tables for the server runs (SERVER_POC_PLAN.md section 8).

usage: python3 summarize.py results/res_server.jsonl --tag e1 [--out results/tables_e1.md]

For every model setting and learning rate: one row per seed and a mean row. Columns are the
model, parameter count, learning rate, steps, seconds per step, in-distribution accuracy at
depths 1-4 and two-chain accuracy at depths 1-8. The learning rate of each setting is chosen
by the mean in-distribution accuracy only (plan section 5, rule 3).

Seconds per step come from train_seconds / steps of each run. Runs share a GPU with other
runs, so these are not clean timings; experiment 7 measures them in isolation.
"""
import argparse
import json
from collections import defaultdict

import numpy as np

SETTING_KEYS = ("model", "read", "L", "dm", "H", "hid", "loops", "loops_train", "loops_eval", "hard_taps",
                "gate_ctx", "tap_sharp", "tap_entropy", "tap_attn", "signed", "pre", "task", "n_tr", "vt",
                "r_train", "r_eval", "n", "m", "D_train", "bs")


def name(run):
    m = run["model"]
    if m == "tied":
        s = f"tied/{run['read']}"
        if run.get("hard_taps"):
            s += f"/hard{run['hard_taps']}"
        if run.get("tap_entropy"):
            s += f"/entropy{run['tap_entropy']:g}"
        if run.get("tap_attn"):
            s += f"/attn{run['tap_attn']}"
        if run.get("signed"):
            s += "/signed"
        if run.get("pre"):
            s += f" after {run['pre']} tf layer(s)"
        if run.get("n", 16) != 16 or run.get("D_train", 4) != 4:
            s += f" n{run['n']} Dtrain{run['D_train']}"
        return s + vt_label(run)
    if m == "tf":
        s = f"tf L{run['L']} d{run['dm']} h{run['H']}"
    elif m == "delta":
        s = f"delta L{run['L']} d{run['dm']}"
    else:
        s = f"lstm L{run['L']} h{run.get('hid')}"
    if run.get("loops"):
        s += f" loops{run['loops_train'] or run['loops']}/eval{run['loops_eval']}"
    return s + vt_label(run)


DEFAULTS = dict(hard_taps=0, gate_ctx=0, tap_entropy=0.0, tap_attn=0, signed=0, pre=0, task="bind")   # runs from before an option existed


def vt_label(run):
    v = run.get("vt")
    if not v:
        return ""
    names = "1-token names" if v["name_max"] == 1 else f"{v['name_min']}-{v['name_max']}-token names"
    return f" [{names}, {'noise to ' + str(v['prog_len']) if v['prog_len'] else 'no noise'}]"


def setting(run):
    def value(k):
        v = run.get(k, DEFAULTS.get(k))
        if k == "tap_entropy":
            return float(v or 0)
        if k in ("hard_taps", "gate_ctx", "tap_attn", "signed"):
            return int(v or 0)
        return v
    return tuple(json.dumps(value(k)) for k in SETTING_KEYS) + (run["steps"],)


def fmt(x):
    return f"{100 * x:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = [json.loads(line) for line in open(args.path) if line.strip()]
    runs = [r for r in runs if r.get("tag") == args.tag]
    groups = defaultdict(lambda: defaultdict(list))
    for r in runs:
        groups[setting(r)][r["lr"]].append(r)

    out = [f"# Tables for tag {args.tag}", "",
           f"Source: `{args.path}`, {len(runs)} runs. Accuracies in %. "
           "'ok' marks a seed whose in-distribution accuracy is at least 99% at every depth (plan section 5, rule 5). "
           "s/step is measured while other runs share the GPU.", ""]
    chosen = {}
    for key, by_lr in sorted(groups.items(), key=lambda kv: name(next(iter(kv[1].values()))[0])):
        first = next(iter(by_lr.values()))[0]
        nm = name(first)
        D_in = sorted(first["acc_in_dist"], key=int)
        D_deep = sorted(first["acc_deep"], key=int)
        out += [f"## {nm}, {first['steps']} steps", ""]
        head = ["seed", "params", "lr", "steps", "s/step"] + [f"in {d}" for d in D_in] + [f"two {d}" for d in D_deep] + ["ok"]
        out += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        lr_score = {}
        for lr in sorted(by_lr):
            rs = sorted(by_lr[lr], key=lambda r: r["seed"])
            for r in rs:
                ok = all(r["acc_in_dist"][d] >= 0.99 for d in D_in)
                row = [str(r["seed"]), str(r["params"]), f"{lr:g}", str(r["steps"]), f"{r['train_seconds'] / r['steps']:.4f}"]
                row += [fmt(r["acc_in_dist"][d]) for d in D_in] + [fmt(r["acc_deep"][d]) for d in D_deep] + ["ok" if ok else ""]
                out.append("| " + " | ".join(row) + " |")
            ind = np.array([[r["acc_in_dist"][d] for d in D_in] for r in rs])
            deep = np.array([[r["acc_deep"][d] for d in D_deep] for r in rs])
            n_ok = int(sum(all(r["acc_in_dist"][d] >= 0.99 for d in D_in) for r in rs))
            row = ["mean", "", f"{lr:g}", "", ""] + [fmt(x) for x in ind.mean(0)] + [fmt(x) for x in deep.mean(0)] + [f"{n_ok}/{len(rs)}"]
            out.append("| " + " | ".join(f"**{c}**" if c else "" for c in row) + " |")
            lr_score[lr] = (float(ind.mean()), len(rs))
        best = max(lr_score, key=lambda lr: lr_score[lr][0])
        chosen[key] = (nm, best, by_lr[best])
        out += ["", "Mean in-distribution accuracy by learning rate: "
                + ", ".join(f"{lr:g}: {fmt(s)} ({n} seeds)" for lr, (s, n) in sorted(lr_score.items()))
                + f". Chosen: **{best:g}**.", ""]

        sq = [r for r in by_lr[best] if "acc_deep_by_squarings" in r]
        if sq:
            rr = sorted(sq[0]["acc_deep_by_squarings"], key=int)
            out += [f"Two-chain accuracy by number of squarings at test time (lr {best:g}, mean of {len(sq)} seeds):", "",
                    "| squarings | " + " | ".join(f"two {d}" for d in D_deep) + " | mean 2-8 |", "|" + "---|" * (len(D_deep) + 2)]
            for k in rr:
                a = np.array([[r["acc_deep_by_squarings"][k][d] for d in D_deep] for r in sq]).mean(0)
                out.append(f"| {k} | " + " | ".join(fmt(x) for x in a) + f" | {fmt(a[1:].mean())} |")
            out.append("")
        lp = [r for r in by_lr[best] if "acc_deep_by_loops" in r]
        if lp:
            ll = sorted(lp[0]["acc_deep_by_loops"], key=int)
            out += [f"Two-chain accuracy by number of block applications at test time (lr {best:g}, mean of {len(lp)} seeds):", "",
                    "| applications | " + " | ".join(f"two {d}" for d in D_deep) + " |", "|" + "---|" * (len(D_deep) + 1)]
            for k in ll:
                a = np.array([[r["acc_deep_by_loops"][k][d] for d in D_deep] for r in lp]).mean(0)
                out.append(f"| {k} | " + " | ".join(fmt(x) for x in a) + " |")
            out.append("")

    out += ["## Summary at the chosen learning rates", "",
            "| setting | lr | seeds | in-dist mean | seeds ok | two-chain mean 2-8 | two-chain mean 5-8 | guess 5-8 |", "|---|---|---|---|---|---|---|---|"]
    for key, (nm, lr, rs) in sorted(chosen.items(), key=lambda kv: kv[1][0]):
        D_in = sorted(rs[0]["acc_in_dist"], key=int)
        deep = lambda r, lo, hi: np.mean([r["acc_deep"][str(d)] for d in range(lo, hi + 1) if str(d) in r["acc_deep"]])
        n_ok = sum(all(r["acc_in_dist"][d] >= 0.99 for d in D_in) for r in rs)
        guess = np.mean([np.mean([r["guess_deep"][str(d)] for d in range(5, 9)]) for r in rs])
        out.append(f"| {nm}, {rs[0]['steps']} steps | {lr:g} | {len(rs)} | {fmt(np.mean([np.mean(list(r['acc_in_dist'].values())) for r in rs]))} "
                   f"| {n_ok}/{len(rs)} | {fmt(np.mean([deep(r, 2, 8) for r in rs]))} | {fmt(np.mean([deep(r, 5, 8) for r in rs]))} | {fmt(guess)} |")
    text = "\n".join(out) + "\n"
    if args.out:
        open(args.out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
