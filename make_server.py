#!/usr/bin/env python3
"""Figures for the server runs, in the style of make_fair.py.

usage: python3 make_server.py results/res_server.jsonl --tag e1 --outdir results/figures

For each setting the learning rate is the one chosen by mean in-distribution accuracy (plan section 5,
rule 3), as in summarize.py. Tables with every number are written by summarize.py.
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from make_fair import AQUA, BAND, BLUE, GRAY, INK, INK2, MUTED, ORANGE, SURFACE, line, panel  # noqa: F401  (sets rcParams)
from matplotlib import font_manager

if not any("Inter" in f.name for f in font_manager.fontManager.ttflist):
    plt.rcParams["font.family"] = ["DejaVu Sans"]

BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
             "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def chosen(runs, **match):
    """Runs of one setting at its chosen learning rate."""
    by_lr = defaultdict(list)
    for r in runs:
        if all(r.get(k) == v for k, v in match.items()):
            by_lr[r["lr"]].append(r)
    if not by_lr:
        return None, []
    lr = max(by_lr, key=lambda x: np.mean([np.mean(list(r["acc_in_dist"].values())) for r in by_lr[x]]))
    return lr, sorted(by_lr[lr], key=lambda r: r["seed"])


def series(r, field="acc_deep"):
    xs = sorted(int(k) for k in r[field])
    return xs, [100 * r[field][str(x)] for x in xs]


def ok(r):
    return all(v >= 0.99 for v in r["acc_in_dist"].values())


def fig_seeds(runs, arms, out, title):
    """One panel per arm: the two-chain test for every seed at the chosen learning rate."""
    fig, axes = plt.subplots(1, len(arms), figsize=(4.2 * len(arms), 4.4), dpi=200, sharey=True)
    for ax, (label, color, match) in zip(np.atleast_1d(axes), arms):
        lr, rs = chosen(runs, **match)
        if not rs:
            ax.set_visible(False)
            continue
        panel(ax, f"{label}\nlr {lr:g}, {len(rs)} seeds, {sum(map(ok, rs))} reach 99% in distribution", rs[0]["D_train"])
        xs, guess = series(rs[0], "guess_deep")
        ax.plot(xs, guess, color=GRAY, linewidth=2, solid_capstyle="round", zorder=2, label="Guessing among chain-end digits")
        for i, r in enumerate(rs):
            x, y = series(r)
            line(ax, x, y, color, "One line per seed" if i == 0 else None, lw=1.6, ms=6, z=4, alpha=0.9)
        ax.set_xticks(xs)
        ax.text(0.62, 104.5, "depths seen in training", color=MUTED, fontsize=9, va="bottom")
        ax.legend(loc="upper left", bbox_to_anchor=(-0.01, -0.2), fontsize=9, handlelength=1.6, labelcolor=INK2, borderaxespad=0)
    np.atleast_1d(axes)[0].set_ylabel("Two-chain accuracy")
    fig.suptitle(title, x=0.012, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_squarings(runs, match, out, title):
    """Heatmap: mean two-chain accuracy by test-time squarings (rows) and chain depth (columns)."""
    lr, rs = chosen(runs, **match)
    rs = [r for r in rs if "acc_deep_by_squarings" in r]
    if not rs:
        return
    rr = sorted(rs[0]["acc_deep_by_squarings"], key=int)
    ds = sorted(rs[0]["acc_deep"], key=int)
    A = np.array([[np.mean([100 * r["acc_deep_by_squarings"][k][d] for r in rs]) for d in ds] for k in rr])
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=200)
    ax.grid(False)
    im = ax.imshow(A, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(len(rr)):
        for j in range(len(ds)):
            ax.text(j, i, f"{A[i, j]:.0f}", ha="center", va="center", fontsize=8.5, color="white" if A[i, j] >= 60 else INK)
    ax.set_xticks(range(len(ds)), [str(d) for d in ds])
    ax.set_yticks(range(len(rr)), [f"{k}  ({2 ** int(k)}x)" for k in rr])
    ax.set_xlabel("Chain depth of the query (hops)")
    ax.set_ylabel("Squarings at test time  (multiplications by S)")
    r_lo, r_hi = rs[0]["r_train"]
    ax.axhspan(r_lo - 0.45, r_hi + 0.45, xmin=1.01, xmax=1.025, color=MUTED, clip_on=False)
    ax.text(len(ds) - 0.25, (r_lo + r_hi) / 2, " used in\n training", va="center", fontsize=8.5, color=MUTED)
    r_eval = rs[0]["r_eval"]
    ax.add_patch(plt.Rectangle((-0.5, r_eval - 0.5), len(ds), 1, fill=False, edgecolor=INK, linewidth=1.2))
    ax.axhspan(r_eval - 0.45, r_eval + 0.45, xmin=1.01, xmax=1.025, color=INK, clip_on=False)
    ax.text(len(ds) - 0.25, r_eval, " used for\n results", va="center", fontsize=8.5, color=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.17)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8.5)
    cb.set_label("Two-chain accuracy, %, mean of seeds", fontsize=9, color=INK2)
    ax.set_title(f"lr {lr:g}, mean of {len(rs)} seeds", loc="left", fontsize=10.5, color=INK2, pad=8)
    fig.suptitle(title, x=0.012, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", default="results/figures")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    runs = [r for r in map(json.loads, open(args.path)) if r.get("tag") == args.tag]
    if args.tag.startswith("e1"):
        arms = [("Squared read (proposed)", BLUE, dict(model="tied", read="square")),
                ("Same layer, state read once", ORANGE, dict(model="tied", read="once")),
                ("Same layer, 2^r reads in sequence", AQUA, dict(model="tied", read="iter"))]
        fig_seeds(runs, arms, f"{args.outdir}/fig_{args.tag}_seeds.png",
                  "Experiment 1: one layer, only the read differs. Two-chain test, trained on depth 1 to 4")
        fig_squarings(runs, dict(model="tied", read="square"), f"{args.outdir}/fig_{args.tag}_squarings.png",
                      "Experiment 1: squared read, accuracy by number of squarings at test time")
    print("figures in", args.outdir)


if __name__ == "__main__":
    main()
